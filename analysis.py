import pdb
from functools import partial
import pandas as pd
from PIL import Image
import numpy as np
import torchvision as tv
import torch
import tqdm
import matplotlib.pyplot as plt
import skimage
import os
import sys
import torch.nn.functional as F
import pydicom

sys.path.append('../torchxrayvision/')
import torchxrayvision as xrv

from data_utils import create_cxp_view_column, get_mimic_jpg_path, \
    load_split_metadf, apply_window, create_mimic_isportable_column, get_mimic_dcm_path
from constants import PROJECT_DIR, R_LABELS, CXP_JPG_DIR, MIMIC_JPG_DIR, MODEL_SAVE_DIR, CXP_LABELS, MIMIC_DCM_DIR
from dicom_analysis import window_dcm_im


def simple_preprocess_image(image_path):
    img = Image.open(image_path)

    transform_test = tv.transforms.Compose([
        tv.transforms.Resize(224),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
    ])

    return transform_test(img).squeeze()


def xrv_preprocess(image_path, final_resize=224, window_width=None, init_resize=None):
    is_dicom = '.dcm' in image_path
    if is_dicom:
        ds = pydicom.dcmread(image_path)
        if window_width is None:
            img = ds.pixel_array
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                max_num = 2 ** ds.BitsStored - 1
                img = max_num - img
        else:
            img = window_dcm_im(ds, width_mult=window_width)
    else:
        if isinstance(image_path, np.ndarray):
            img = image_path
        else:
            img = skimage.io.imread(image_path)

        if window_width and window_width != 256:
            img = apply_window(img, 256. / 2, window_width, y_min=0, y_max=255)

    if init_resize is None:
        init_resize = final_resize

    maxval = np.max(img) if is_dicom else 255
    img = xrv.datasets.normalize(img, maxval=maxval, reshape=True)

    transforms = [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(init_resize)]
    transform = tv.transforms.Compose(transforms)
    img = transform(img)

    if init_resize != final_resize:
        diff = int((init_resize - final_resize) / 2)
        img = img[:, diff:(diff + final_resize), diff:(diff + final_resize)]

    return img


def compute_mean_image_over_files(files, img_proc_fxn):
    count = 0
    av_im = None

    for f in tqdm.tqdm(files):
        try:
            img = img_proc_fxn(f)
        except:
            continue
        if count:
            av_im = av_im * count / (count + 1) + img * 1 / (count + 1)
        else:
            av_im = img
        count += 1

    return av_im


def plot_and_save_image(im, out_path, size=(4, 4), dpi=80):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('gray')
    ax.imshow(im, aspect='equal')
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def run_compute_mean_image(dataset, split='train'):
    df = load_split_metadf(dataset, split)
    df = df[df.Mapped_Race.isin(R_LABELS)]

    if dataset == 'cxp':
        df['file_path'] = [CXP_JPG_DIR + v for v in df.Path.values]
    else:
        df['file_path'] = df.apply(partial(get_mimic_jpg_path, small=True), axis=1)

    out_dir = PROJECT_DIR + 'mean_images/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    base_save_name = out_dir + dataset + '_' + split

    for r in R_LABELS:
        files = df.loc[df.Mapped_Race == r, 'file_path'].values
        av_im = compute_mean_image_over_files(files, simple_preprocess_image)

        out_path = base_save_name + '-' + r
        plot_and_save_image(av_im, out_path + '.png')
        np.save(out_path + '.npy', av_im)


def load_model(model_name, checkpoint_name, num_classes):
    model = xrv.models.DenseNet(num_classes=num_classes, in_channels=1,
                                **xrv.models.get_densenet_params('densenet')).cuda()

    model_pref = '{}-densenet'.format('chex' if 'cxp' in model_name else 'mimic_ch') # auto added by xrv
    weights_path = os.path.join(MODEL_SAVE_DIR, model_name, '{}-{}-{}.pt'.format(model_pref, model_name, checkpoint_name))
    model.load_state_dict(torch.load(weights_path).state_dict())

    model.eval()
    return model


def run_predictions(model_name, checkpoint_name, dataset, split='test', window_width=None, init_resize=None,
                    label_type='race', filt_mapped_race=True, confounder_resampled=False,
                    use_dicoms=False):
    if label_type == 'race':
        labels = R_LABELS
    elif label_type == 'pathology':
        labels = xrv.datasets.default_pathologies
    elif label_type == 'pathology_wnofinding':
        labels = CXP_LABELS
    model = load_model(model_name, checkpoint_name, len(labels))

    df = load_split_metadf(dataset, split, confounder_resampled=confounder_resampled)
    if filt_mapped_race:
        df = df[df.Mapped_Race.isin(R_LABELS)]
    if dataset == 'cxp':
        df['file_path'] = [CXP_JPG_DIR + v for v in df.Path.values]
        df = create_cxp_view_column(df)
    else:
        if use_dicoms:
            df['file_path'] = df.apply(partial(get_mimic_dcm_path), axis=1)
        else:
            df['file_path'] = df.apply(partial(get_mimic_jpg_path, small=True), axis=1)
        df['view'] = df.ViewPosition

    im_proc_fxn = partial(xrv_preprocess, window_width=window_width, init_resize=init_resize)

    pred_data = []
    with torch.no_grad():
        for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            try:
                x = torch.from_numpy(im_proc_fxn(row['file_path'])).unsqueeze(0).cuda()
                if label_type == 'race':
                    preds = F.softmax(model(x), dim=-1).cpu().squeeze().numpy()
                else:
                    preds = torch.sigmoid(model(x)).cpu().squeeze().numpy()
                pred_data.append([row['file_path'], row['Mapped_Race'], row['view']] + preds.tolist())
            except:
                pred_data.append([row['file_path'], row['Mapped_Race'], row['view']] + [np.nan]*len(labels))

    pred_df = pd.DataFrame(pred_data, columns=['Path', 'Mapped_Race', 'View'] + ['Pred_' + r for r in labels])
    out_dir = PROJECT_DIR + 'prediction_dfs/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir += model_name + '-' + checkpoint_name + '/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_name = dataset + '-' + split
    if confounder_resampled:
        out_name += '-confounder_resampled'
    if window_width:
        out_name += '-window{}'.format(window_width)
    if init_resize:
        out_name += '-initresize{}'.format(init_resize)
    if use_dicoms:
        out_name += '_dicoms'
    pred_df.to_csv(out_dir + out_name + '.csv')


def analyze_mean_predictions(model_name, checkpoint_name, dataset, split='test',
                             confounder_resampled=False, use_dicoms=False, bmi=False):
    # compute mean prediction score per race over different image permutations
    if use_dicoms:
        add_window_widths = [0.95, 0.9, 0.85, 0.8]
        default_width = 1
    else:
        add_window_widths = [205, 218, 230, 243]
        default_width = None
    add_init_resizes = [235, 246, 258, 269]
    views = ['AP', 'PA', ('LATERAL', 'LL')]

    base_df_dir = PROJECT_DIR + 'prediction_dfs/' + model_name + '-' + checkpoint_name + '/'
    def get_pred_df_path(w, rz):
        out_name = dataset + '-' + split
        if confounder_resampled:
            out_name += '-confounder_resampled'
        if w:
            out_name += '-window{}'.format(w)
        if rz:
            out_name += '-initresize{}'.format(rz)
        if use_dicoms:
            out_name += '_dicoms'
        return base_df_dir + out_name + '.csv'

    all_pred_dfs = {}
    all_pred_dfs['default'] = pd.read_csv(get_pred_df_path(default_width, None))

    for window_width in add_window_widths:
        all_pred_dfs['w_{}'.format(window_width)] = pd.read_csv(get_pred_df_path(window_width, None))

    for init_resize in add_init_resizes:
        all_pred_dfs['rz_{}'.format(init_resize)] = pd.read_csv(get_pred_df_path(default_width, init_resize))

    for window_width in add_window_widths:
        for init_resize in add_init_resizes:
            all_pred_dfs[f'w_{window_width}_rz{init_resize}'] = pd.read_csv(get_pred_df_path(window_width, init_resize))

    cols = [('default', None)] + [('w', w) for w in add_window_widths] + [('rz', rz) for rz in add_init_resizes] + [('v', v) for v in views]
    for window_width in add_window_widths:
        for init_resize in add_init_resizes:
            cols.append(('wrz', (window_width, init_resize)))

    if dataset == 'mimic':
        cols += [('vp', 'AP_port'), ('vp', 'AP_nonport')]
        meta_df = load_split_metadf(dataset, split)
        meta_df = create_mimic_isportable_column(meta_df)
        portable_map = meta_df[['is_portable', 'dicom_id']].set_index('dicom_id')['is_portable']
        for k in all_pred_dfs:
            all_pred_dfs[k]['dicom_id'] = [p.split('/')[-1][:-4] for p in all_pred_dfs[k]['Path'].values]
            assert pd.isnull(all_pred_dfs[k]['dicom_id']).sum() == 0
            all_pred_dfs[k]['is_portable'] = all_pred_dfs[k]['dicom_id'].map(portable_map)
            assert pd.isnull(all_pred_dfs[k]['is_portable']).sum() == 0
            assert all_pred_dfs[k]['is_portable'].sum() > 0

        if bmi:
            resampled_df = load_split_metadf(dataset, split, confounder_resampled='bmi')
            for k in all_pred_dfs:
                all_pred_dfs[k].set_index('dicom_id', inplace=True)
                all_pred_dfs[k] = all_pred_dfs[k].loc[resampled_df.dicom_id]
                all_pred_dfs[k].reset_index(inplace=True)

    rows = ['weighted_{}'.format(r) for r in R_LABELS]
    mean_scores = np.zeros((len(rows), len(cols)))
    counts = np.zeros((len(rows), len(cols)))
    for j, c in enumerate(cols):
        if c[0] in ['default', 'v', 'vp']:
            this_df = all_pred_dfs['default']
            if c[0] == 'v':
                if isinstance(c[1], tuple):
                    this_df = this_df[this_df.View.isin(c[1])]
                else:
                    this_df = this_df[this_df.View == c[1]]
            elif c[0] == 'vp':
                this_view = c[1].split('_')[0]
                is_portable = c[1].split('_')[1] == 'port'
                this_df = this_df[(this_df.View == this_view) & (this_df.is_portable == is_portable)]
        elif c[0] == 'w':
            this_df = all_pred_dfs['w_{}'.format(c[1])]
        elif c[0] == 'rz':
            this_df = all_pred_dfs['rz_{}'.format(c[1])]
        elif c[0] == 'wrz':
            this_df = all_pred_dfs[f'w_{c[1][0]}_rz{c[1][1]}']

        ex_pred_col = [pcol for pcol in this_df.columns if 'Pred_' in pcol][0]
        idx = pd.isnull(this_df[ex_pred_col])
        if use_dicoms: #nans allowed b/c some dicom elements are missing in some files
            this_df = this_df.loc[~idx]
            print('nans count and %, and remaining: ', idx.sum(), 100 * idx.mean(), len(this_df))
        else:
            assert idx.sum() == 0

        for i, r in enumerate(rows):
            score_r = r.split('_')[1]
            if 'weighted' in r:
                means = this_df.groupby('Mapped_Race')[f'Pred_{score_r}'].mean()
                mean_val = means.mean()
            else:
                raise ValueError()
            mean_scores[i, j] = mean_val
            counts[i, j] = len(vals)

    out_dir = PROJECT_DIR + 'factor_correlation_analysis/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    mean_df = pd.DataFrame(mean_scores, columns=cols, index=rows)
    count_df = pd.DataFrame(counts, columns=cols, index=rows)
    base_name = '{}_{}_{}_{}'.format(model_name, checkpoint_name, dataset, split)

    if confounder_resampled:
        base_name += '-confounder_resampled'

    if bmi:
        assert dataset == 'mimic'
        base_name += '-bmi_resampled'

    if use_dicoms:
        base_name += '_dicoms'

    mean_df.to_csv(out_dir + base_name + '-mean_scores.csv')
    count_df.to_csv(out_dir + base_name + '-counts.csv')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model_name = 'cxp_race_orig'
    checkpoint_name = 'best'
    dataset = 'cxp'
    for window_width in [None, 205, 218, 230, 243]: # increments of 5%
        for init_resize in [None, 235, 246, 258, 269]:
            run_predictions(model_name, checkpoint_name, dataset, split='test', window_width=window_width,
                            init_resize=init_resize, confounder_resampled=True)
