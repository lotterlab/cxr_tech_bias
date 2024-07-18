import pandas as pd
import pdb
import sklearn
import numpy as np
import random
import os
import tqdm
import skimage
import json
import pickle

import sys
sys.path.append('../torchxrayvision/')
import torchxrayvision as xrv

from constants import PROJECT_DIR, MIMIC_JPG_DIR, MIMIC_JPG_DIR_SMALL, CXP_LABELS, R_LABELS, MIMIC_DCM_DIR, MIMIC_BASE_DIR


def create_cxp_splits(cv_version, props=(0.7, 0.1, 0.2), rand_seed=0):
    out_dir = PROJECT_DIR + 'cxp_cv_splits/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir += 'version_{}/'.format(cv_version)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    orig_train_df = pd.read_csv('../../../datasets/CheXpert-v1.0-small/train-with_race.csv')

    np.random.seed(rand_seed)
    random.seed(rand_seed)

    u_pats = np.random.permutation(orig_train_df.Patient.unique())

    count = 0
    for p, split in zip(props, ['train', 'val', 'test']):
        if split == 'test':
            next_count = len(u_pats)
        else:
            next_count = int(count + len(u_pats) * p)

        split_pats = u_pats[count:next_count]
        this_df = orig_train_df[orig_train_df.Patient.isin(split_pats)]
        print(split, count, next_count, len(u_pats), len(this_df))
        this_df.to_csv(out_dir + split + '.csv', index=False)

        count = next_count


def get_mimic_jpg_path(row, small=False):
    p = str(row['subject_id'])
    s = str(row['study_id'])
    d = row['dicom_id']
    if small:
        base_dir = MIMIC_JPG_DIR_SMALL
    else:
        base_dir = MIMIC_JPG_DIR
    return base_dir + 'p{}/p{}/s{}/{}.{}'.format(p[:2], p, s, d, 'png' if small else 'jpg')


def get_mimic_dcm_path(row):
    p = str(row['subject_id'])
    s = str(row['study_id'])
    d = row['dicom_id']
    return MIMIC_DCM_DIR + 'p{}/p{}/s{}/{}.dcm'.format(p[:2], p, s, d)


def load_split_metadf(dataset, split, only_good_files=True, confounder_resampled=False):
    base_cv_dir = PROJECT_DIR + '{}_cv_splits/version_0/'.format(dataset)
    if dataset == 'cxp':
        fname = base_cv_dir + split
    elif dataset == 'mimic':
        fname = base_cv_dir + 'meta_' + split
    if confounder_resampled:
        fname += '-confounder_resampled'
        if confounder_resampled == 'bmi':
            fname = fname.replace('-confounder', '-bmi')
    fname += '.csv'

    df = pd.read_csv(fname)
    if dataset == 'mimic' and only_good_files:
        good_dicoms = np.load(
            MIMIC_BASE_DIR + 'good_image_dicoms.npy')  # excluding a handful of files that are corrupted in the MXR dataset
        df = df[df.dicom_id.isin(good_dicoms)].copy()
    return df


def create_mimic_splits(cv_version, props=(0.7, 0.1, 0.2), rand_seed=0):
    out_dir = PROJECT_DIR + 'mimic_cv_splits/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir += 'version_{}/'.format(cv_version)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    meta_df = pd.read_csv(MIMIC_BASE_DIR + 'mimic-cxr-2.0.0-metadata-with_race.csv')
    label_df = pd.read_csv(MIMIC_BASE_DIR + 'mimic-cxr-2.0.0-chexpert.csv')

    np.random.seed(rand_seed)
    random.seed(rand_seed)

    u_pats = np.random.permutation(meta_df.subject_id.unique())

    count = 0
    for p, split in zip(props, ['train', 'val', 'test']):
        if split == 'test':
            next_count = len(u_pats)
        else:
            next_count = int(count + len(u_pats) * p)

        split_pats = u_pats[count:next_count]
        this_meta_df = meta_df[meta_df.subject_id.isin(split_pats)]
        this_label_df = label_df[label_df.subject_id.isin(split_pats)]
        print(split, count, next_count, len(u_pats), len(this_meta_df), len(this_label_df))
        this_meta_df.to_csv(out_dir + 'meta_' + split + '.csv', index=False)
        this_label_df.to_csv(out_dir + 'cxp-labels_' + split + '.csv', index=False)

        count = next_count


def create_cxp_view_column(df):
    df['view'] = df['AP/PA'].copy()
    df.loc[df['Frontal/Lateral'] == 'Lateral', 'view'] = 'LATERAL'
    return df


def create_mimic_isportable_column(df):
    df['is_portable'] = ['port' in str(v).lower() for v in df.PerformedProcedureStepDescription.values]
    return df


def apply_window(arr, center, width, y_min=0, y_max=255):
    y_range = y_max - y_min
    arr = arr.astype('float64')
    width = float(width)

    below = arr <= (center - width / 2)
    above = arr > (center + width / 2)
    between = np.logical_and(~below, ~above)

    arr[below] = y_min
    arr[above] = y_max
    if between.any():
        arr[between] = (
                ((arr[between] - center) / width + 0.5) * y_range + y_min
        )

    return arr


def create_small_mimic(sort_version=None):
    resize_shape = 300.

    out_base_dir = MIMIC_JPG_DIR_SMALL
    if not os.path.exists(out_base_dir):
        os.mkdir(out_base_dir)

    meta_df = pd.read_csv(MIMIC_BASE_DIR + 'mimic-cxr-2.0.0-metadata.csv')
    all_orig_files = meta_df.apply(get_mimic_jpg_path, axis=1).values

    if sort_version == 'reverse':
        all_orig_files = np.flipud(all_orig_files)
    elif sort_version == 'random':
        all_orig_files = np.random.permutation(all_orig_files)
    bad_files = []
    for f in tqdm.tqdm(all_orig_files):
        save_name = f.replace('2.0.0/files', '2.0.0/files_small').replace('.jpg', '.png')
        if os.path.exists(save_name):
            continue

        try:
            im = skimage.io.imread(f)
        except ValueError:
            print('bad file: {}'.format(f))
            bad_files.append(f)
            continue

        assert im.ndim == 2
        if np.min(im.shape) > resize_shape:
            rescale_factor = resize_shape / np.min(im.shape)
            im = skimage.transform.rescale(im, rescale_factor, mode='constant', preserve_range=True)
            im = np.round(im).astype(np.uint8)

        f_out_dir = save_name[:save_name.rfind('/')]
        if not os.path.exists(f_out_dir):
            os.makedirs(f_out_dir)

        skimage.io.imsave(save_name, im)

    bad_out_file = MIMIC_BASE_DIR + 'bad_image_files.npy'
    if os.path.exists(bad_out_file):
        bad_out_file.replace('.npy', '1.npy')
    print('number of bad files:', len(bad_files))
    np.save(bad_out_file, np.array(bad_files))


def create_mimic_good_file_list():
    # create list of non-corrupted MIMIC files (99.7% of files)
    good_dicoms = []
    meta_df = pd.read_csv(
        MIMIC_BASE_DIR + 'mimic-cxr-2.0.0-metadata.csv')
    meta_df['file_path'] = meta_df.apply(get_mimic_jpg_path, axis=1).values
    for idx, row in tqdm.tqdm(meta_df.iterrows(), total=len(meta_df)):
        save_name = row['file_path'].replace('2.0.0/files', '2.0.0/files_small').replace('.jpg', '.png')
        if os.path.exists(save_name):
            good_dicoms.append(row['dicom_id'])

    good_out_file = MIMIC_BASE_DIR + 'good_image_dicoms.npy'
    np.save(good_out_file, np.array(good_dicoms))
    print('# files', len(good_dicoms))


def create_confounder_data(dataset, n_digits=4, rounding_tol=0.001):
    # create dataframe of values per each confounding factor for each files
    if dataset == 'cxp':
        file_df = pd.read_csv('../../../datasets/CheXpert-v1.0-small/train-with_race.csv')
        file_df.set_index('Path', inplace=True)
        is_male = file_df['Sex'] == 'Male'
        file_df.loc[is_male, 'Sex'] = 'M'
        file_df.loc[~is_male, 'Sex'] = 'F'
    elif dataset == 'mimic':
        meta_df = pd.read_csv(
            MIMIC_BASE_DIR + 'mimic-cxr-2.0.0-metadata-with_race.csv')
        label_df = pd.read_csv(
            MIMIC_BASE_DIR + 'mimic-cxr-2.0.0-chexpert.csv')
        patients_df = pd.read_csv(
            MIMIC_BASE_DIR + 'physionet.org/files/mimiciv/2.1/hosp/patients.csv')

        for c in ['gender', 'anchor_year', 'anchor_age']:
            c_map = patients_df[['subject_id', c]].set_index('subject_id')[c]
            meta_df[c.replace('gender', 'Sex')] = meta_df['subject_id'].map(c_map)

        meta_df['StudyYear'] = [int(str(v)[:4]) for v in meta_df.StudyDate.values]

        meta_df['Age'] = meta_df['StudyYear'] - meta_df['anchor_year'] + meta_df['anchor_age']

        file_df = pd.merge(meta_df, label_df, how='left', on='study_id')
        file_df.set_index('dicom_id', inplace=True)

    cols = ['Age', 'Sex', 'Mapped_Race'] + CXP_LABELS
    file_df = file_df[cols]

    # curate labels
    no_findings_idx = file_df['No Finding'] == 1
    findings_cols = [c for c in CXP_LABELS if c not in ['No Finding', 'Support Devices']]
    assert len(findings_cols) == (len(CXP_LABELS) - 2)
    has_findings = (file_df[findings_cols] == 1).sum(axis=1) > 0
    nan_no_finding = pd.isnull(file_df['No Finding'])
    fill_idx = has_findings & nan_no_finding
    if fill_idx.sum():
        print('filling no finding for ', fill_idx.sum())
        file_df.loc[fill_idx, 'No Finding'] = 0

    for pathology in findings_cols:  # if dont have findings, set path columns to 0
        file_df.loc[no_findings_idx, pathology] = 0

    for pathology in CXP_LABELS:  # set to binary for this analysis
        file_df[pathology] = (file_df[pathology] == 1).astype(int)

    # bin age
    def bin_age(age):
        if age != age:
            return np.nan
        for age_limit in [30, 40, 50, 60, 70, 80]:
            if age <= age_limit:
                return age_limit
        return 81

    file_df['Age_Bin'] = file_df['Age'].apply(bin_age)

    out_dir = '../torchxrayvision/data/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # check remaining nans
    for c in ['Age', 'Age_Bin', 'Sex', 'Mapped_Race']:
        print(c, '% nan:', 100*pd.isnull(file_df[c]).mean())

    file_df.to_csv(out_dir + f'{dataset}_confounder_df.csv')

    for subset in ['all', 'ABW']:
        agg_proportions = {}
        cols_to_use = ['Age_Bin', 'Sex', 'Mapped_Race'] + CXP_LABELS
        if subset == 'all':
            this_df = file_df
        elif subset == 'ABW':
            this_df = file_df[file_df.Mapped_Race.isin(R_LABELS)]
        else:
            raise ValueError()

        for c in cols_to_use:
            val_props = this_df[c].value_counts(normalize=True).to_dict()
            # proportions still work with nans (they're just not included in counts)

            print(subset, c, np.sum(list(val_props.values())))

            # normalize to exactly sum to 1 (for later sampling)
            new_vals = val_props.copy()
            running_sum = 0
            for val_i, v in enumerate(new_vals.keys()):
                if val_i == (len(new_vals) - 1):
                    new_vals[v] = round(1 - running_sum, n_digits)
                else:
                    new_vals[v] = round(new_vals[v], n_digits)
                    running_sum += new_vals[v]

            assert len(new_vals) == len(val_props)
            assert np.abs(np.sum(list(new_vals.values())) - 1) < rounding_tol
            for v in new_vals:
                assert np.abs(new_vals[v] - val_props[v]) < rounding_tol # tolerance after rounding

            agg_proportions[c] = new_vals

        with open(out_dir + f'{dataset}_confounder_proportions-{subset}.json', 'w') as f:
            json.dump(agg_proportions, f)

        with open(out_dir + f'{dataset}_confounder_proportions-{subset}.pkl', 'wb') as f:
            pickle.dump(agg_proportions, f)


def create_confounder_controlled_split(dataset, split, sample_mult_factor=3, bmi=False):
    orig_df = load_split_metadf(dataset, split, only_good_files=True)
    orig_df = orig_df[orig_df.Mapped_Race.isin(R_LABELS)]

    class_balancing_labels_df = pd.read_csv(f'../torchxrayvision/data/{dataset}_confounder_df.csv', index_col=0)
    with open(f'../torchxrayvision/data/{dataset}_confounder_proportions-ABW.pkl', 'rb') as f:
        all_props = pickle.load(f)

    class_balancing_props = []
    class_balancing_props.append(('Mapped_Race', ['Asian', 'Black', 'White']))
    if bmi:
        class_balancing_props.append(('BMI_Bin', all_props['BMI_Bin']))
    else:
        for confounder in ['Age_Bin', 'Sex']:
            class_balancing_props.append((confounder, all_props[confounder]))

        path_props = {}
        for pathology in CXP_LABELS:
            path_props[pathology] = all_props[pathology]
        class_balancing_props.append(('pathology', path_props))

    # from datasets
    if dataset == 'cxp':
        filt_labels_df = class_balancing_labels_df.loc[orig_df.Path]
    elif dataset == 'mimic':
        filt_labels_df = class_balancing_labels_df.loc[orig_df.dicom_id]
    idxs_per_sample_key = xrv.datasets.calculate_multifactorial_sample_idxs(filt_labels_df, class_balancing_props)

    n_target = len(orig_df) * sample_mult_factor
    sampled_idxs = np.zeros(n_target, int)
    for i in range(n_target):
        sampled_idxs[i] = xrv.datasets.sample_idx_multifactorial(class_balancing_props, idxs_per_sample_key)

    new_df = orig_df.iloc[sampled_idxs]

    base_cv_dir = PROJECT_DIR + '{}_cv_splits/version_0/'.format(dataset)
    if dataset == 'cxp':
        fname = base_cv_dir + split + '-confounder_resampled.csv'
    elif dataset == 'mimic':
        fname = base_cv_dir + 'meta_' + split + '-confounder_resampled.csv'
    if bmi:
        fname = fname.replace('-confounder', '-bmi')
    new_df.to_csv(fname, index=False)


def create_mimic_bmi_df(path_to_chart_csv):
    # based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts_postgres/measurement/height.sql
    # and https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts_postgres/demographics/weight_durations.sql
    chart_df = pd.read_csv(path_to_chart_csv)
    h_df = chart_df[chart_df.itemid == 226730]  # in cm
    w_df = chart_df[chart_df.itemid == 226512]  # in kg
    assert pd.isnull(h_df.valuenum).sum() == 0
    assert pd.isnull(w_df.valuenum).sum() == 0
    idx = (h_df.valuenum > 120) & (h_df.valuenum < 230)  # clean
    h_df = h_df.loc[idx]
    idx = (w_df.valuenum > 35) & (w_df.valuenum < 225)  # clean
    w_df = w_df.loc[idx]
    h_map = h_df.groupby('subject_id')['valuenum'].mean()
    w_map = w_df.groupby('subject_id')['valuenum'].mean()
    meta_df = pd.read_csv(MIMIC_BASE_DIR + 'mimic-cxr-2.0.0-metadata-with_race.csv')
    meta_df['height'] = meta_df.subject_id.map(h_map)
    meta_df['weight'] = meta_df.subject_id.map(w_map)
    meta_df['bmi'] = meta_df['weight'] / ((meta_df['height'] / 100) ** 2)
    meta_df.to_csv(MIMIC_BASE_DIR + 'mimic-cxr-2.0.0-metadata-with_race_and_bmi.csv')

    for c in ['height', 'weight', 'bmi']:
        print(c)
        print('% null', 100*pd.isnull(meta_df[c]).mean())
        print('describe:')
        print(meta_df[c].describe())


def append_mimic_bmi_to_confounder_data():
    dataset = 'mimic'
    out_dir = '../torchxrayvision/data/'
    n_digits = 4
    rounding_tol = 0.001

    bmi_df = pd.read_csv(MIMIC_BASE_DIR + 'mimic-cxr-2.0.0-metadata-with_race_and_bmi.csv')
    confounder_df = pd.read_csv(out_dir + f'{dataset}_confounder_df.csv', index_col=0)
    assert (bmi_df['dicom_id'] == confounder_df.index).mean() == 1

    bmi_df.set_index('dicom_id', inplace=True)
    for c in ['height', 'weight', 'bmi']:
        confounder_df[c] = confounder_df.index.map(bmi_df[c])

    def bin_bmi(bmi):
        if bmi != bmi:
            return np.nan
        for bin_info in [(1, 18.5), (2, 25), (3, 30)]:
            if bmi <= bin_info[1]:
                return bin_info[0]
        return 4

    confounder_df['BMI_Bin'] = confounder_df['bmi'].apply(bin_bmi)

    print('BMI Bin distribution:')
    print(confounder_df['BMI_Bin'].value_counts(normalize=True))

    for subset in ['all', 'ABW']:
        with open(out_dir + f'{dataset}_confounder_proportions-{subset}.pkl', 'rb') as f:
            agg_proportions = pickle.load(f)

        if subset == 'all':
            this_df = confounder_df
        elif subset == 'ABW':
            this_df = confounder_df[confounder_df.Mapped_Race.isin(R_LABELS)]

        for c in ['BMI_Bin']:
            val_props = this_df[c].value_counts(normalize=True).to_dict()
            # proportions still work with nans (they're just not included in counts)

            print(subset, c, np.sum(list(val_props.values())))

            # normalize to exactly sum to 1 (for later sampling)
            new_vals = val_props.copy()
            running_sum = 0
            for val_i, v in enumerate(new_vals.keys()):
                if val_i == (len(new_vals) - 1):
                    new_vals[v] = round(1 - running_sum, n_digits)
                else:
                    new_vals[v] = round(new_vals[v], n_digits)
                    running_sum += new_vals[v]

            assert len(new_vals) == len(val_props)
            assert np.abs(np.sum(list(new_vals.values())) - 1) < rounding_tol
            for v in new_vals:
                assert np.abs(new_vals[v] - val_props[v]) < rounding_tol  # tolerance after rounding

            agg_proportions[c] = new_vals

            with open(out_dir + f'{dataset}_confounder_proportions-{subset}.json', 'w') as f:
                json.dump(agg_proportions, f)

            with open(out_dir + f'{dataset}_confounder_proportions-{subset}.pkl', 'wb') as f:
                pickle.dump(agg_proportions, f)

    confounder_df.to_csv(out_dir + f'{dataset}_confounder_df.csv')


if __name__ == '__main__':
    #create_cxp_splits(0)
    #create_mimic_splits(0)
    #create_small_mimic(None)
    #create_mimic_good_file_list()
    #create_confounder_data('mimic')
    #create_confounder_controlled_split('mimic', 'test', sample_mult_factor=3)
    #create_mimic_bmi_df('../../../datasets/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.0.0/physionet.org/files/mimiciv/2.1/icu/chartevents.csv')
    #append_mimic_bmi_to_confounder_data()
    create_confounder_controlled_split('mimic', 'test', sample_mult_factor=3, bmi=True)
