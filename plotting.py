import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import pickle as pkl

from constants import PROJECT_DIR, R_LABELS
from data_utils import load_split_metadf, create_cxp_view_column, create_mimic_isportable_column
from results import load_pred_df


def create_full_range_im(orig_im, max_val, min_val=None):
    if min_val is None:
        min_val = -1 * max_val
    new_im = np.zeros((orig_im.shape[0], orig_im.shape[1] + 1))
    new_im[:, :-1] = orig_im
    new_im[0, -1] = max_val
    new_im[1, -1] = min_val
    return new_im


def create_window_resize_heatmap_plot(model_name, checkpoint_name, dataset, split='test',
                                      confounder_resampled=False, use_dicoms=False):
    base_dir = PROJECT_DIR + 'factor_correlation_analysis/'
    if confounder_resampled:
        if confounder_resampled == 'bmi':
            con_tag = '-bmi_resampled'
        else:
            con_tag = '-confounder_resampled'
    else:
        con_tag = ''
    dicom_tag = '_dicoms' if use_dicoms else ''
    fname = f'{model_name}_{checkpoint_name}_{dataset}_{split}{con_tag}{dicom_tag}-mean_scores.csv'
    df_path = os.path.join(base_dir, fname)
    df = pd.read_csv(df_path, index_col=0)
    print('loading: ' + df_path)

    if use_dicoms:
        windows = [1, 0.95, 0.9, 0.85, 0.8]
    else:
        windows = [256, 243, 230, 218, 205]
    resizes = [224, 235, 246, 258, 269]

    orig_scores = {}
    for r in R_LABELS:
        orig_scores[r] = df.loc[f'weighted_{r}', "('default', None)"]

    score_mats = {}
    for r in R_LABELS:
        score_mats[r] = np.zeros((len(windows), len(resizes)))
        for i, w in enumerate(windows):
            for j, rz in enumerate(resizes):
                if i == 0 and j == 0:
                    continue
                else:
                    if i == 0:
                        col = f"('rz', {rz})"
                    elif j == 0:
                        col = f"('w', {w})"
                    else:
                        col = f"('wrz', ({w}, {rz}))"
                this_val = df.loc[f'weighted_{r}', col]
                score_mats[r][i, j] = 100 * (this_val - orig_scores[r]) / orig_scores[r]  # percent change

    max_abs_change = 0
    for r in R_LABELS:
        this_abs = np.max([score_mats[r].max(), np.abs(score_mats[r].min())])
        if this_abs > max_abs_change:
            max_abs_change = this_abs

    out_dir = base_dir + 'plots/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir += '{}-{}-{}-{}/'.format(model_name, checkpoint_name, dataset, split)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    pkl_out_path = os.path.join(out_dir, f'window_width_x_size_score_change{con_tag}{dicom_tag}-data.pkl')
    with open(pkl_out_path, 'wb') as f:
        pkl.dump(score_mats, f)

    ax_labels = [0, 5, 10, 15, 20] # percent change
    for r in R_LABELS:
        plt.figure()
        new_mat = np.zeros((len(windows), len(resizes) + 1))
        new_mat[:, :-1] = score_mats[r]
        new_mat[0, -1] = max_abs_change
        new_mat[1, -1] = -1 * max_abs_change
        plt.imshow(np.flipud(new_mat), cmap='bwr')
        plt.colorbar()
        plt.xlim(-0.5, len(resizes) - 0.5)
        plt.xticks(range(len(resizes)), labels=ax_labels)
        plt.yticks(range(len(windows)), labels=np.flipud(ax_labels))
        plt.xlabel('Size Crop Change (%)', fontweight='bold')
        plt.ylabel('Window Width Change (%)', fontweight='bold')
        plt.title(f'{r} Patients', fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        out_path = os.path.join(out_dir, f'window_width_x_size_score_change{con_tag}{dicom_tag}-{r}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close()


def create_aggregate_window_resize_heatmap_plot(model_tag, version):
    base_dir = PROJECT_DIR + 'factor_correlation_analysis/'

    if version == 'orig':
        tags = ['cxp-orig', 'mimic-orig']
    elif version == 'mimic_all':
        tags = ['mimic-orig', 'mimic-test_resampled', 'mimic-tr_resampled', 'mimic-dicom']
    elif version == 'cxp_all':
        tags = ['cxp-orig', 'cxp-test_resampled', 'cxp-tr_resampled']
    elif version == 'mimic_allv3':
        tags = ['mimic-orig', 'mimic-test_resampled', 'mimic-tr_resampled', 'mimic-dicom', 'mimic-bmi_tr_resampled']

    n_rows = len(tags)

    all_score_mats = {}
    for tag in tags:
        vals = tag.split('-')
        dataset_name = vals[0]
        if vals[1] == 'orig':
            con_tag = ''
            dicom_tag = ''
        elif 'resampled' in vals[1]:
            if 'bmi' in vals[1]:
                con_tag = '-bmi_resampled'
            else:
                con_tag = '-confounder_resampled'
            dicom_tag = ''
        elif vals[1] == 'dicom':
            dicom_tag = '_dicoms'
            con_tag = ''
        else:
            raise ValueError()

        if vals[1] == 'tr_resampled':
            model_name = dataset_name + model_tag + '_multifact-class-balance'
        elif vals[1] == 'bmi_tr_resampled':
            model_name = dataset_name + model_tag + '_BMI-class-balance'
        else:
            model_name = dataset_name + model_tag
        this_dir = base_dir + 'plots/' + '{}-{}-{}-{}/'.format(model_name, 'best', dataset_name, 'test')
        fname = f'window_width_x_size_score_change{con_tag}{dicom_tag}-data.pkl'
        print('loading: ', this_dir + fname)
        with open(this_dir + fname, 'rb') as f:
            all_score_mats[tag] = pkl.load(f)

    max_abs_change = 0
    for d in all_score_mats:
        for r in R_LABELS:
            this_abs = np.max([all_score_mats[d][r].max(), np.abs(all_score_mats[d][r].min())])
            if this_abs > max_abs_change:
                max_abs_change = this_abs

    ax_labels = [0, 5, 10, 15, 20]  # percent change
    n_levels = len(ax_labels)
    if n_rows == 2:
        fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    elif n_rows == 3:
        fig, axes = plt.subplots(3, 3, figsize=(10, 10.5))
    elif n_rows == 4:
        fig, axes = plt.subplots(4, 3, figsize=(10, 14))
    elif n_rows == 5:
        fig, axes = plt.subplots(5, 3, figsize=(10, 17.5))

    def map_tag_name(tag):
        vals = tag.split('-')
        dataset_name = vals[0]
        if dataset_name == 'mimic':
            tag_name = 'MXR'
        elif dataset_name == 'cxp':
            tag_name = 'CXP'

        if vals[1] == 'orig':
            if version != 'orig':
                tag_name += '-Orig'
        elif vals[1] == 'resampled':
            tag_name += '-Resampled'
        elif vals[1] == 'dicom':
            tag_name += '-Dcm'
        elif vals[1] == 'test_resampled':
            tag_name += '-Tst Resampled'
        elif vals[1] == 'tr_resampled':
            tag_name += '-Tr/Tst Resampled'
        elif vals[1] == 'bmi_tr_resampled':
            tag_name += '-BMI'

        return tag_name

    for i, tag in enumerate(tags):
        tag_name = map_tag_name(tag)
        for j, r in enumerate(R_LABELS):
            new_mat = np.zeros((n_levels, n_levels + 1))
            new_mat[:, :-1] = all_score_mats[tag][r]
            new_mat[0, -1] = max_abs_change
            new_mat[1, -1] = -1 * max_abs_change

            this_ax = axes[i, j]

            im = this_ax.imshow(np.flipud(new_mat), cmap='bwr')
            this_ax.set_xlim(-0.5, n_levels - 0.5)
            if i == (n_rows - 1):
                this_ax.set_xticks(range(n_levels), labels=ax_labels, fontweight='bold')
                this_ax.set_xlabel('% Field of View Decrease', fontweight='bold', size=10.5)
            else:
                this_ax.set_xticks(range(n_levels), labels=[], fontweight='bold')

            if j == 0:
                this_ax.set_yticks(range(n_levels), labels=np.flipud(ax_labels), fontweight='bold')
                this_ax.set_ylabel('% Window Width Decrease', fontweight='bold', size=10.5)
                this_ax.text(-0.31, 0.5, tag_name, rotation=90, horizontalalignment='center', verticalalignment='center',
                                transform=this_ax.transAxes, size=16, fontweight='bold')
            else:
                this_ax.set_yticks(range(n_levels), labels=[], fontweight='bold')

            if i == 0:
                this_ax.set_title(r, fontweight='bold', size=16)

    fig.subplots_adjust(left=0.05, right=0.8, wspace=0.06, hspace=0.04)

    cb_ax = fig.add_axes([0.815, 0.12, 0.02, 0.75])
    cbar = fig.colorbar(im, cax=cb_ax)
    for label in cb_ax.get_yticklabels():
        label.set_fontweight('bold')

    cb_ax.text(3.75, 0.5, '% Change in Mean Race Prediction Score', rotation=270, horizontalalignment='center',
               verticalalignment='center',
               transform=cb_ax.transAxes, size=16, fontweight='bold')  # was size 12

    for ext in ['png', 'pdf']:
        out_path = os.path.join(base_dir, 'plots', f'agg_window_width_x_size_score_changev3-{model_tag}_{version}.{ext}')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def create_aggregate_mean_im_plot():
    base_dir = PROJECT_DIR + 'mean_images/'
    split = 'train'

    mean_images = {}
    for d in ['cxp', 'mimic']:
        for r in R_LABELS:
            base_name = base_dir + d + f'_{split}'
            mean_images[(d, r)] = np.load(base_name + f'-{r}.npy')

        mean_images[(d, 'all')] = (mean_images[(d, 'Asian')] + mean_images[(d, 'Black')] + mean_images[
            (d, 'White')]) / 3

    diff_images = {}
    max_diffs = {}
    min_vals = {}
    max_vals = {}
    for d in ['cxp', 'mimic']:
        max_abs_diff = 0
        max_val = 0
        min_val = 9999
        for r in R_LABELS:
            diff_images[(d, r)] = mean_images[(d, r)] - mean_images[(d, 'all')]
            this_abs = np.max([diff_images[(d, r)].max(), np.abs(diff_images[(d, r)].min())])
            if this_abs > max_abs_diff:
                max_abs_diff = this_abs
            max_val = max(max_val, mean_images[(d, r)].max())
            min_val = min(min_val, mean_images[(d, r)].min())
        max_diffs[d] = max_abs_diff
        min_vals[d] = min_val
        max_vals[d] = max_val

    sns.set_theme()
    sns.set_style("ticks")

    fig, axes = plt.subplots(2, 9, figsize=(22, 22 * 2 / 9))

    ax_counter = 1
    for i, d in enumerate(['cxp', 'mimic']):
        mean_ax = 0 if d == 'cxp' else 5
        this_im = create_full_range_im(mean_images[(d, 'all')], max_vals[d], min_vals[d])
        axes[0, mean_ax].imshow(this_im, aspect='equal', cmap='gray')
        axes[0, mean_ax].set_title('Mean', fontweight='bold', size=14)

        axes[1, mean_ax].text(0.5, 0.5, 'Relative Difference\nCompared to Mean', ha='center', va='center',
                              fontweight='bold', size=12)
        for j, r in enumerate(R_LABELS):
            this_im = create_full_range_im(mean_images[(d, r)], max_vals[d], min_vals[d])
            axes[0, ax_counter].imshow(this_im, aspect='equal', cmap='gray')
            axes[0, ax_counter].set_xlim(-0.5, this_im.shape[1] - 1.5)
            axes[0, ax_counter].set_title(f'{r} Patients', fontweight='bold', size=14)

            this_im = create_full_range_im(diff_images[(d, r)], max_diffs[d])
            axes[1, ax_counter].imshow(this_im, aspect='equal', cmap='bwr')
            axes[1, ax_counter].set_xlim(-0.5, this_im.shape[1] - 1.5)

            ax_counter += 1
        ax_counter += 2

    for ax in axes.flatten():
        sns.despine(ax=ax, bottom=True, left=True)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    x = (axes[0, 1].get_position()._points[1, 0] + axes[0, 2].get_position()._points[0, 0]) / 2
    fig.text(x, 0.97, 'CXP', ha='center', va='center', fontweight='bold', size=18)
    x = (axes[0, 6].get_position()._points[1, 0] + axes[0, 7].get_position()._points[0, 0]) / 2
    fig.text(x, 0.97, 'MXR', ha='center', va='center', fontweight='bold', size=18)

    for ext in ['png', 'pdf']:
        out_path = os.path.join(base_dir, 'plots', f'aggregate_mean_im_plot.{ext}')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)


def create_score_change_by_view_plot(model_name, checkpoint_name, dataset, split='test',
                                     confounder_resampled=False, use_dicoms=False):
    base_dir = PROJECT_DIR + 'factor_correlation_analysis/'
    if confounder_resampled:
        if confounder_resampled == 'bmi':
            con_tag = '-bmi_resampled'
        else:
            con_tag = '-confounder_resampled'
    else:
        con_tag = ''
    dicom_tag = '_dicoms' if use_dicoms else ''

    fname = f'{model_name}_{checkpoint_name}_{dataset}_{split}{con_tag}{dicom_tag}-mean_scores.csv'
    df_path = os.path.join(base_dir, fname)
    df = pd.read_csv(df_path, index_col=0)
    print('loading: ' + fname)

    train_df = load_split_metadf(dataset, 'train') # get counts from train
    train_df = train_df[train_df.Mapped_Race.isin(R_LABELS)]

    if dataset == 'cxp':
        views = ['PA', 'AP', ('LATERAL', 'LL')]
        train_df = create_cxp_view_column(train_df)
    else:
        views = ['PA', 'AP', ('LATERAL', 'LL'), 'AP_nonport', 'AP_port']
        train_df = create_mimic_isportable_column(train_df)
        train_df['view'] = train_df.ViewPosition

    view_counts = {}
    for r in R_LABELS:
        r_idx = train_df.Mapped_Race == r
        for v in views:
            if 'port' in v:
                this_view = v.split('_')[0]
                is_port = v.split('_')[1] == 'port'
                idx = r_idx & (train_df.view == this_view) & (train_df.is_portable == is_port)
            elif isinstance(v, tuple):
                idx = r_idx & (train_df.view.isin(v))
            else:
                idx = r_idx & (train_df.view == v)
            view_counts[(r, v)] = idx.sum()

    orig_scores = {}
    orig_counts = {}
    for r in R_LABELS:
        orig_scores[r] = df.loc[f'weighted_{r}', "('default', None)"]
        orig_counts[r] = train_df.Mapped_Race.value_counts().loc[r]

    score_diffs = np.zeros((len(R_LABELS), len(views)))
    counts = np.zeros((len(R_LABELS), len(views)))
    for i, r in enumerate(R_LABELS):
        for j, v in enumerate(views):
            if 'port' in v:
                col = f"('vp', '{v}')"
            elif isinstance(v, tuple):
                col = f"('v', {v})"
            else:
                col = f"('v', '{v}')"
            this_val = df.loc[f'weighted_{r}', col]
            score_diffs[i, j] = 100 * (this_val - orig_scores[r]) / orig_scores[r]  # percent change
            counts[i, j] = view_counts[(r, v)]

    base_perc = np.array([orig_counts[r] for r in R_LABELS])
    base_perc = 100 * base_perc / base_perc.sum()

    count_diffs = np.zeros_like(counts)
    for j, v in enumerate(views):
        view_total = counts[:, j].sum()
        for i, r in enumerate(R_LABELS):
            this_perc = 100 * counts[i, j] / view_total
            count_diffs[i, j] = 100 * (this_perc - base_perc[i]) / base_perc[i]

    out_dir = PROJECT_DIR + 'factor_correlation_analysis/plots/'
    out_dir += '{}-{}-{}-{}/'.format(model_name, checkpoint_name, dataset, split)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    pkl_out_path = os.path.join(out_dir, f'score_change_by_view{con_tag}{dicom_tag}-data.pkl')
    with open(pkl_out_path, 'wb') as f:
        pkl.dump([score_diffs, count_diffs, views], f)

    # create plots
    x = np.arange(len(views))  # the label locations
    width = 0.35  # the width of the bars

    # load std err which were computed based on bootstrapping
    boot_dir = PROJECT_DIR + 'factor_correlation_analysis/bootstrap_analysis/'
    with open(boot_dir + f'view_score_percent_diffs_bootstrap-{model_name}-{dataset}{con_tag}{dicom_tag}.pkl', 'rb') as f:
        score_boots = pkl.load(f)
    with open(boot_dir + f'view_percent_diffs_bootstrap-{dataset}.pkl', 'rb') as f:
        view_boots = pkl.load(f)

    sns.set_theme()
    sns.set_style("ticks")
    for i, r in enumerate(R_LABELS):
        fig, ax = plt.subplots()
        score_plot_errs = []
        view_plot_errs = []
        for v in views:
            score_plot_errs.append(np.std(score_boots[(r, v)]))
            view_plot_errs.append(np.std(view_boots[(r, v)]))

        rects1 = ax.bar(x - width / 2, score_diffs[i], width, yerr=score_plot_errs, label='Mean AI Score')
        rects2 = ax.bar(x + width / 2, count_diffs[i], width, yerr=view_plot_errs, label='Frequency of View')

        ax.set_ylabel('Relative Change Compared to Baseline (%)', fontweight='bold', size=12)
        view_labels = []
        for v in views:
            if isinstance(v, tuple):
                view_labels.append('LAT')
            elif v == 'AP_port':
                view_labels.append('Port. AP')
            elif v == 'AP_nonport':
                view_labels.append('Std. AP')
            else:
                view_labels.append(v)

        ax.set_xticks(x, view_labels, fontweight='bold', size=12)
        plt.yticks(fontweight='bold')
        if i == 2:
            ax.legend(prop={'weight': 'bold', 'size': 11})

        sns.despine()

        plt.title(r + ' Patients', fontweight='bold', size=14)
        plt.tight_layout()
        plt.savefig(out_dir + f'score_change_by_view_plot{con_tag}{dicom_tag}-{r}.png')
        plt.close()


def create_aggregate_score_change_by_view_plot(model_tag, confounder_resampled=False, use_dicoms=False):
    base_dir = PROJECT_DIR + 'factor_correlation_analysis/'

    con_tag = '-confounder_resampled' if confounder_resampled else ''
    dicom_tag = '_dicoms' if use_dicoms else ''

    if use_dicoms:
        datasets = ['mimic-jpg', 'mimic-dicom']
    else:
        datasets = ['cxp', 'mimic']

    all_score_diffs = {}
    all_count_diffs = {}
    all_views = {}
    all_score_boots = {}
    all_view_boots = {}
    for dataset_tag in datasets:
        if '-' in dataset_tag:
            vals = dataset_tag.split('-')
            dataset = vals[0]
            d_tag = dicom_tag if vals[1] == 'dicom' else ''
        else:
            d_tag = dicom_tag
            dataset = dataset_tag
        model_name = dataset + model_tag
        this_dir = base_dir + 'plots/' + '{}-{}-{}-{}/'.format(model_name, 'best', dataset, 'test')
        fname = f'score_change_by_view{con_tag}{d_tag}-data.pkl'
        print('loading: ' + fname)
        with open(this_dir + fname, 'rb') as f:
            this_data = pkl.load(f)
            all_score_diffs[dataset_tag] = this_data[0]
            all_count_diffs[dataset_tag] = this_data[1]
            all_views[dataset_tag] = this_data[2]

        boot_dir = PROJECT_DIR + 'factor_correlation_analysis/bootstrap_analysis/'
        with open(boot_dir + f'view_score_percent_diffs_bootstrap-{model_name}-{dataset}{con_tag}{d_tag}.pkl', 'rb') as f:
            all_score_boots[dataset_tag] = pkl.load(f)
        with open(boot_dir + f'view_percent_diffs_bootstrap-{dataset}.pkl', 'rb') as f:
            all_view_boots[dataset_tag] = pkl.load(f)

    sns.set_theme()
    sns.set_style("ticks")
    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
    width = 0.35  # the width of the bars
    n_views = len(all_views[datasets[-1]])
    for i, d in enumerate(datasets):
        if d == 'mimic':
            d_name = 'MXR'
        elif d == 'cxp':
            d_name = 'CXP'
        elif d == 'mimic-jpg':
            d_name = 'MXR-JPG'
        elif d == 'mimic-dicom':
            d_name = 'MXR-DCM'
        #d_name = 'MXR' if d == 'mimic' else 'CXP'
        # x = np.arange(len(all_views[d]))  # the label locations
        x = np.arange(n_views)
        for j, r in enumerate(R_LABELS):
            score_plot_errs = []
            view_plot_errs = []
            for v in all_views[d]:
                score_plot_errs.append(np.std(all_score_boots[d][(r, v)]))
                view_plot_errs.append(np.std(all_view_boots[d][(r, v)]))

            if d == 'cxp':
                these_score_diffs = np.append(all_score_diffs[d][j], [np.nan, np.nan])
                these_count_diffs = np.append(all_count_diffs[d][j], [np.nan, np.nan])
                score_plot_errs.extend([np.nan, np.nan])
                view_plot_errs.extend([np.nan, np.nan])
            else:
                these_score_diffs = all_score_diffs[d][j]
                these_count_diffs = all_count_diffs[d][j]

            rects1 = axes[i, j].bar(x - width / 2, these_score_diffs, width, yerr=score_plot_errs,
                                    label='Mean Race Prediction Score')
            rects2 = axes[i, j].bar(x + width / 2, these_count_diffs, width, yerr=view_plot_errs,
                                    label='Frequency of View')

            if j == 0:
                axes[i, j].set_ylabel('% Change from Baseline', fontweight='bold', size=17)
                axes[i, j].text(-0.22, 0.5, d_name, rotation=90, horizontalalignment='center',
                                verticalalignment='center',
                                transform=axes[i, j].transAxes, size=29, fontweight='bold')

            if d == 'cxp':
                axes[i, j].text(3, 0, 'N/A', size=14, horizontalalignment='center', verticalalignment='center',
                                fontweight='bold')
                axes[i, j].text(4, 0, 'N/A', size=14, horizontalalignment='center', verticalalignment='center',
                                fontweight='bold')

            view_labels = []
            for v in all_views[d]:
                if isinstance(v, tuple):
                    view_labels.append('Lat')
                elif v == 'AP_port':
                    view_labels.append('Port.')
                elif v == 'AP_nonport':
                    view_labels.append('Std.')
                else:
                    view_labels.append(v)

            if d == 'cxp':
                view_labels.extend(['Std.', 'Port.'])

            axes[i, j].set_xticks(x, view_labels, fontweight='bold', size=16)
            axes[i, j].text(0.79, -.15, 'AP View Type', horizontalalignment='center', verticalalignment='center',
                            transform=axes[i, j].transAxes, size=16, fontweight='bold')

            for label in axes[i, j].get_yticklabels():
                label.set_fontsize(14)
                label.set_fontweight('bold')

            if i == 0 and j == 0:
                axes[i, j].yaxis.set_major_locator(plt.MaxNLocator(6))
            elif j == 2:
                axes[i, j].yaxis.set_major_locator(plt.MaxNLocator(7))

            sns.despine()

            if i == 0:
                axes[i, j].set_title(r, fontweight='bold', size=25)
                if j == 2:
                    axes[i, j].legend(prop={'weight': 'bold', 'size': 16})

    for j in range(3):
        axes[0, j].set_xlim(axes[1, j].get_xlim())

    fig.subplots_adjust(wspace=0.17, hspace=0.25)

    for ext in ['png', 'pdf']:
        out_path = os.path.join(base_dir, 'plots', f'agg_view_score_change{model_tag}{con_tag}{dicom_tag}_v2.{ext}')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def create_score_change_by_view_plot_comparison(model_tag, dataset_name, include_bmi=False):
    base_dir = PROJECT_DIR + 'factor_correlation_analysis/'

    if dataset_name == 'mimic':
        if include_bmi:
            tags = ['orig', 'test_resampled', 'tr_resampled', 'dicom', 'bmi_tr_resampled']
        else:
            tags = ['orig', 'test_resampled', 'tr_resampled', 'dicom']
    elif dataset_name == 'cxp':
        tags = ['orig', 'test_resampled', 'tr_resampled'] #['orig', 'resampled']

    for ti, tag in enumerate(tags):
        if tag == 'orig':
            con_tag = ''
            dicom_tag = ''
        elif 'resampled' in tag:
            if 'bmi' in tag:
                con_tag = '-bmi_resampled'
            else:
                con_tag = '-confounder_resampled'
            dicom_tag = ''
        elif tag == 'dicom':
            dicom_tag = '_dicoms'
            con_tag = ''
        else:
            raise ValueError()

        if tag == 'tr_resampled':
            model_name = dataset_name + model_tag + '_multifact-class-balance'
        elif tag == 'bmi_tr_resampled':
            model_name = dataset_name + model_tag + '_BMI-class-balance'
        else:
            model_name = dataset_name + model_tag
        this_dir = base_dir + 'plots/' + '{}-{}-{}-{}/'.format(model_name, 'best', dataset_name, 'test')
        fname = f'score_change_by_view{con_tag}{dicom_tag}-data.pkl'
        print('loading: ' + fname)
        with open(this_dir + fname, 'rb') as f:
            this_data = pkl.load(f)
            if ti == 0:
                views = this_data[-1]
                all_score_diffs = np.zeros((len(tags), len(R_LABELS), len(views)))
            all_score_diffs[ti] = this_data[0]
            assert str(this_data[-1]) == str(views)

        boot_dir = PROJECT_DIR + 'factor_correlation_analysis/bootstrap_analysis/'
        with open(boot_dir + f'view_score_percent_diffs_bootstrap-{model_name}-{dataset_name}{con_tag}{dicom_tag}.pkl',
                  'rb') as f:
            boot_data = pkl.load(f)
            if ti == 0:
                all_score_errs = np.zeros((len(tags), len(R_LABELS), len(views)))
            for ri, r in enumerate(R_LABELS):
                for vi, v in enumerate(views):
                    all_score_errs[ti, ri, vi] = np.std(boot_data[(r, v)])

    def map_tag_name(tag):
        if dataset_name == 'mimic':
            tag_name = 'MXR'
        elif dataset_name == 'cxp':
            tag_name = 'CXP'

        if tag == 'orig':
            tag_name += '-Orig'
        elif tag == 'resampled':
            tag_name += '-Resampled'
        elif tag == 'dicom':
            tag_name += '-Dcm'
        elif tag == 'test_resampled':
            tag_name += '-Tst Resampled'
        elif tag == 'tr_resampled':
            tag_name += '-Tr/Tst Resampled'
        elif tag == 'bmi_tr_resampled':
            tag_name += '-BMI'

        return tag_name

    sns.set_theme()
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, 3, figsize=(24, 5))
    n_tags = len(tags)
    if n_tags == 2:
        width = 0.35
    elif n_tags == 3:
        width = 0.25
    elif n_tags == 4:
        width = 0.2
    elif n_tags == 5:
        width = 0.175
    #width = 0.25 if n_tags == 3 else 0.35
    n_views = len(views)
    tag_names = [map_tag_name(tag) for tag in tags]

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors_to_use = [default_colors[i] for i in [0, 2, 4, 6, 8]]  # skip colors used before

    x = np.arange(n_views)
    for j, r in enumerate(R_LABELS):
        for ti, t in enumerate(tag_names):
            if ti == 0:
                if n_tags == 2:
                    delta = -0.5 * width
                elif n_tags == 3:
                    delta = -1. * width
                elif n_tags == 4:
                    delta = -1.5 * width
                elif n_tags == 5:
                    delta = -2 * width
            elif ti == 1:
                if n_tags == 2:
                    delta = 0.5 * width
                elif n_tags == 3:
                    delta = 0
                elif n_tags == 4:
                    delta = -0.5 * width
                elif n_tags == 5:
                    delta = -1 * width
            elif ti == 2:
                if n_tags == 3:
                    delta = width
                elif n_tags == 4:
                    delta = 0.5 * width
                elif n_tags == 5:
                    delta = 0
            elif ti == 3:
                if n_tags == 4:
                    delta = 1.5 * width
                else:
                    delta = width
            else:
                delta = 2 * width


            axes[j].bar(x + delta, all_score_diffs[ti, j], width, yerr=all_score_errs[ti, j],
                        label=t, color=colors_to_use[ti])

        if j == 0:
            axes[j].set_ylabel('% Change in Race Prediction Score', fontweight='bold', size=15)
        #         axes[j].text(-0.22, 0.5, t.split('-')[0], rotation=90, horizontalalignment='center',
        #                         verticalalignment='center',
        #                         transform=axes[j].transAxes, size=29, fontweight='bold')

        view_labels = []
        for v in views:
            if isinstance(v, tuple):
                view_labels.append('Lat')
            elif v == 'AP_port':
                view_labels.append('Port.')
            elif v == 'AP_nonport':
                view_labels.append('Std.')
            else:
                view_labels.append(v)

        axes[j].set_xticks(x, view_labels, fontweight='bold', size=16)
        if dataset_name == 'mimic':
            axes[j].text(0.79, -.15, 'AP View Type', horizontalalignment='center', verticalalignment='center',
                         transform=axes[j].transAxes, size=16, fontweight='bold')

        for label in axes[j].get_yticklabels():
            label.set_fontsize(14)
            label.set_fontweight('bold')

        #     if i == 0 and j == 0:
        #         axes[i, j].yaxis.set_major_locator(plt.MaxNLocator(6))
        #     elif j == 2:
        #         axes[i, j].yaxis.set_major_locator(plt.MaxNLocator(7))

        sns.despine()

        axes[j].set_title(r, fontweight='bold', size=25)
        # if j == 0:
        #     axes[j].legend(prop={'weight': 'bold', 'size': 15}, loc='upper left')
        if j == 1:
            axes[j].legend(prop={'weight': 'bold', 'size': 15}, loc='best')

    fig.subplots_adjust(wspace=0.17, hspace=0.25)

    bmi_tag = '-with_bmi' if include_bmi else ''
    for ext in ['png', 'pdf']:
        out_path = os.path.join(base_dir, 'plots', f'agg_view_score_changev4{model_tag}{dataset_name}_comparison{bmi_tag}.{ext}')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    print('Correlations in scores for', model_tag, dataset_name)
    print('tag', 'corr', '# in same direction', '% same direction')
    base_vals = all_score_diffs[0].flatten()
    for ti, tag in enumerate(tags):
        if ti == 0:
            continue
        these_vals = all_score_diffs[ti].flatten()
        corr = np.corrcoef(base_vals, these_vals)[0, 1]
        n_same_dir = np.sum((these_vals > 0) == (base_vals > 0))
        print(tag, corr, n_same_dir, n_same_dir / len(base_vals))


def create_disparity_change_plot(dataset, confounder_resampled=False, use_dicoms=False, simple_title=False):
    sns.set_theme()
    sns.set_style("ticks")

    if dataset == 'mimic':  # computed based on results
        # results on MIMIC test set, by model and race
        # MIMIC - Asian, MIMIC - Black, CXP - Asian, CXP - Black
        mean_disps = {}
        std_disps = {}
        # numbers computed from results and entered for ease of use (and reviewed)
        if confounder_resampled:
            assert use_dicoms == False
            if confounder_resampled == 'tr_test':
                mean_disps['orig'] = [80.28 - 75.01, 80.28 - 74.58, 77.03 - 75.65, 77.03 - 73.63]
                mean_disps['aug'] = [80.34 - 77.40, 80.34 - 74.43, 77.68 - 71.53, 77.68 - 74.70]
                mean_disps['per_view'] = [80.74 - 76.55, 80.74 - 77.13, 76.62 - 76.31, 76.62 - 75.52]
                std_disps['orig'] = [1.69, 0.91, 1.66, 0.88]
                std_disps['aug'] = [1.66, 0.91, 1.79, 0.92]
                std_disps['per_view'] = [1.61, 0.87, 1.66, 0.86]
            elif confounder_resampled == 'bmi':
                mean_disps['orig'] = [89.18 - 82.96, 89.18 - 87.55, 0, 0]
                mean_disps['aug'] = [88.43 - 89.79, 88.43 - 86.18, 0, 0]
                mean_disps['per_view'] = [85.45 - 81.60, 85.45 - 85.03, 0, 0]
                std_disps['orig'] = [4.31, 0.9, 0, 0]
                std_disps['aug'] = [2.49, 0.91, 0, 0]
                std_disps['per_view'] = [4.24, 0.93, 0, 0]
            else:
                mean_disps['orig'] = [82.52 - 78.36, 82.52 - 77.76, 80.27 - 75.16, 80.27 - 77.39]
                mean_disps['aug'] = [83.02 - 77.88, 83.02 - 76.58, 81.23 - 74.91, 81.23 - 76.02]
                mean_disps['per_view'] = [82.19 - 80.70, 82.19 - 80.90, 80.26 - 76.38, 80.26 - 78.86]
                std_disps['orig'] = [1.63, 0.87, 1.64, 0.86]
                std_disps['aug'] = [1.63, 0.88, 1.68, 0.86]
                std_disps['per_view'] = [1.48, 0.82, 1.6, 0.82]
        elif use_dicoms:
            mean_disps['orig'] = [83.02 - 78.96, 83.02 - 74.75, 82.62 - 77.61, 82.62 - 75.48]
            mean_disps['aug'] = [80.93 - 77.53, 80.93 - 71.86, 83.98 - 78.56, 83.98 - 75.98]
            mean_disps['per_view'] = [82.59 - 81.25, 82.59 - 78.80, 82.26 - 79.43, 82.26 - 77.23]
            std_disps['orig'] = [1.46, 0.79, 1.46, 0.73]
            std_disps['aug'] = [1.49, 0.83, 1.44, 0.76]
            std_disps['per_view'] = [1.38, 0.71, 1.34, 0.71]
        else:
            mean_disps['orig'] = [83.52 - 80.47, 83.52 - 75.77, 81.64 - 77.79, 81.64 - 74.11]
            mean_disps['aug'] = [83.96 - 80.32, 83.96 - 74.31, 82.73 - 77.39, 82.73 - 73.19]
            mean_disps['per_view'] = [83.3 - 82.29, 83.30 - 79.07, 81.57 - 78.89, 81.57 - 76.16]
            std_disps['orig'] = [1.39, 0.77, 1.38, 0.76]
            std_disps['aug'] = [1.4, 0.81, 1.44, 0.79]
            std_disps['per_view'] = [1.26, 0.72, 1.33, 0.72]

    x = np.array([0, 2, 5.5, 7.5])  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - 1 * width, mean_disps['orig'], width, yerr=std_disps['orig'], label='Baseline Approach')
    rects2 = ax.bar(x, mean_disps['aug'], width, yerr=std_disps['aug'], label='Data Augmentation')
    rects3 = ax.bar(x + 1 * width, mean_disps['per_view'], width, yerr=std_disps['per_view'], label='Per View Threshold')

    ax.set_ylabel('Disparity in Sensitivity (Absolute %)', fontweight='bold', size=12)

    ax.tick_params(axis='x', length=0)
    ax.set_xticks([])

    plt.yticks(fontweight='bold')
    if confounder_resampled == 'bmi':
        plt.text(x[:2].mean(), -4.35+0.2, 'Train on MXR', ha='center', size=14, fontweight='bold')
        plt.text(x[2:].mean(), -4.35+0.2, 'Train on CXP', ha='center', size=14, fontweight='bold')
        plt.text(x[0], -2.3+0.2, 'Asian\nPatients', ha='center', va='top', size=12, fontweight='bold')
        plt.text(x[1], -2.3+0.2, 'Black\nPatients', ha='center', va='top', size=12, fontweight='bold')
        plt.text(x[2], -2.3+0.2, 'Asian\nPatients', ha='center', va='top', size=12, fontweight='bold')
        plt.text(x[3], -2.3+0.2, 'Black\nPatients', ha='center', va='top', size=12, fontweight='bold')
    else:
        plt.text(x[:2].mean(), -2.35, 'Train on MXR', ha='center', size=14, fontweight='bold')
        plt.text(x[2:].mean(), -2.35, 'Train on CXP', ha='center', size=14, fontweight='bold')
        plt.text(x[0], -0.3, 'Asian\nPatients', ha='center', va='top', size=12, fontweight='bold')
        plt.text(x[1], -0.3, 'Black\nPatients', ha='center', va='top', size=12, fontweight='bold')
        plt.text(x[2], -0.3, 'Asian\nPatients', ha='center', va='top', size=12, fontweight='bold')
        plt.text(x[3], -0.3, 'Black\nPatients', ha='center', va='top', size=12, fontweight='bold')
    
    if confounder_resampled or use_dicoms:
        if confounder_resampled == 'bmi':
            ax.set_ylim((-2.5, 11.006999999999993))
        else:
            ax.set_ylim((-0.8070000000000092, 11.006999999999993)) # same as original

    ax.legend(prop={'weight': 'bold', 'size': 10}, loc='upper center' if confounder_resampled else (.37, .75))

    sns.despine(bottom=True)
    #plt.tight_layout()

    if simple_title:
        if confounder_resampled:
            if confounder_resampled == 'tr_test':
                plt.title('MXR-Tr/Tst Resampled', fontweight='bold', size=14)
            elif confounder_resampled == 'bmi':
                plt.title('MXR-BMI', fontweight='bold', size=14)
            else:
                plt.title('MXR-Tst Resampled', fontweight='bold', size=14)
                #plt.title('MXR-Resampled', fontweight='bold', size=14)
        elif use_dicoms:
            plt.title('MXR-Dcm', fontweight='bold', size=14)
        else:
            plt.title('MXR-Orig', fontweight='bold', size=14)
    else:
        if confounder_resampled:
            plt.title('Sensitivity Disparity in MXR Test Split - Resampled', fontweight='bold', size=14)
        elif use_dicoms:
            plt.title('Sensitivity Disparity in MXR Test Split - DICOMs', fontweight='bold', size=14)
        else:
            plt.title('Sensitivity Disparity in MXR Test Split', fontweight='bold', size=14)
    out_dir = PROJECT_DIR + 'disparity_analysis/plots/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    con_tag = '-confounder_resampled' if confounder_resampled else ''
    if confounder_resampled == 'bmi':
        con_tag = con_tag.replace('confounder', 'bmi')
    elif confounder_resampled == 'tr_test':
        con_tag += '-tr_test'
    dicom_tag = '_dicoms' if use_dicoms else ''
    sim_tag = '_simple-title' if simple_title else ''
    for ext in ['png', 'pdf']:
        plt.savefig(out_dir + f'sensitivity_disparity-{dataset}{con_tag}{dicom_tag}{sim_tag}.{ext}')

    plt.close()


def plot_aggregate_score_distribution():
    model_tag = '_disease_orig'
    all_pred_dfs = {}
    for dataset in ['cxp', 'mimic']:
        model_name = dataset + model_tag
        pred_df = load_pred_df(model_name, dataset, split='test', merge_labels=False)

        pred_df = pred_df[pred_df.View.isin(['AP', 'PA', 'LATERAL', 'LL'])]
        lat_idx = pred_df.View.isin(['LL', 'LATERAL'])
        pred_df.loc[lat_idx, 'View'] = 'LAT'

        if dataset == 'mimic':
            test_df = load_split_metadf(dataset, 'test')
            proc_map = test_df[['PerformedProcedureStepDescription', 'dicom_id']].set_index('dicom_id')
            pred_df['PerformedProcedureStepDescription'] = pred_df.dicom_id.map(
                proc_map['PerformedProcedureStepDescription'])
            pred_df = create_mimic_isportable_column(pred_df)

            idx = (pred_df.View == 'AP') & (pred_df.is_portable == True)
            pred_df.loc[idx, 'View'] = 'Port-AP'
            idx = (pred_df.View == 'AP') & (pred_df.is_portable == False)
            pred_df.loc[idx, 'View'] = 'Std-AP'

        all_pred_dfs[dataset] = pred_df

    sns.set_theme()
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, d in enumerate(['cxp', 'mimic']):
        d_name = 'MXR' if d == 'mimic' else 'CXP'
        if d == 'mimic':
            hue_order = ['PA', 'LAT', 'Port-AP', 'Std-AP']
        else:
            hue_order = ['PA', 'LAT', 'AP']

        all_pred_dfs[d]["Pred_Findings"] = 1 - all_pred_dfs[d]["Pred_No Finding"]
        ax = sns.kdeplot(
            data=all_pred_dfs[d], x="Pred_Findings", hue="View", hue_order=hue_order, common_norm=False, clip=[0, 1], linewidth=2,
            ax=axes[i])

        for j in range(len(ax.lines)):
            orig_y = ax.lines[j].get_ydata()
            ax.lines[j].set_ydata(orig_y / orig_y.sum())
            ax.get_legend().texts[j]._fontproperties._weight = 'bold'

        ax.set_ylim([0, 0.08 if d == 'cxp' else 0.035])
        ax.set_xlim([0, 1])
        ax.set_xlabel('Findings Present Score', fontweight='bold', fontsize=14)
        ax.set_ylabel('Score Distribution Density', fontweight='bold', fontsize=14)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')

        ax.set_title(d_name, fontweight='bold', fontsize=16)
        ax.get_legend()._legend_title_box._text._fontproperties._weight = 'bold'
        sns.despine()

    plt.tight_layout()

    out_dir = PROJECT_DIR + 'factor_correlation_analysis/plots/'
    for ext in ['png', 'pdf']:
        plt.savefig(out_dir + f'score_dist_by_view-{model_tag}.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()


if __name__ == '__main__':
    create_disparity_change_plot('mimic', confounder_resampled='tr_test', use_dicoms=False, simple_title=True)
