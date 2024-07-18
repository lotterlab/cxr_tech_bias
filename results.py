import pandas as pd
import sys
import numpy as np
import pdb
import os
import tqdm
import pickle as pkl
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.append('../torchxrayvision/')

import torchxrayvision as xrv

from data_utils import create_mimic_isportable_column, load_split_metadf, create_cxp_view_column
from constants import *
from delong_helpers import delong_roc_variance, delong_ci


def load_dataset(dataset_name, split):
    if dataset_name == 'cxp':
        this_path = PROJECT_DIR + f'cxp_cv_splits/version_0/{split}.csv'
        dataset = xrv.datasets.CheX_Dataset(
            imgpath='',
            csvpath=this_path,
            transform=[], data_aug=None, unique_patients=False, views='all', use_no_finding=True)
    else:
        csvpath = PROJECT_DIR + f'mimic_cv_splits/version_0/cxp-labels_{split}.csv'
        metacsvpath = csvpath.replace('cxp-labels', 'meta')
        dataset = xrv.datasets.MIMIC_Dataset(
            imgpath='',
            csvpath=csvpath,
            metacsvpath=metacsvpath,
            transform=[], data_aug=None, unique_patients=False, views='all', use_no_finding=True)

    return dataset


def compute_metrics(df, races, views, threshold, is_portable=None):
    # compute auc, sensitivity, and specificity per filtering by different inclusion criteria
    if not isinstance(races, list):
        if races == 'all':
            if pd.isnull(df.Mapped_Race).sum():
                df.loc[pd.isnull(df.Mapped_Race), 'Mapped_Race'] = 'nan'
            races = df.Mapped_Race.unique()
        else:
            races = [races]
    if not isinstance(views, list):
        if views == 'all':
            if pd.isnull(df.View).sum():
                df.loc[pd.isnull(df.View), 'View'] = 'nan'
            views = df.View.unique()
        else:
            views = [views]
    idx = df.Mapped_Race.isin(races) & df.View.isin(views)
    if is_portable is not None:
        idx = idx & (df.is_portable == is_portable)

    keep_idx = idx & df[NO_FINDINGS_TAG].isin([0, 1])
    yhat = df.loc[keep_idx, 'Pred_{}'.format(NO_FINDINGS_TAG)]
    y = df.loc[keep_idx, NO_FINDINGS_TAG]

    auc = roc_auc_score(y.values, yhat.values)

    spec = 100 * np.mean(yhat[y == 1] >= threshold)
    sens = 100 * np.mean(yhat[y == 0] < threshold)

    return auc, sens, spec


def compute_threshold(df, target_sens='balanced'):

    idx = df.View.isin(ALL_VIEWS)

    if target_sens == 'balanced':
        keep_idx = idx & (df[NO_FINDINGS_TAG].isin([0,1]))
        yhat = df.loc[keep_idx, f'Pred_{NO_FINDINGS_TAG}'].values
        y = df.loc[keep_idx, NO_FINDINGS_TAG].values
        threshold = compute_thresh_eq_sens_spec(yhat, y)
    else:
        keep_idx = idx & (df[NO_FINDINGS_TAG] == 0)
        yhat = df.loc[keep_idx, f'Pred_{NO_FINDINGS_TAG}'].values
        threshold = np.percentile(yhat, target_sens)

    return threshold


def compute_thresh_eq_sens_spec(scores, y):
    fpr, sens, threshs = roc_curve(y, scores)
    spec = 1 - fpr
    return threshs[np.argmin(np.abs(spec - sens))]


def compute_thresholds_and_yhat_per_view(df, target_sens='balanced', only_thresholds=False):
    thresholds = {}

    if target_sens == 'balanced': # get target sens for dataset
        idx = df.View.isin(ALL_VIEWS)
        keep_idx = idx & (df[NO_FINDINGS_TAG].isin([0, 1]))
        yhat = df.loc[keep_idx, f'Pred_{NO_FINDINGS_TAG}'].values
        y = df.loc[keep_idx, NO_FINDINGS_TAG].values
        fpr, sens, _ = roc_curve(y, yhat)
        spec = 1 - fpr
        min_idx = np.argmin(np.abs(spec - sens))
        target_sens = 100 * sens[min_idx]

    if 'is_portable' in df.columns:
        view_groups = ['PA', ('LATERAL', 'LL'), ('AP', False), ('AP', True)]
    else:
        view_groups = ['PA', ('LATERAL', 'LL'), 'AP']

    if not only_thresholds:
        df[f'yhat_{NO_FINDINGS_TAG}'] = np.nan
    for v in view_groups:
        if isinstance(v, tuple):
            if 'LATERAL' in v:
                view_idx = df.View.isin(v)
            else:
                view_idx = (df.View == v[0]) & (df.is_portable == v[1])
        else:
            view_idx = df.View == v

        pos_idx = view_idx & (df[NO_FINDINGS_TAG] == 0)

        thresholds[v] = np.percentile(df.loc[pos_idx, f'Pred_{NO_FINDINGS_TAG}'], target_sens)
        if not only_thresholds:
            df.loc[view_idx, f'yhat_{NO_FINDINGS_TAG}'] = df.loc[view_idx, f'Pred_{NO_FINDINGS_TAG}'] > thresholds[v]

    return thresholds


def load_pred_df(model_name, dataset_name, split, merge_labels=True, window_width=None, resize_factor=None,
                 confounder_resampled=False, use_dicoms=False):
    tag = ''
    if confounder_resampled:
        tag += '-confounder_resampled'
    if window_width or resize_factor:
        if window_width:
            tag += f'-window{window_width}'
        if resize_factor:
            tag += f'-initresize{resize_factor}_midcrop'
    if use_dicoms:
        tag += '_dicoms'

    pred_path = os.path.join(PROJECT_DIR + 'prediction_dfs', model_name + '-best', dataset_name + '-' + split + tag + '.csv')
    pred_df = pd.read_csv(pred_path)
    if dataset_name == 'cxp':
        pred_df['orig_path'] = [p[p.find('datasets/'):] for p in pred_df.Path.values]
        study_ids = []
        for p in pred_df.Path.values:
            vals = p.split('/')
            study_ids.append(vals[-3] + '-' + vals[-2])
        pred_df['study_id'] = study_ids
    else:
        pred_df['dicom_id'] = [p.split('/')[-1][:-4] for p in pred_df.Path.values]
        pred_df['study_id'] = [p.split('/')[-2] for p in pred_df.Path.values]

    if merge_labels:
        xrv_dataset = load_dataset(dataset_name, split)

        if dataset_name == 'cxp':
            gt_df = pd.DataFrame(xrv_dataset.labels, columns=xrv_dataset.pathologies, index=xrv_dataset.csv.Path)
            merge_df = pd.merge(pred_df, gt_df, how='left', left_on='orig_path', right_index=True)
        else:
            gt_df = pd.DataFrame(xrv_dataset.labels, columns=xrv_dataset.pathologies, index=xrv_dataset.csv.dicom_id)
            merge_df = pd.merge(pred_df, gt_df, how='left', left_on='dicom_id', right_index=True)
            proc_map = xrv_dataset.csv[['PerformedProcedureStepDescription', 'dicom_id']].set_index('dicom_id')
            merge_df['PerformedProcedureStepDescription'] = merge_df.dicom_id.map(
                proc_map['PerformedProcedureStepDescription'])
            merge_df = create_mimic_isportable_column(merge_df)

        return merge_df
    else:
        return pred_df


def compute_metrics_with_view_thresholds(df, race, thresholds):
    idx = df.Mapped_Race == race

    if 'is_portable' in df.columns:
        view_groups = ['PA', ('LATERAL', 'LL'), ('AP', False), ('AP', True)]
    else:
        view_groups = ['PA', ('LATERAL', 'LL'), 'AP']

    df[f'yhat_{NO_FINDINGS_TAG}'] = np.nan
    for v in view_groups:
        if isinstance(v, tuple):
            if 'LATERAL' in v:
                view_idx = idx & (df.View.isin(v))
            else:
                view_idx = idx & (df.View == v[0]) & (df.is_portable == v[1])
        else:
            view_idx = idx & (df.View == v)

        df.loc[view_idx, f'yhat_{NO_FINDINGS_TAG}'] = df.loc[view_idx, f'Pred_{NO_FINDINGS_TAG}'] > thresholds[v]

    idx = idx & (df[NO_FINDINGS_TAG].isin([0, 1]))
    idx = idx & ~pd.isnull(df[f'yhat_{NO_FINDINGS_TAG}'])

    yhat = df.loc[idx, f'yhat_{NO_FINDINGS_TAG}']
    y = df.loc[idx, NO_FINDINGS_TAG]
    spec = 100 * np.mean(yhat[y == 1])
    sens = 100 * (1 - np.mean(yhat[y == 0]))

    return sens, spec


def run_metric_analysis(model_name, dataset_name, use_per_view_threshs=False, target_sens='balanced',
                        confounder_resampled=False, use_dicoms=False):
    pred_dfs = {}
    for split in ['val', 'test']:
        if use_dicoms and split == 'test':
            pred_dfs[split] = load_pred_df(model_name, dataset_name, split, use_dicoms=use_dicoms, window_width=1)
            idx = pd.isnull(pred_dfs[split]['Pred_No Finding'])
            print('%nan predictions', 100 * idx.mean())  # improper dicom files
            pred_dfs[split] = pred_dfs[split].loc[~idx]
        else:
            pred_dfs[split] = load_pred_df(model_name, dataset_name, split)

    if confounder_resampled:
        use_bmi = confounder_resampled == 'bmi'
        print('use_bmi', use_bmi)
        pred_dfs['test'] = confounder_resample_pred_df(pred_dfs['test'], dataset_name, use_bmi=use_bmi)

    print('===============================')
    print(f'Results for model={model_name}, data={dataset_name}, per_view_thresh={use_per_view_threshs}:')
    if use_per_view_threshs:
        thresholds = compute_thresholds_and_yhat_per_view(pred_dfs['val'], only_thresholds=True, target_sens=target_sens)

        print('Race', 'Sensitivity', 'Specificity')
        for r in R_LABELS:
            sens, spec = compute_metrics_with_view_thresholds(pred_dfs['test'], r, thresholds)
            print(r, sens, spec)

    else:
        print('Races', 'Views', 'AUROC', 'Sensitivity', 'Specificity')
        threshold = compute_threshold(pred_dfs['val'], target_sens=target_sens)

        to_eval = [(R_LABELS, ALL_VIEWS)]
        to_eval.append(('all', ALL_VIEWS))
        to_eval.append((R_LABELS[0], ALL_VIEWS))
        to_eval.append((R_LABELS[1], ALL_VIEWS))
        to_eval.append((R_LABELS[2], ALL_VIEWS))
        for tup in to_eval:
            auc, sens, spec = compute_metrics(pred_dfs['test'], tup[0], tup[1], threshold)
            print(tup[0], tup[1], f'{auc:2.3f} {sens:2.2f} {spec:2.2f}')
    print('===============================')


def confounder_resample_pred_df(orig_pred_df, dataset_name, use_bmi=False):
    pred_df = orig_pred_df.copy()
    if use_bmi:
        resampled_meta_df = load_split_metadf(dataset_name, 'test', confounder_resampled='bmi')
    else:
        resampled_meta_df = load_split_metadf(dataset_name, 'test', confounder_resampled=True)
    if dataset_name == 'cxp':
        pred_df['key'] = [p[p.find('CheX'):] for p in pred_df['Path'].values]
        match_key = 'Path'
    else:
        pred_df['key'] = [p.split('/')[-1][:-4] for p in pred_df.Path.values]
        match_key = 'dicom_id'
    assert resampled_meta_df[match_key].isin(pred_df['key']).mean() == 1
    pred_df.set_index('key', inplace=True)
    pred_df = pred_df.loc[resampled_meta_df[match_key].values]
    assert (pred_df.index == resampled_meta_df[match_key]).mean() == 1

    return pred_df


def run_bootstrap(model_name, dataset_name, n_boot=2000, target_sens='balanced', confounder_resampled=False, use_dicoms=False):
    pred_dfs = {}
    for split in ['val', 'test']:
        if use_dicoms and split == 'test': # still use orig data for thresholds
            pred_dfs[split] = load_pred_df(model_name, dataset_name, split, use_dicoms=use_dicoms, window_width=1)
            idx = pd.isnull(pred_dfs[split]['Pred_No Finding'])
            print('%nan predictions', 100 * idx.mean())  # improper dicom files
            pred_dfs[split] = pred_dfs[split].loc[~idx]
        elif confounder_resampled and (split == 'test') and ('multifact-class-balance' in model_name):
            pred_dfs[split] = load_pred_df(model_name, dataset_name, split, confounder_resampled=confounder_resampled)
        else:
            pred_dfs[split] = load_pred_df(model_name, dataset_name, split)

    if confounder_resampled and ('multifact-class-balance' not in model_name):
        use_bmi = confounder_resampled == 'bmi'
        print('use_bmi', use_bmi)
        pred_dfs['test'] = confounder_resample_pred_df(pred_dfs['test'], dataset_name, use_bmi=use_bmi)

    thresh_tags = ['orig', 'by_view']
    all_sens = {v: np.zeros((n_boot, 3)) for v in thresh_tags}
    all_spec = {v: np.zeros((n_boot, 3)) for v in thresh_tags}
    all_auc = {v: np.zeros((n_boot, 3)) for v in thresh_tags}

    all_studies = {}
    for v in ['val', 'test']:
        all_studies[v] = pred_dfs[v]['study_id'].unique()
        pred_dfs[v] = pred_dfs[v].set_index('study_id')

    for i in tqdm.tqdm(range(n_boot), total=n_boot):
        # val_boot_df = get_bootstrap_df(pred_dfs['val'])
        # test_boot_df = get_bootstrap_df(pred_dfs['test'])
        val_boot_studies = np.random.choice(all_studies['val'], len(all_studies['val']), replace=True)
        test_boot_studies = np.random.choice(all_studies['test'], len(all_studies['test']), replace=True)
        val_boot_df = pred_dfs['val'].loc[val_boot_studies]
        test_boot_df = pred_dfs['test'].loc[test_boot_studies]

        orig_thresh = compute_threshold(val_boot_df, target_sens=target_sens)
        view_thresh = compute_thresholds_and_yhat_per_view(val_boot_df, only_thresholds=True, target_sens=target_sens)

        for j, r in enumerate(R_LABELS):
            auc, sens, spec = compute_metrics(test_boot_df, r, ALL_VIEWS, orig_thresh)
            all_sens['orig'][i, j] = sens
            all_spec['orig'][i, j] = spec
            all_auc['orig'][i, j] = auc

        for j, r in enumerate(R_LABELS):
            sens, spec = compute_metrics_with_view_thresholds(test_boot_df, r, view_thresh)
            all_sens['by_view'][i, j] = sens
            all_spec['by_view'][i, j] = spec

    print('===============================')
    print(f'Bootstrap Results for model={model_name}, data={dataset_name}')
    print('Comparison to White patients:')
    print('thresh_type race sens_diff spec_diff')
    for t in thresh_tags:
        print(f'For thresholds: {t}')
        for j, r in enumerate(R_LABELS[:-1]):
            print(f'  For race: {r}')
            diffs = {}
            diffs['sens'] = all_sens[t][:, j] - all_sens[t][:, -1]
            diffs['spec'] = all_spec[t][:, j] - all_spec[t][:, -1]
            for m in ['sens', 'spec']:
                print(f'   {m} mean diff and CI:')
                ci_low = np.percentile(diffs[m], 2.5)
                ci_hi = np.percentile(diffs[m], 97.5)
                ci_mean = np.mean(diffs[m])
                std = np.std(diffs[m])
                print(f'     {ci_mean:2.2f} lo:{ci_low:2.2f} hi:{ci_hi:2.2f} std:{std:2.2f}')

    print('')
    print('Comparison of Thresh per view vs original')
    for j, r in enumerate(R_LABELS[:-1]):
        print(f'  For race: {r}')
        diffs = {}
        for t in thresh_tags:
            diffs[(t, 'sens')] = all_sens[t][:, j] - all_sens[t][:, -1]
            diffs[(t, 'spec')] = all_spec[t][:, j] - all_spec[t][:, -1]
        agg_diffs = {}
        for m in ['sens', 'spec']:
            agg_diffs[m] = diffs[('by_view', m)] - diffs[('orig', m)]
            print(f'   {m} mean diff and CI:')
            ci_low = np.percentile(agg_diffs[m], 2.5)
            ci_hi = np.percentile(agg_diffs[m], 97.5)
            ci_mean = np.mean(agg_diffs[m])
            std = np.std(agg_diffs[m])
            print(f'     {ci_mean:2.2f} lo:{ci_low:2.2f} hi:{ci_hi:2.2f} std:{std:2.2f}')

    print('====================')
    out_dir = PROJECT_DIR + 'disparity_analysis/'
    out_dir += 'bootstrap_outputs/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    con_tag = '-confounder_resampled' if confounder_resampled else ''
    dicom_tag = '_dicoms' if use_dicoms else ''
    np.savez(out_dir + f'{model_name}-{dataset_name}{con_tag}{dicom_tag}.npy', sens=all_sens, spec=all_spec, auc=all_auc)


def get_bootstrap_df(orig_df, df2=None):
    all_studies = orig_df.study_id.unique()
    sampled_studies = np.random.choice(all_studies, len(all_studies), replace=True)
    df_by_study = orig_df.set_index('study_id')
    if df2 is not None:
        df2_by_study = df2.set_index('study_id')
        return df_by_study.loc[sampled_studies], df2_by_study.loc[sampled_studies]
    else:
        return df_by_study.loc[sampled_studies]


def run_view_percentage_bootstrap(dataset, n_boot=2000):
    # this is based on plotting.py: create_score_change_by_view_plot
    train_df = load_split_metadf(dataset, 'train')  # get counts from train
    train_df = train_df[train_df.Mapped_Race.isin(R_LABELS)]

    if dataset == 'cxp':
        views = ['PA', 'AP', ('LATERAL', 'LL')]
        train_df = create_cxp_view_column(train_df)
        study_ids = []
        for p in train_df.Path.values:
            vals = p.split('/')
            study_ids.append(vals[-3] + '-' + vals[-2])
        train_df['study_id'] = study_ids
    else:
        views = ['PA', 'AP', ('LATERAL', 'LL'), 'AP_nonport', 'AP_port']
        train_df = create_mimic_isportable_column(train_df)
        train_df['view'] = train_df.ViewPosition

    boot_vals = {}
    for r in R_LABELS:
        for v in views:
            boot_vals[(r, v)] = np.zeros(n_boot)

    for i in tqdm.tqdm(range(n_boot), total=n_boot):
        boot_df = get_bootstrap_df(train_df)
        boot_perc_diffs = compute_view_percent_diffs(boot_df, views)
        for tup in boot_vals:
            boot_vals[tup][i] = boot_perc_diffs[tup]

    print('Std of percent diffs:')
    for tup in boot_vals:
        print(tup, np.std(boot_vals[tup]))

    out_dir = PROJECT_DIR + 'factor_correlation_analysis/bootstrap_analysis/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(out_dir + f'view_percent_diffs_bootstrap-{dataset}.pkl', 'wb') as f:
        pkl.dump(boot_vals, f)


def compute_view_percent_diffs(df, views):
    view_counts = {}
    for r in R_LABELS:
        r_idx = df.Mapped_Race == r
        for v in views:
            if 'port' in v:
                this_view = v.split('_')[0]
                is_port = v.split('_')[1] == 'port'
                idx = r_idx & (df.view == this_view) & (df.is_portable == is_port)
            elif isinstance(v, tuple):
                idx = r_idx & (df.view.isin(v))
            else:
                idx = r_idx & (df.view == v)
            view_counts[(r, v)] = idx.sum()

    base_percs = 100 * df.Mapped_Race.value_counts(normalize=True)

    perc_diffs = {}
    for v in views:
        view_total = np.sum([view_counts[(r, v)] for r in R_LABELS])
        for r in R_LABELS:
            this_perc = 100 * view_counts[(r, v)] / view_total
            perc_diffs[(r, v)] = 100 * (this_perc - base_percs.loc[r]) / base_percs.loc[r]

    return perc_diffs


def run_view_score_diff_bootstrap(model_name, dataset, n_boot=2000, confounder_resampled=False, use_dicoms=False, bmi_resample=False):
    con_tag = '-confounder_resampled' if confounder_resampled else ''
    dicom_tag = '_dicoms' if use_dicoms else ''
    bmi_tag = '-bmi_resampled' if bmi_resample else ''
    pred_df = load_pred_df(model_name, dataset, split='test', merge_labels=False,
                           confounder_resampled=confounder_resampled, use_dicoms=use_dicoms)

    if bmi_resample:
        resampled_df = load_split_metadf(dataset, 'test', confounder_resampled='bmi')
        pred_df['dicom_id'] = [p.split('/')[-1][:-4] for p in pred_df['Path'].values]
        pred_df.set_index('dicom_id', inplace=True)
        pred_df = pred_df.loc[resampled_df.dicom_id]
        pred_df.reset_index(inplace=True)

    if dataset == 'cxp':
        views = ['AP', 'PA', ('LATERAL', 'LL')]
    else:
        views = ['PA', 'AP', ('LATERAL', 'LL'), 'AP_nonport', 'AP_port']
        test_df = load_split_metadf(dataset, 'test')
        proc_map = test_df[['PerformedProcedureStepDescription', 'dicom_id']].set_index('dicom_id')
        pred_df['PerformedProcedureStepDescription'] = pred_df.dicom_id.map(
            proc_map['PerformedProcedureStepDescription'])
        pred_df = create_mimic_isportable_column(pred_df)

    boot_vals = {}
    for r in R_LABELS:
        for v in views:
            boot_vals[(r, v)] = np.zeros(n_boot)

    for i in tqdm.tqdm(range(n_boot), total=n_boot):
        boot_df = get_bootstrap_df(pred_df)
        boot_perc_diffs = compute_view_score_percent_diffs(boot_df, views)
        for tup in boot_vals:
            boot_vals[tup][i] = boot_perc_diffs[tup]

    print('Std of score percent diffs:')
    for tup in boot_vals:
        print(tup, np.std(boot_vals[tup]))

    out_dir = PROJECT_DIR + 'factor_correlation_analysis/bootstrap_analysis/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    fname = f'view_score_percent_diffs_bootstrap-{model_name}-{dataset}{con_tag}{bmi_tag}{dicom_tag}.pkl'
    print('saving: ' + fname)
    with open(out_dir + fname, 'wb') as f:
        pkl.dump(boot_vals, f)


def compute_view_score_percent_diffs(df, views):
    orig_means = {r: df.groupby('Mapped_Race')[f'Pred_{r}'].mean().mean() for r in R_LABELS}

    perc_diffs = {}
    for v in views:
        if 'port' in v:
            this_view = v.split('_')[0]
            is_port = v.split('_')[1] == 'port'
            idx = (df.View == this_view) & (df.is_portable == is_port)
        elif isinstance(v, tuple):
            idx = df.View.isin(v)
        else:
            idx = df.View == v
        view_df = df.loc[idx]
        view_means = {r: view_df.groupby('Mapped_Race')[f'Pred_{r}'].mean().mean() for r in R_LABELS}
        for r in R_LABELS:
            perc_diffs[(r, v)] = 100 * (view_means[r] - orig_means[r]) / orig_means[r]
    return perc_diffs


def compute_view_statistics(dataset, confounder_resampled=False):
    if confounder_resampled:
        df = load_split_metadf(dataset, 'test', confounder_resampled=confounder_resampled)
    else:
        # compute for entire dataset
        if dataset == 'cxp':
            meta_path = '../../../datasets/CheXpert-v1.0-small/train-with_race.csv'
        else:
            meta_path = MIMIC_BASE_DIR + 'mimic-cxr-2.0.0-metadata-with_race.csv'

        df = pd.read_csv(meta_path)

    if dataset == 'cxp':
        df = create_cxp_view_column(df)
    else:
        df = create_mimic_isportable_column(df)
        df['view'] = df.ViewPosition

    df.loc[df.view == 'LL', 'view'] = 'LATERAL'
    df.loc[pd.isnull(df.view), 'view'] = 'NaN'

    con_tag = '-confounder_resampled' if confounder_resampled else ''
    out_path = PROJECT_DIR + f'factor_correlation_analysis/view_statistics-{dataset}{con_tag}.txt'
    with open(out_path, 'w') as sys.stdout:

        for r in ['All'] + R_LABELS:
            if r == 'All':
                race_df = df
            else:
                race_df = df.query(f"Mapped_Race=='{r}'")

            print(f'For Race={r}\n')
            print(f'total count: {len(race_df)}')
            print('--counts--')
            print(race_df.view.value_counts())
            print('\n')
            print('--proportions--')
            print(race_df.view.value_counts(normalize=True))

            if dataset == 'mimic':
                view_df = race_df.query("view=='AP'")
                print('\n ---by AP portable')
                print(view_df.is_portable.value_counts())
                print('\n')
                print(view_df.is_portable.value_counts(normalize=True))
                print('\n ---total portable - ALL VIEWS')
                print(race_df.is_portable.value_counts())
                print('\n')
                print(race_df.is_portable.value_counts(normalize=True))
                print('\n ---portable by view')
                print(race_df[race_df.is_portable == True]['view'].value_counts())
                print('\n')
                print(race_df[race_df.is_portable == True]['view'].value_counts(normalize=True))

            print('\n\n\n')


def compute_race_auc_results(model_name, dataset_name, split, use_dicoms=False, confounder_resampled=False, bmi_resample=False):
    d_tag = '-window1_dicoms' if use_dicoms else ''  # use default windowing
    con_tag = '-confounder_resampled' if confounder_resampled else ''
    pred_path = os.path.join(PROJECT_DIR + 'prediction_dfs', model_name + '-best', dataset_name + '-' + split + con_tag + d_tag + '.csv')
    pred_df = pd.read_csv(pred_path)

    # for BMI, just resampling post hoc instead of initially running on resampled list (more efficient, same results)
    if bmi_resample:
        resampled_df = load_split_metadf(dataset_name, split, confounder_resampled='bmi')
        pred_df['dicom_id'] = [p.split('/')[-1][:-4] for p in pred_df['Path'].values]
        pred_df.set_index('dicom_id', inplace=True)
        pred_df = pred_df.loc[resampled_df.dicom_id]
        pred_df.reset_index(inplace=True)

    print('loading ' + pred_path)
    idx = ~pd.isnull(pred_df[f'Pred_{R_LABELS[0]}'])
    print('% not Nan: ', 100 * idx.mean())
    pred_df = pred_df.loc[idx]

    print('Race AUC analysis for {}, {}, {}'.format(model_name, dataset_name, split))
    print('Race: AUC CI')
    for r in R_LABELS:
        y = pred_df.Mapped_Race == r
        y = y.values.astype(np.int)
        yhat = pred_df[f'Pred_{r}'].values
        auc = roc_auc_score(y, yhat)

        ci = delong_ci(y, yhat)
        print(f'{r}: {auc:2.3f} {ci[0]:2.3f} {ci[1]:2.3f}')


def compare_orig_and_resampled_sets(dataset_name):
    dfs = {}
    dfs['orig'] = load_split_metadf(dataset_name, 'test')
    dfs['resampled'] = load_split_metadf(dataset_name, 'test', confounder_resampled=True)

    orig_confounder_df = pd.read_csv('../torchxrayvision/data/' + f'{dataset_name}_confounder_df.csv',
                                     index_col=0)  # previously computed
    orig_confounder_df['IsMale'] = orig_confounder_df['Sex'] == 'M'

    for tag in dfs:
        if dataset_name == 'cxp':
            dfs[tag] = create_cxp_view_column(dfs[tag])
        else:
            dfs[tag] = create_mimic_isportable_column(dfs[tag])
            dfs[tag]['view'] = dfs[tag]['ViewPosition']
        dfs[tag].loc[dfs[tag].view == 'LL', 'view'] = 'LATERAL'
        dfs[tag].loc[pd.isnull(dfs[tag].view), 'view'] = 'NaN'

    confounder_dfs = {}
    for tag in dfs:
        dfs[tag] = dfs[tag][dfs[tag].Mapped_Race.isin(R_LABELS)]
        if dataset_name == 'mimic':
            dfs[tag].set_index('dicom_id', inplace=True)
        elif dataset_name == 'cxp':
            dfs[tag].set_index('Path', inplace=True)
        confounder_dfs[tag] = orig_confounder_df.loc[dfs[tag].index]

    factors = ['Age', 'IsMale'] + np.sort(CXP_LABELS).tolist()
    out_data = []
    for factor in factors:
        for tag in ['orig', 'resampled']:
            r_means = confounder_dfs[tag].groupby('Mapped_Race')[factor].mean()
            this_row = [factor, tag] + [r_means.loc[r] for r in R_LABELS]
            out_data.append(this_row)

    for tag in ['orig', 'resampled']:
        r_props = dfs[tag]['Mapped_Race'].value_counts(normalize=True)
        this_row = ['Mapped_Race', tag] + [r_props.loc[r] for r in R_LABELS]
        out_data.append(this_row)

    for view in ['AP', 'PA', 'LATERAL']:
        for tag in ['orig', 'resampled']:
            view_props = dfs[tag].groupby('Mapped_Race')['view'].value_counts(normalize=True)
            this_row = [view, tag] + [view_props.loc[(r, view)] for r in R_LABELS]
            out_data.append(this_row)

    if dataset_name == 'mimic':
        for tag in ['orig', 'resampled']:
            port_props = dfs[tag].groupby('Mapped_Race')['is_portable'].mean()
            this_row = ['is_portable', tag] + [port_props.loc[r] for r in R_LABELS]
            out_data.append(this_row)

    out_df = pd.DataFrame(out_data, columns=['Factor', 'Tag'] + R_LABELS)
    out_df.to_csv(PROJECT_DIR + f'factor_correlation_analysis/orig_v_resampled_breakdown-{dataset_name}.csv', index=False)


if __name__ == '__main__':
    models = ['cxp_disease_orig']
    for m in models:
        for d in ['mimic']:
            for v in [False, True]:
                run_metric_analysis(m, d, v, target_sens='balanced', confounder_resampled=True)
