### Code for "Acquisition parameters influence AI recognition of race in chest x-rays and mitigating these factors reduces underdiagnosis bias". Nature Communications (2024).



## Prereqs and Setup
- Model training relies on a custom [fork](https://github.com/lotterlab/torchxrayvision/tree/tech_bias_analysis) of the Torchxrayvision (Cohen et al., 2020) library.
- Python 3.9.12 was used with a conda environment using the packages contained in cxr_packages.txt
- AI model training/testing was performed on a combination of Nvidia TitanX GPUs and A100 GPUs. Model training typically takes ~1 day, with evaluation taking a few hours.
- The CheXpert dataset can be downloaded from: https://stanfordmlgroup.github.io/competitions/chexpert/
- The MIMIC-CXR dataset can be downloaded from: https://physionet.org/content/mimic-cxr/2.0.0/
- The constants.py file contains constants that can be adjusted (e.g., the location of where datasets are stored)

## Model Training
- Training/val/testing splits were created using the `create_cxp_splits()` and `create_mimic_splits()` in data_utils.py
- Model training was performed using the torchxrayvision/scripts/train_model.py script with the following commands:

Training models to predict patient race:
```
python train_model.py --name cxp_race_orig --dataset chex --model densenet --im_size 224 --gpu 1 --label_type race3 --fixed_splits --no_taskweights --num_epochs 40 --threads 12 --all_views
python train_model.py --name mimic_race_orig --dataset mimic_ch --model densenet --im_size 224 --gpu 1 --label_type race3 --fixed_splits --no_taskweights --num_epochs 40 --threads 12 --all_views
```

Training models to predict disease labels (original, no window width/field of view augmentation):
```
python train_model.py --name cxp_disease_orig --dataset chex --model densenet --im_size 224 --gpu 0 --fixed_splits --threads 12 --num_epochs 40 --all_views --imagenet_pretrained --use_no_finding
python train_model.py --name mimic_disease_orig --dataset mimic_ch --model densenet --im_size 224 --gpu 0 --fixed_splits --threads 12 --num_epochs 40 --all_views --imagenet_pretrained --use_no_finding
```

Training models to predict disease labels with window width and field of view data augmentation:
```
python train_model.py --name cxp_disease_aug --dataset chex --model densenet --im_size 224 --gpu 1 --fixed_splits --threads 12 --num_epochs 40 --all_views --imagenet_pretrained --data_aug_window_width_min 205 --data_aug_max_resize 269 --use_no_finding
python train_model.py --name mimic_disease_aug --dataset mimic_ch --model densenet --im_size 224 --gpu 1 --fixed_splits --threads 12 --num_epochs 40 --all_views --imagenet_pretrained --data_aug_window_width_min 205 --data_aug_max_resize 269 --use_no_finding
```
The "best" weights (as measured on the validation set during training) were chosen as the final weights for each model.

## Model Analysis
- Predictions on the test set were generated using the `run_predictions` function in analysis.py. For instance, to generate predictions across the different window width and field of view parameters, the following code snippet can be used:
  ```
  model_name = 'cxp_race_orig'
  checkpoint_name = 'best'
  dataset = 'cxp'
  for window_width in [None, 205, 218, 230, 243]: # increments of 5%
      for init_resize in [None, 235, 246, 258, 269]:
          run_predictions(model_name, checkpoint_name, dataset, split='test', window_width=window_width,
                          init_resize=init_resize, label_type='race')  # label_type set to 'pathology_wnofinding' for disease classification task
  ```
- The average predictions by race across the technical parameters (e.g., to be used for Figures 2 and 3) are subsequently computed by the `analyze_mean_predictions` function in analysis.py.

## Results and Plotting
- AUROC performance for the race prediction task is computed using the `compute_race_auc_results` function in results.py
- Figure 2 is created by running `create_window_resize_heatmap_plot` followed by running `create_aggregate_window_resize_heatmap_plot` in plotting.py
- Figure 3 is created by running `create_score_change_by_view_plot` followed by running `create_aggregate_score_change_by_view_plot` in plotting.py
- The "No Findings" prediction performance and underdiagnosis bias results are computed by running `run_metric_analysis` in results.py, where `use_per_view_threshs` can be set to True to implement the per-view threshold strategy. The `run_bootstrap` function is used to compute confidence intervals.
- Figure 4 is created by running `create_disparity_change_plot` in plotting.py


## Confounder analysis
- To perform the confounder analysis, the empirical proportions of the confounding variables are first computed for each dataset using the function `create_confounder_data` in data_utils.py
- The resampled test splits are created by running `create_confounder_controlled_split` in data_utils.py
- The models trained with resampling are trained using the same commands described in Model Training except appending `--multifactorial_class_balance` to the command line arguments
- The comparison of the original and resampled test splits in Supplementary Table 4 is computed by running `compare_orig_and_resampled_sets` in results.py
- Supplementary Figure 1 is created by running `create_aggregate_window_resize_heatmap_plot` with `version` set to `cxp_all` or `mimic_all` in plotting.py
- Supplementary Figure 2 is created by running `create_score_change_by_view_plot_comparison` in plotting.py
- Supplementary Figure 3 is created by first running the `run_metric_analysis` code described above with `confounder_resampled=True` or `use_dicoms=True` following by running `create_disparity_change_plot` in plotting.py for each model/approach. 

