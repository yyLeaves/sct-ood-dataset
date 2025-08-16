# sct-ood-dataset

## Labeling
### HU>2000
Using HU as a threshold is not ideal for all cases, as it may not account for variations in tissue density and composition.

Refer [notebook2](notebooks/2_ct_HU_stats_pelvis.ipynb) for following visualizations.

![HU Value Distribution](images/2_hu_value_distribution_example_cts.png)
![HU > 2000 Slice Ratios](images/2_hu_gt2000_slice_ratios.png)
![CT HU EDA](images/2_ct_hu_eda.png)

Refer [notebook3](notebooks/3_ct_HU_label_coverage_analysis.ipynb) for following visualizations.

![HU > 2000 Coverage per OOD Scan](images/3_hu2000_coverage_per_ood_scan.png)
![HU > 2000 Distribution Coverage](images/3_hu2000_distribution_coverage.png)
![HU > 2000 Scan Level Confusion Matrix](images/3_hu2000_scan_level_confusion_matrix.png)

### Per Scan Thresholding

Using max HU value from all other normal slices as global threshold gives good performance on slice-level segmentation.

Refer [notebook4](notebooks/4_ct_scan_visualization.ipynb) for following visualizations.
![CT Example](images/4_1PA133_ct_example.png)
![HU Threshold Example](images/4_1PA133_hu_thresh_example.png)

![CT Example](images/4_1PA136_ct_example.png)
![HU Threshold Example](images/4_1PA136_hu_thresh_example.png)

![CT Example](images/4_1PA147_ct_example.png)
![HU Threshold Example](images/4_1PA147_hu_thresh_example.png)

![CT Example](images/4_1PA151_ct_example.png)
![HU Threshold Example](images/4_1PA151_hu_thresh_example.png)

![CT Example](images/4_1PA152_ct_example.png)
![HU Threshold Example](images/4_1PA152_hu_thresh_example.png)

![CT Example](images/4_1PA169_ct_example.png)
![HU Threshold Example](images/4_1PA169_hu_thresh_example.png)