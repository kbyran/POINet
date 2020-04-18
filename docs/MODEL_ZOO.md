# Model Zoo

This page summarizes all the trained models.

## Person re-identification

### Market-1501

model | backbone | GPU | FT | RR | mAP | Rank-1 | Rank-5 | Rank-10 | download
---|---|---|---|---|---|---|---|---|---
[gluoncv_baseline](https://github.com/kbyran/poi/tree/master/configs/reid_strong_baseline/gluoncv_baseline_market1501_r50v1_xent.py) | R50v1 | 1 | no | no | 79.6 | 91.7 | 97.0 | 98.1 | /
[strong_baseline](https://github.com/kbyran/poi/tree/master/configs/strong_baseline_market1501_r50v1_xent_tri_cent.py) | R50v1 | 1 | no | no | 86.3 | 94.6 | 98.3 | 99.1 | /
[strong_baseline](https://github.com/kbyran/poi/tree/master/configs/strong_baseline_market1501_r50v1_xent_tri_cent_gpu8.py) | R50v1 | 8 | no | no | 86.2 | 94.5 | 98.3 | 98.9 | /

- `FT` for flip-test, `RR` for re-ranking.

## Pedestrian Attribute Recognition

### RAP v2.0

model | backbone | GPU | mA | Acc | Prec | Rec | F1 | download
---|---|---|---|---|---|---|---|---
[deepmar](https://github.com/kbyran/poi/tree/master/configs/attr_deepmar/deepmar_rapv2_r50v1.py) | R50v1 | 1 | 0.7591 | 0.6176 | 0.7528 | 0.7504 | 0.7516 | /

## Pose Estimation

### COCO keypoints

model | backbone | GPU | AP (0.5: 0.95) | AP (0.5) | AP (0.7) | download
---|---|---|---|---|---|---
[simple_pose](https://github.com/kbyran/poi/tree/master/configs/pose_simple_baseline/simple_pose_r50v1.py) | R50v1 | 1 | 0.702 | 0.913 | 0.776 | /

## Dense Human Pose Estimation (DensePose)

### DensePose-COCO

model | backbone | GPU | AP (0.5: 0.95) | AP (0.5) | AP (0.7) | AP (medium) | AP (large) | download
---|---|---|---|---|---|---|---|---
[densepose_baseline](https://github.com/kbyran/poi/tree/master/configs/densepose_baseline/dense_pose_r50v1.py) | R50v1 | 1 | 0.600 | 0.915 | 0.660 | 0.636 | 0.607 | /
