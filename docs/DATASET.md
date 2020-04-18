# Dataset

The datasets on diverse tasks could be used.

## Person re-identification

### Market-1501

Market-1501 is a widely-used person re-identification dataset, which contains 32,668 annotated bounding boxes of 1,501 identities in 6 cameras. The dataset could be downloaded in [link](https://pan.baidu.com/s/1ntIi2Op).

```
# unpack
unzip Market-1501-v15.09.zip
# link the dataset
ln -s Market-1501-v15.09.15 data/market1501

# the structure of the folder is as follows
data
└── market1501
     ├── bounding_box_test
     ├── bounding_box_train
     ├── gt_bbox
     ├── gt_query
     ├── query
     └── readme.txt

# create db for train/query/gallery subsets
python3 tools/create_market_db.py
```

```
@inproceedings{zheng2015scalable,
  title={Scalable Person Re-identification: A Benchmark},
  author={Zheng, Liang and Shen, Liyue and Tian, Lu and Wang, Shengjin and Wang, Jingdong and Tian, Qi},
  booktitle={Computer Vision, IEEE International Conference on},
  year={2015}
}
```

## Pedestrian Attribute Recognition

### RAP v2.0

The Richly Annotated Pedestrian (RAP) v2.0 is a large-scale datasets. 84,928 images are annotated with 72 kinds of attributes. However, 54 binary attributes are selected in our experiments as usual practice.
You should obtain license agreement in [link](https://drive.google.com/file/d/1hoPIB5NJKf3YGMvLFZnIYG5JDcZTxHph/) and request data from the author.

```
# unpack
mkdir -p data/RAPv2
unzip RAP_dataset.zip
unzip RAP_annotation.zip
# link the dataset
ln -s RAP_dataset data/RAPv2/
ln -s RAP_annotation data/RAPv2/
python3 tools/create_rapv2_db.py

# the structure of the folder is as follows
data
└── RAPv2
     ├── RAP_dataset
     └── RAP_annotation

# create db for train/val
python3 tools/create_rapv2_db.py
```

```
@article{li2018richly,
    title={A Richly Annotated Pedestrian Dataset for Person Retrieval in Real Surveillance Scenarios},
    author={Li, Dangwei and Zhang, Zhang and Chen, Xiaotang and Huang, Kaiqi},
    journal={IEEE Transactions on Image Processing},
    volume={28},
    number={4},
    pages={1575--1590},
    year={2019},
    publisher={IEEE}
}
```

## Pose Estimation

### COCO keypoints

COCO keypoints dataset contains annotations on person detection and keypoint. The images and annotations should be placed as follows,
```
data
└── coco_keypoints
     ├── annotations
     |    ├── person_keypoints_train2017.json
     |    └── person_keypoints_val2017.json
     └── images
          ├── train2017/
          |    ├── 000000000009.jpg
          |    ├── 000000000025.jpg
          |    └── ...
          └── val2017/
               ├── 000000000139.jpg
               ├── 000000000285.jpg
               └── ...
```

```
# create db for train2017
python3 tools/create_coco_keypoints_db.py --dataset coco_keypoints --dataset-split train2017

# create db for val2017
python3 tools/create_coco_keypoints_db.py --dataset coco_keypoints --dataset-split val2017
```

```
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```


## Dense Human Pose Estimation (DensePose)

### DensePose-COCO

DensePose-COCO builds the mapping from all human pixels of an RGB image to the 3D surface of the human body. Images and annotations are downloaded from COCO dataset.

```
# enter dataset root
mkdir -p data/coco_densepose

# get_DensePose_COCO.sh
mkdir -p data/coco_densepose/annotations && cd data/coco_densepose/annotations
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_valminusminival.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_minival.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_test.json

# get_densepose_uv.sh
mkdir -p data/coco_densepose/UV_data && cd data/coco_densepose/UV_data
wget https://dl.fbaipublicfiles.com/densepose/densepose_uv_data.tar.gz
tar xvf densepose_uv_data.tar.gz
rm densepose_uv_data.tar.gz

# Download eval_data
mkdir -p data/coco_densepose/eval_data && cd data/coco_densepose/eval_data
wget https://dl.fbaipublicfiles.com/densepose/densepose_eval_data.tar.gz
tar xvf densepose_eval_data.tar.gz
rm densepose_eval_data.tar.gz
```

```
data
└── coco_densepose
     ├── annotations
     |    ├── densepose_coco_2014_train.json
     |    ├── densepose_coco_2014_test.json
     |    ├── densepose_coco_2014_minival.json
     |    └── densepose_coco_2014_valminusminival.json
     ├── images
     |    ├── train2014/
     |    |    ├── COCO_train2014_000000000009.jpg
     |    |    ├── COCO_train2014_000000000025.jpg
     |    |    └── ...
     |    └── val2014/
     |         ├── COCO_val2014_000000000042.jpg
     |         ├── COCO_val2014_000000000073.jpg
     |         └── ...
     ├── UV_data
     |    ├── UV_Processed.mat
     |    └── UV_symmetry_transforms.mat
     └── eval_data
          ├── Pdist_matrix.mat
          ├── SMPL_subdiv.mat
          └── SMPL_SUBDIV_TRANSFORM.mat
```

```
# create db for train
python3 tools/create_coco_densepose_db.py --dataset coco_densepose --dataset-split train

# create db for minival
python3 tools/create_coco_densepose_db.py --dataset coco_densepose --dataset-split minival
```

```
@InProceedings{Guler2018DensePose,
title={DensePose: Dense Human Pose Estimation In The Wild},
author={R\{i}za Alp G\"uler, Natalia Neverova, Iasonas Kokkinos},
journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2018}
}
```
