import argparse
import os
import pickle as pkl
import numpy as np
from pycocotools.coco import COCO


dataset_split_mapping = {
    "train": "train2014",
    "valminusminival": "val2014",
    "minival": "val2014",
    "test": "test2014"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate SimpleDet GroundTruth Database for COCO-like dataset')
    parser.add_argument('--dataset', help='dataset name', type=str)
    parser.add_argument('--dataset-split',
                        help='dataset split, e.g. train2017, minival2014', type=str)

    args = parser.parse_args()
    return args.dataset, args.dataset_split


def generate_groundtruth_database(dataset_name, dataset_split):
    annotation_path = "data/%s/annotations/densepose_coco_2014_%s.json" % (
        dataset_name, dataset_split)
    # print(annotation_path)
    assert os.path.exists(annotation_path)
    num_joints = 17

    dataset = COCO(annotation_path)
    img_ids = dataset.getImgIds()
    roidb = []
    for img_id in img_ids:
        img_anno = dataset.loadImgs(img_id)[0]

        im_filename = img_anno['file_name']
        im_w = img_anno['width']
        im_h = img_anno['height']

        # split mapping is specific to coco as it uses annotation files to manage split
        split = dataset_split in dataset_split_mapping and dataset_split_mapping[
            dataset_split] or dataset_split

        image_url = 'data/%s/images/%s/%s' % (dataset_name, split, im_filename)
        assert os.path.exists(image_url)

        ins_anno_ids = dataset.getAnnIds(imgIds=img_id, iscrowd=False)
        trainid_to_datasetid = dict(
            {i + 1: cid for i, cid in enumerate(dataset.getCatIds())})  # 0 for bg
        datasetid_to_trainid = dict({cid: tid for tid, cid in trainid_to_datasetid.items()})
        instances = dataset.loadAnns(ins_anno_ids)

        # sanitize bboxes
        for inst in instances:
            # clip bboxes
            x, y, box_w, box_h = inst['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(im_w - 1, x1 + max(0, box_w - 1))
            y2 = min(im_h - 1, y1 + max(0, box_h - 1))

            # joints 3d: (num_joints, 3), 3 is for x, y, z position
            # joints_vis: (num_joints, ), 1 for visibility
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_vis = np.zeros((num_joints,), dtype=np.int)
            for i in range(num_joints):
                joints_3d[i, 0] = inst['keypoints'][i * 3 + 0]
                joints_3d[i, 1] = inst['keypoints'][i * 3 + 1]
                visible = min(1, inst['keypoints'][i * 3 + 2])
                joints_vis[i] = visible

            if x2 >= x1 and y2 >= y1 and 'dp_masks' in inst:
                assert inst['category_id'] == 1, inst['category_id']
                assert max(inst['keypoints']) > 0 and np.sum(joints_vis) > 0 and inst['area'] > 0
                inst['clean_bbox'] = np.array([x1, y1, x2, y2])
                inst['joints_3d'] = joints_3d
                inst['joints_vis'] = joints_vis
                roi_rec = {
                    'image_url': image_url,
                    'im_id': img_id,
                    'h': im_h,
                    'w': im_w,
                    'gt_class': datasetid_to_trainid[inst['category_id']],
                    'gt_bbox': inst['clean_bbox'],
                    # 'gt_poly': 'segmentation' in inst and inst['segmentation'] or None,
                    'gt_joints_3d': inst['joints_3d'],
                    'gt_joints_vis': inst['joints_vis'],
                    'gt_kp': inst['keypoints'],
                    'gt_num_kp': inst['num_keypoints'],
                    'dp_masks': inst['dp_masks'],
                    'dp_I': inst['dp_I'],
                    'dp_U': inst['dp_U'],
                    'dp_V': inst['dp_V'],
                    'dp_x': inst['dp_x'],
                    'dp_y': inst['dp_y'],
                    'flipped': False
                }
                # print(roi_rec)

                roidb.append(roi_rec)

    # print(len(roidb))
    return roidb


if __name__ == "__main__":
    d, dsplit = parse_args()
    roidb = generate_groundtruth_database(d, dsplit)
    os.makedirs("data/cache", exist_ok=True)
    with open("data/cache/%s_%s.db" % (d, dsplit), "wb") as fout:
        pkl.dump(roidb, fout)
