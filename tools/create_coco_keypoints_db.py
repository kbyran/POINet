import argparse
import os
import json
import pickle as pkl
import numpy as np
from pycocotools.coco import COCO


dataset_annotation_mapping = {
    "train2017": "annotations/person_keypoints_train2017.json",
    "val2017": "annotations/person_keypoints_val2017.json",
    "COCO_val2017_detections_AP_H_56_person": "annotations/person_keypoints_val2017.json"
}

dataset_detection_mapping = {
    "COCO_val2017_detections_AP_H_56_person":
        "person_detection_results/COCO_val2017_detections_AP_H_56_person.json",
    "COCO_test-dev2017_detections_AP_H_609_person":
        "person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json",
}

dataset_image_mapping = {
    "COCO_val2017_detections_AP_H_56_person": "val2017",
    "COCO_test-dev2017_detections_AP_H_609_person": "test2017"
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
    dataset_type = "person_detections" if dataset_split in dataset_detection_mapping \
        else "person_keypoints"
    annotation_path = "data/%s/%s" % (dataset_name, dataset_annotation_mapping[dataset_split])
    assert os.path.exists(annotation_path)
    if dataset_type == "person_detections":
        detection_path = "data/%s/%s" % (dataset_name, dataset_detection_mapping[dataset_split])
        detections = json.load(open(detection_path, "r"))
        imgs = {}
        for d in detections:
            if d["image_id"] not in imgs:
                imgs[d["image_id"]] = [d]
            else:
                imgs[d["image_id"]] += [d]

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
        split = dataset_split in dataset_image_mapping and dataset_image_mapping[
            dataset_split] or dataset_split

        image_url = 'data/%s/images/%s/%s' % (dataset_name, split, im_filename)
        assert os.path.exists(image_url)

        if dataset_type == "person_keypoints":
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

                if max(inst['keypoints']) > 0 and np.sum(joints_vis) > 0 and \
                        inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                    assert inst['category_id'] == 1, inst['category_id']
                    inst['clean_bbox'] = np.array([x1, y1, x2, y2])
                    inst['joints_3d'] = joints_3d
                    inst['joints_vis'] = joints_vis
                    roi_rec = {
                        'image_url': image_url,
                        'im_id': img_id,
                        'h': im_h,
                        'w': im_w,
                        'area': inst['area'],
                        'gt_class': datasetid_to_trainid[inst['category_id']],
                        'gt_bbox': inst['clean_bbox'],
                        # 'gt_poly': 'segmentation' in inst and inst['segmentation'] or None,
                        'gt_joints_3d': inst['joints_3d'],
                        'gt_joints_vis': inst['joints_vis'],
                        'gt_kp': inst['keypoints'],
                        'gt_num_kp': inst["num_keypoints"],
                        'flipped': False
                    }
                    # print(roi_rec)

                    roidb.append(roi_rec)
        else:
            if img_id in imgs:
                for d in imgs[img_id]:
                    if d["score"] > 0:
                        assert d["category_id"] == 1
                        x, y, box_w, box_h = d['bbox']
                        x1 = max(0, x)
                        y1 = max(0, y)
                        x2 = min(im_w - 1, x1 + max(0, box_w - 1))
                        y2 = min(im_h - 1, y1 + max(0, box_h - 1))
                        roi_rec = {
                            'image_url': image_url,
                            'im_id': img_id,
                            'h': im_h,
                            'w': im_w,
                            'area': box_w * box_h,
                            'gt_class': d['category_id'],
                            'gt_bbox': np.array([x1, y1, x2, y2]),
                            'flipped': False
                        }
                        roidb.append(roi_rec)

    print(len(roidb))

    return roidb


if __name__ == "__main__":
    d, dsplit = parse_args()
    roidb = generate_groundtruth_database(d, dsplit)
    os.makedirs("data/cache", exist_ok=True)
    with open("data/cache/%s_%s.db" % (d, dsplit), "wb") as fout:
        pkl.dump(roidb, fout)
