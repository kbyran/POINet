import argparse
import os
import pickle as pkl
from scipy.io import loadmat
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate database for RAPv2 dataset")
    parser.add_argument("--dataset-split", help="dataset split, e.g. train, val, test",
                        default="train,val,test,trainval")

    args = parser.parse_args()
    return args.dataset_split


def print_database_statistics(split, db):
    imgs, labels = set(), list()
    for rec in db:
        assert -1 not in rec["labels"]
        imgs.add(rec["image_url"])
        labels.append(rec["labels"])
    assert len(imgs) == len(db)
    num_imgs = len(imgs)
    print("Statistics for {} subset: #images={}.".format(split, num_imgs))
    labels_weight = np.mean(np.asarray(labels).astype('float32') == 1, axis=0).tolist()
    print("Labels weight is {}".format(", ".join(["%.6f" % l for l in labels_weight])))


def Generate_groundtruth_database(split):
    data = loadmat(open("data/RAPv2/RAP_annotation/RAP_annotation.mat", "rb"))
    annotation = data["RAP_annotation"][0][0]  # index for annotations
    # we use partition 0
    if split == "train":
        partition = (annotation[4][0, 0][0][0][0][0, :] - 1).tolist()
    elif split == "val":
        partition = (annotation[4][0, 0][0][0][1][0, :] - 1).tolist()
    elif split == "test":
        partition = (annotation[4][0, 0][0][0][2][0, :] - 1).tolist()
    elif split == "trainval":
        partition = (annotation[4][0, 0][0][0][0][0, :] - 1).tolist() + \
            (annotation[4][0, 0][0][0][1][0, :] - 1).tolist()

    selected_attribute = (annotation[3][0, :] - 1).tolist()

    db = []
    for idx in partition:
        img_path = "data/RAPv2/RAP_dataset/" + annotation[0][idx][0][0]
        labels = annotation[1][idx][selected_attribute].tolist()
        rec = {
            "im_id": idx,
            "image_url": img_path,
            "labels": labels,
            "split": split
        }
        db.append(rec)
        # print(rec)

    print_database_statistics(split, db)

    return db


if __name__ == "__main__":
    splits = parse_args()
    os.makedirs("data/cache", exist_ok=True)
    for split in splits.split(","):
        db = Generate_groundtruth_database(split)
        dump_path = "data/cache/{}_{}.db".format("rapv2", split)
        with open(dump_path, "wb") as f:
            pkl.dump(db, f)
            print("Subset {} for RAPv2 is created in {}.".format(split, dump_path))
