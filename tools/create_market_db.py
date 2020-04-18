import argparse
import os
import pickle as pkl
import glob
import re

split_mappings = {
    "train": "bounding_box_train",
    "query": "query",
    "gallery": "bounding_box_test"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate database for Market1501 dataset")
    parser.add_argument("--dataset-split", help="dataset split, e.g. train, query, gallery",
                        default="train,query,gallery")

    args = parser.parse_args()
    return args.dataset_split


def print_database_statistics(split, db):
    imgs, pids, cams = set(), set(), set()
    for rec in db:
        imgs.add(rec["image_url"])
        pids.add(rec["pid"])
        cams.add(rec["cid"])
    assert len(imgs) == len(db)
    num_imgs = len(imgs)
    num_pids = len(pids)
    num_cams = len(cams)
    print("Statistics for {} subset: #ids={}, #images={}, #cameras={}".format(
        split, num_pids, num_imgs, num_cams))


def Generate_groundtruth_database(split):
    relabel = True if split == "train" else False
    dataset_dir = os.path.join("data/market1501", split_mappings[split])
    img_paths = glob.glob(os.path.join(dataset_dir, "*jpg"))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    pid_container = set()
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue  # junk images are just ignored
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    db = []
    for im_id, img_path in enumerate(img_paths):
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue  # junk images are just ignored
        assert 0 <= pid <= 1501  # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1  # index starts from 0
        if relabel:
            pid = pid2label[pid]
        rec = {
            "im_id": im_id,
            "image_url": img_path,
            "pid": pid,
            "cid": camid,
            "split": split
        }
        db.append(rec)

    print_database_statistics(split, db)

    return db


if __name__ == "__main__":
    splits = parse_args()
    os.makedirs("data/cache", exist_ok=True)
    for split in splits.split(","):
        assert split in split_mappings
        db = Generate_groundtruth_database(split)
        dump_path = "data/cache/{}_{}.db".format("market1501", split)
        with open(dump_path, "wb") as f:
            pkl.dump(db, f)
            print("Subset {} for Market1501 is created in {}.".format(split, dump_path))
