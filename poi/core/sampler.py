import random
import numpy as np
from collections import defaultdict


class Sampler(object):
    """Base class for samplers.

    sampling index of db.
    """
    def __init__(self, **kwargs):
        raise NotImplementedError

    def __call__(self, **kwargs):
        raise NotImplementedError


class RegularSampler(Sampler):
    """Return samples without shuffles."""
    def __init__(self):
        pass

    def __call__(self, db):
        total_index = np.arange(len(db))
        return total_index


class RandomSampler(Sampler):
    """Randomly samples db with batch_size."""
    def __init__(self):
        pass

    def __call__(self, db):
        total_index = np.arange(len(db))
        np.random.shuffle(total_index)
        # print("pids: ", [db[idx]["pid"] for idx in total_index])

        return total_index


class TripletSampler(Sampler):
    """Sample for triplets
    Randomly sample N identities, then for each identity, randomly sample K instances.
    Therefore batch size is N * K.
    input: db, a list of instances
    output: total_index, a list of index
    """
    def __init__(self, pSampler):
        self.p = pSampler

    def __call__(self, db):
        p = self.p
        batch_image = p.batch_image
        num_instance = p.num_instance

        num_pids_per_batch = batch_image // num_instance
        index_dict = defaultdict(list)
        for index, input_record in enumerate(db):
            pid = input_record["pid"]
            index_dict[pid].append(index)

        pids = list(index_dict)
        batch_index_dict = defaultdict(list)

        # padding instances for each identity
        for pid in pids:
            index_this_pid = index_dict[pid]
            if len(index_this_pid) < num_instance:
                index_this_pid = np.random.choice(index_this_pid, size=num_instance, replace=True)
            random.shuffle(index_this_pid)
            split_index_this_pid = []
            for index in index_this_pid:
                split_index_this_pid.append(index)
                if len(split_index_this_pid) == num_instance:
                    batch_index_dict[pid].append(split_index_this_pid)
                    split_index_this_pid = []

        total_index = []

        while len(pids) >= num_pids_per_batch:
            selected_pids = random.sample(pids, num_pids_per_batch)
            for pid in selected_pids:
                batch_index = batch_index_dict[pid].pop(0)
                total_index += batch_index
                if len(batch_index_dict[pid]) == 0:
                    pids.remove(pid)

        # print("pids: ", [db[idx]["pid"] for idx in total_index])

        return total_index
