import time
import prettytable as pt
import numpy as np
import mxnet as mx
from poi.eval.reid.re_ranking import re_ranking, c_distance_opt
import logging


class ReIDMetric(object):
    def __init__(self, pTest):
        self.p = pTest
        self.gpu = self.p.gpus[0]
        self.distmat = None
        self.q_pids = None
        self.g_pids = None
        self.q_camids = None
        self.g_camids = None
        self.cmc = None
        self.mAP = None
        self.logger = None

    def eval_func(self):
        """ Evaluation with market1501 metric. """
        tik = time.time()
        p = self.p
        max_rank = p.max_rank
        num_q, num_g = self.distmat.shape
        self.logger.info("num_q: {}, num_g: {}".format(num_q, num_g))
        if num_g < max_rank:
            max_rank = num_g
            self.logger.info("Note: number of gallery samples is quite small, "
                             "got {} as max_rank".format(num_g))
        indices = np.argsort(self.distmat, axis=1)
        matches = (self.g_pids[indices] == self.q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = self.q_pids[q_idx]
            q_camid = self.q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
            keep = np.invert(remove)

            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1

            # compute average precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery."

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        self.cmc = all_cmc
        self.mAP = mAP
        tok = time.time()
        self.logger.info("eval uses {:.1f}".format(tok - tik))

    def parse_results(self, results):
        tik = time.time()
        p = self.p
        dist_type = p.dist_type

        q_pids = list()
        g_pids = list()
        q_camids = list()
        g_camids = list()
        q_features = list()
        g_features = list()
        for r in results:
            split = r["split"]
            if split.startswith("gallery"):
                g_pids.append(r["pid"])
                g_camids.append(r["cid"])
                g_features.append(r["feature"])
            elif split.endswith("query"):
                q_pids.append(r["pid"])
                q_camids.append(r["cid"])
                q_features.append(r["feature"])
            else:
                raise ValueError("No setting for split {}".format(split))

        tok1 = time.time()
        self.logger.info(
            "{} instances in gallery and {} in query.".format(len(g_pids), len(q_pids)))
        self.logger.info("collect gallery and query uses {:.1f}".format(tok1 - tik))

        self.q_pids = np.asarray(q_pids)
        self.g_pids = np.asarray(g_pids)
        self.q_camids = np.asarray(q_camids)
        self.g_camids = np.asarray(g_camids)

        if dist_type == "euclidean":
            q_features_np = np.array(q_features, dtype=np.float32)
            g_features_np = np.array(g_features, dtype=np.float32)
            distmat = c_distance_opt(q_features_np, g_features_np, ctx=mx.gpu(self.gpu),
                                     normalize=False)
            self.distmat = distmat
        elif dist_type == "cosine":
            q_features_np = np.array(q_features)
            g_features_np = np.array(g_features)
            distmat = c_distance_opt(q_features_np, g_features_np, ctx=mx.gpu(self.gpu),
                                     normalize=True)
            self.distmat = distmat
        elif dist_type == "reranking":
            self.distmat = re_ranking(q_features, g_features, k1=20, k2=6, lambda_value=0.3,
                                      ctx=mx.gpu(self.gpu))
        else:
            raise ValueError("No setting for dist type {}".format(dist_type))

        tok2 = time.time()
        self.logger.info("compute dist matrix uses {:.1f}".format(tok2 - tok1))

    def process(self, results, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        # parse results into gallery and query
        self.parse_results(results)
        # compute rank# and mAP
        self.eval_func()

    def summarize(self):
        table = pt.PrettyTable()
        field_names = ["mAP", "Rank-1", "Rank-5", "Rank-10"]
        row_values = []
        row_values.append("{:.1%}".format(self.mAP))
        for r in [1, 5, 10]:
            row_values.append("{:.1%}".format(self.cmc[r - 1]))
        max_name_length = max([len(name) for name in field_names + row_values])
        field_names = [name.rjust(max_name_length, " ") for name in field_names]
        row_values = [name.rjust(max_name_length, " ") for name in row_values]
        table.field_names = field_names
        table.add_row(row_values)
        self.logger.info("validation results: \n{}".format(table))


if __name__ == "__main__":
    import json
    path = "logs/strong_baseline_market1501_r50v1_xent_tri_cent/market1501_gallery_result.json"
    with open(path, "r") as f:
        results = json.load(f)

    class Config(object):
        gpus = [0]
        dist_type = "cosine"
        max_rank = 50

    pConfig = Config()

    metric = ReIDMetric(pConfig)
    metric.process(results)
    metric.summarize()
