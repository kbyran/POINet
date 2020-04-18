import time
import logging
import prettytable as pt
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]
        ) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'] for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class PoseMetric(object):
    def __init__(self, pTest):
        self.p = pTest
        self.coco = COCO(pTest.annotation)
        self.results = None
        self.logger = None

    def eval_func(self):
        """ Evaluation with COCO keypoints metric. """
        tik = time.time()
        pred = self.coco.loadRes(self.results)
        self.coco_eval = COCOeval(self.coco, pred, 'keypoints')
        self.coco_eval.params.useSegm = None
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        tok = time.time()
        self.logger.info("eval uses {:.1f}".format(tok - tik))

    def parse_results(self, results):
        tik = time.time()
        if self.p.oks_nms:
            # oks nms
            nms_results = []
            images = defaultdict(list)
            for result in results:
                images[result["image_id"]].append(result)
            for image in images:
                img_kpts = images[image]
                for n_p in img_kpts:
                    box_kpt = np.asarray(n_p['keypoints']).reshape((self.p.num_joints, 3))
                    box_score = n_p['score']
                    kpt_score = 0
                    valid_num = 0
                    for n_jt in range(0, self.p.num_joints):
                        t_s = box_kpt[n_jt][2]
                        if t_s > self.p.in_vis_thre:
                            kpt_score = kpt_score + t_s
                            valid_num = valid_num + 1
                    if valid_num != 0:
                        kpt_score = kpt_score / valid_num
                    # rescoring
                    n_p['score'] = kpt_score * box_score
                keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))],
                               self.p.oks_thre)
                if len(keep) == 0:
                    nms_results.extend(img_kpts)
                else:
                    nms_results.extend([img_kpts[_keep] for _keep in keep])
            self.results = nms_results
        else:
            self.results = results

        self.logger.info(len(self.results))
        tok = time.time()
        self.logger.info("parse uses {:.1f}".format(tok - tik))

    def process(self, results, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        self.parse_results(results)
        self.eval_func()

    def summarize(self):
        self.coco_eval.summarize()
        stats = self.coco_eval.stats
        table = pt.PrettyTable()
        field_names = ["AP (0.5: 0.95)", "AP (0.5)", "AP (0.7)"]
        row_values = ["{:0.3f}".format(stats[i]) for i in range(3)]
        max_name_length = max([len(name) for name in field_names + row_values])
        field_names = [name.rjust(max_name_length, " ") for name in field_names]
        row_values = [name.rjust(max_name_length, " ") for name in row_values]
        table.field_names = field_names
        table.add_row(row_values)
        self.logger.info("validation results: \n{}".format(table))


if __name__ == "__main__":
    import json
    path = "logs/simple_pose_r50v1/coco_keypoints_COCO_val2017_result.json"
    with open(path, "r") as f:
        results = json.load(f)

    class Config(object):
        annotation = "data/coco_keypoints/annotations/person_keypoints_val2017.json"
        num_joints = 17
        in_vis_thre = 0.0
        oks_thre = 0.5
        oks_nms = True

    pConfig = Config()

    metric = PoseMetric(pConfig)
    metric.process(results)
    metric.summarize()
