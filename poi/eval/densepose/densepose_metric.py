import time
import logging
import prettytable as pt
import numpy as np
from pycocotools.coco import COCO
from poi.eval.densepose.densepose_cocoeval import denseposeCOCOeval


class DensePoseMetric(object):
    def __init__(self, pTest):
        self.p = pTest
        self.coco = COCO(pTest.annotation)
        self.results = None
        self.logger = None

    def eval_func(self):
        """ Evaluation with COCO keypoints metric. """
        tik = time.time()
        img_ids = self.coco.getImgIds()
        img_ids.sort()
        pred = self.coco.loadRes(self.results)
        self.coco_eval = denseposeCOCOeval(self.coco, pred, 'uv', 0.255)
        self.coco_eval.params.imgIds = img_ids
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        tok = time.time()
        self.logger.info("eval uses {:.1f}".format(tok - tik))

    def parse_results(self, results):
        tik = time.time()
        for result in results:
            uv = np.array(result["uv"])
            uv[1:3, :, :] = uv[1:3, :, :] * 255
            result["uv"] = uv.astype("uint8")
        self.results = results

        self.logger.info(len(self.results))
        tok = time.time()
        self.logger.info("parse uses {:.1f}".format(tok - tik))

    def process(self, results, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        self.parse_results(results)
        self.eval_func()

    def summarize(self):
        self.logger.info("validation results: ")
        self.coco_eval.summarize()
        stats = self.coco_eval.stats
        table = pt.PrettyTable()
        field_names = ["AP (0.5: 0.95)", "AP (0.5)", "AP (0.7)", "AP (medium)", "AP (large)"]
        row_values = ["{:0.3f}".format(stats[i]) for i in [0, 1, 6, 11, 12]]
        max_name_length = max([len(name) for name in field_names + row_values])
        field_names = [name.rjust(max_name_length, " ") for name in field_names]
        row_values = [name.rjust(max_name_length, " ") for name in row_values]
        table.field_names = field_names
        table.add_row(row_values)
        self.logger.info("validation results: \n{}".format(table))


if __name__ == "__main__":
    import json
    path = "logs/dense_pose_r50v1/coco_densepose_minival_result.json"
    with open(path, "r") as f:
        results = json.load(f)

    class Config(object):
        annotation = "data/coco_densepose/annotations/densepose_coco_2014_minival.json"

    pConfig = Config()

    metric = DensePoseMetric(pConfig)
    metric.process(results)
    metric.summarize()
