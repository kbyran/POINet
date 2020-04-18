import time
import prettytable as pt
import numpy as np
import logging


def attribute_evaluate(gt_result, pt_result, logger):
    """
    Input: gt_result, pt_result, N*L, with 0/1
    Output: result, a dictionary, including label-based and instance-based evaluation
            label-based: label_pos_acc, label_neg_acc, label_acc
            instance-based: instance_acc, instance_precision, instance_recall, instance_F1
    """
    # obtain the label-based and instance-based accuracy
    # compute the label-based accuracy
    if gt_result.shape != pt_result.shape:
        logger.info('Shape beteen groundtruth and predicted results are different')
    # compute the label-based accuracy
    result = {}
    gt_pos = np.sum((gt_result == 1).astype(float), axis=0)
    gt_neg = np.sum((gt_result == 0).astype(float), axis=0)
    pt_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=0)
    pt_neg = np.sum((gt_result == 0).astype(float) * (pt_result == 0).astype(float), axis=0)
    label_pos_acc = 1.0 * pt_pos / gt_pos
    label_neg_acc = 1.0 * pt_neg / gt_neg
    label_acc = (label_pos_acc + label_neg_acc) / 2
    result['label_pos_acc'] = label_pos_acc
    result['label_neg_acc'] = label_neg_acc
    result['label_acc'] = label_acc
    # compute the instance-based accuracy
    # precision
    gt_pos = np.sum((gt_result == 1).astype(float), axis=1)
    pt_pos = np.sum((pt_result == 1).astype(float), axis=1)
    intersect_pos = np.sum(
        (gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=1)
    union_pos = np.sum(((gt_result == 1) + (pt_result == 1)).astype(float), axis=1)
    # avoid empty label in predicted results
    cnt_eff = float(gt_result.shape[0])
    for iter, key in enumerate(gt_pos):
        if key == 0:
            union_pos[iter] = 1
            pt_pos[iter] = 1
            gt_pos[iter] = 1
            cnt_eff = cnt_eff - 1
            continue
        if pt_pos[iter] == 0:
            pt_pos[iter] = 1
    instance_acc = np.sum(intersect_pos / union_pos) / cnt_eff
    instance_precision = np.sum(intersect_pos / pt_pos) / cnt_eff
    instance_recall = np.sum(intersect_pos / gt_pos) / cnt_eff
    instance_F1 = 2 * instance_precision * instance_recall / (instance_precision + instance_recall)
    result['instance_acc'] = instance_acc
    result['instance_precision'] = instance_precision
    result['instance_recall'] = instance_recall
    result['instance_F1'] = instance_F1
    return result


class AttributeMetric(object):
    def __init__(self, pTest):
        self.p = pTest
        self.gt_result = None
        self.pt_result = None
        self.results = None
        self.logger = None

    def eval_func(self):
        """ Evaluation with COCO keypoints metric. """
        tik = time.time()
        self.results = attribute_evaluate(self.gt_result, self.pt_result, self.logger)
        tok = time.time()
        self.logger.info("eval uses {:.1f}".format(tok - tik))

    def parse_results(self, results):
        tik = time.time()
        self.results = results
        self.logger.info("%d instances" % len(self.results))
        gt_result = []
        pt_result = []
        for ret in self.results:
            gt_result.append(ret["labels"])
            pt_result.append(ret["logits"])
        self.gt_result = np.array(gt_result)
        self.pt_result = np.array(pt_result)
        tok = time.time()
        self.logger.info("parse uses {:.1f}".format(tok - tik))

    def process(self, results, logger=None):
        self.logger = logger if logger is not None else logging.getLogger()
        self.parse_results(results)
        self.eval_func()

    def summarize(self):
        label_table = pt.PrettyTable()
        label_field_names = self.p.attr_name + ["Average"]
        label_row_values = ["{:0.4f}".format(
            v) for v in self.results["label_acc"].tolist() + [np.mean(self.results["label_acc"])]]
        max_name_length = max([len(name) for name in label_field_names + label_row_values])
        label_field_names = [name.rjust(max_name_length, " ") for name in label_field_names]
        label_row_values = [name.rjust(max_name_length, " ") for name in label_row_values]
        label_table.add_column("attributes", label_field_names)
        label_table.add_column("mA", label_row_values)
        self.logger.info("Label-based evaluation: \n{}".format(label_table))
        instance_table = pt.PrettyTable()
        instance_field_names = ["Acc", "Prec", "Rec", "F1"]
        instance_row_values = ["{:0.4f}".format(v) for v in
                               [self.results["instance_acc"], self.results["instance_precision"],
                                self.results["instance_recall"], self.results["instance_F1"]]]
        max_name_length = max([len(name) for name in instance_field_names + instance_row_values])
        instance_field_names = [name.rjust(max_name_length, " ") for name in instance_field_names]
        instance_row_values = [name.rjust(max_name_length, " ") for name in instance_row_values]
        instance_table.field_names = instance_field_names
        instance_table.add_row(instance_row_values)
        self.logger.info("Instance-based evaluation: \n{}".format(instance_table))


if __name__ == "__main__":
    import json
    path = "logs/deepmar_rapv2_r50v1/rapv2_test_result.json"
    with open(path, "r") as f:
        results = json.load(f)

    class Config(object):
        attr_name = [
            "Femal", "AgeLess16", "Age17-30", "Age31-45", "Age46-60",
            "BodyFat", "BodyNormal", "BodyThin", "Customer", "Employee",
            "hs-BaldHead", "hs-LongHair", "hs-BlackHair", "hs-Hat", "hs-Glasses",
            "ub-Shirt", "ub-Sweater", "ub-Vest", "ub-TShirt", "ub-Cotton",
            "ub-Jacket", "ub-SuitUp", "ub-Tight", "ub-ShortSleeve", "ub-Others",
            "lb-LongTrousers", "lb-Skirt", "lb-ShortSkirt", "lb-Dress", "lb-Jeans",
            "lb-TightTrousers", "shoes-Leather", "shoes-Sports", "shoes-Boots",
            "shoes-Cloth", "shoes-Casual", "shoes-Other", "attachment-Backpack",
            "attachment-ShoulderBag", "attachment-HandBag", "attachment-Box",
            "attachment-PlasticBag", "attachment-PaperBag", "attachment-HandTrunk",
            "attachment-Other", "action-Calling", "action-Talking", "action-Gathering",
            "action-Holding", "action-Pushing", "action-Pulling",
            "action-CarryingByArm", "action-CarryingByHand", "action-Other"]

    pConfig = Config()

    metric = AttributeMetric(pConfig)
    metric.process(results)
    metric.summarize()
