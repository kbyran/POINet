import mxnet as mx
import numpy as np


class EvalMetricWithSummary(mx.metric.EvalMetric):
    def __init__(self, name, output_names=None, label_names=None, summary=None, **kwargs):
        super().__init__(name, output_names=output_names, label_names=label_names, **kwargs)
        self.summary = summary
        self.global_step = 0

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            self.global_step += 1
            if self.summary:
                self.summary.add_scalar(tag=self.name, value=self.sum_metric / self.num_inst,
                                        global_step=self.global_step)
            return (self.name, self.sum_metric / self.num_inst)


class LossWithIgnore(EvalMetricWithSummary):
    def __init__(self, name, output_names, label_names, ignore_label=-1, **kwargs):
        super().__init__(name, output_names, label_names, **kwargs)
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        raise NotImplementedError


class AccWithIgnore(LossWithIgnore):
    def __init__(self, name, output_names, label_names, ignore_label=-1, **kwargs):
        super().__init__(name, output_names, label_names, ignore_label, **kwargs)

    def update(self, labels, preds):
        if len(preds) == 1 and len(labels) == 1:
            pred = preds[0]
            label = labels[0]
        elif len(preds) == 2:
            pred = preds[0]
            label = preds[1]
        else:
            raise Exception("unknown loss output: len(preds): {}, len(labels): {}".format(
                len(preds), len(labels)))

        pred_label = mx.ndarray.argmax_channel(pred).astype('int32').asnumpy().reshape(-1)
        label = label.astype('int32').asnumpy().reshape(-1)

        keep_inds = np.where(label != self.ignore_label)[0]
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label == label)
        self.num_inst += len(pred_label)


class MultiAccWithIgnore(LossWithIgnore):
    def __init__(self, name, output_names, label_names, ignore_label=-1, threshold=0, **kwargs):
        super().__init__(name, output_names, label_names, ignore_label, **kwargs)
        self.thr = threshold

    def update(self, labels, preds):
        if len(preds) == 1 and len(labels) == 1:
            pred = preds[0]
            label = labels[0]
        elif len(preds) == 2:
            pred = preds[0]
            label = preds[1]
        else:
            raise Exception("unknown loss output: len(preds): {}, len(labels): {}".format(
                len(preds), len(labels)))

        pred_label = (pred > self.thr).astype('int32').asnumpy().reshape(-1)
        label = label.astype('int32').asnumpy().reshape(-1)

        keep_inds = np.where(label != self.ignore_label)[0]
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label == label)
        self.num_inst += len(pred_label)


class CeWithIgnore(LossWithIgnore):
    def __init__(self, name, output_names, label_names, ignore_label=-1, **kwargs):
        super().__init__(name, output_names, label_names, ignore_label, **kwargs)

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        label = label.astype('int32').asnumpy().reshape(-1)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))  # -1 x c

        keep_inds = np.where(label != self.ignore_label)[0]
        label = label[keep_inds]
        prob = pred[keep_inds, label]

        prob += 1e-6
        ce_loss = -1 * np.log(prob)
        ce_loss = np.sum(ce_loss)
        self.sum_metric += ce_loss
        self.num_inst += label.shape[0]


class MakeLoss(LossWithIgnore):
    def __init__(self, name, output_names, label_names, ignore_label=-1, **kwargs):
        super().__init__(name, output_names, label_names, ignore_label, **kwargs)

    def update(self, labels, preds):
        pred = preds[0].asnumpy()
        label = labels[0].asnumpy()

        label = label.astype('int32').reshape(-1)
        num_inst = len(np.where(label != self.ignore_label)[0])

        self.sum_metric += np.sum(pred)
        self.num_inst += num_inst


class HeatmapAcc(LossWithIgnore):
    def __init__(self, name, output_names, label_names, ignore_label=-1, threshold=0.5, **kwargs):
        super().__init__(name, output_names, label_names, ignore_label, **kwargs)
        self.thr = threshold

    def update(self, labels, preds):
        from poi.ops.fuse.index_ops import heatmap2points
        heatmap = preds[0]
        ctx = heatmap.context
        target = labels[0].as_in_context(ctx)
        target_weight = labels[1].reshape((0, -1)).as_in_context(ctx)

        _, _, h, w = heatmap.shape

        points, _ = heatmap2points(mx.nd, heatmap, h, w)
        labels, _ = heatmap2points(mx.nd, target, h, w)
        norm = mx.nd.ones_like(points, ctx=ctx) * mx.nd.array([h, w], ctx=ctx) / 10.0
        dist = mx.nd.norm((points - labels) / norm, axis=2)
        pos = (dist < self.thr) * target_weight
        num_target = mx.nd.sum(target_weight, axis=1)
        num_pos = mx.nd.sum(pos, axis=1)
        pos_metric = mx.nd.where(
            num_target > 0, num_pos / num_target, mx.nd.zeros_like(num_target, ctx=ctx))
        num_inst = mx.nd.sum(num_target > 0).asscalar()
        self.sum_metric += mx.nd.sum(pos_metric).asscalar()
        self.num_inst += num_inst


class HeatmapLoss(LossWithIgnore):
    def __init__(self, name, output_names, label_names, ignore_label=-1, **kwargs):
        super().__init__(name, output_names, label_names, ignore_label, **kwargs)

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        num_inst = label.shape[0]

        self.sum_metric += np.sum(pred).asscalar()
        self.num_inst += num_inst


class OutputInput(EvalMetricWithSummary):
    def __init__(self, name, output_names, label_names, **kwargs):
        super().__init__(name, output_names, label_names, **kwargs)

    def update(self, labels, preds):
        self.sum_metric += 1
        self.num_inst += 1
