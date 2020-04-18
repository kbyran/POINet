from poi.models.pose_simple_baseline.builder import Builder
from poi.models.common.backbone.resnet_v1 import ResNet50V1C5 as Backbone
from poi.models.pose_simple_baseline.builder import Neck as Neck
from poi.models.pose_simple_baseline.builder import Head as Head
from poi.ops.symbol.complicate import normalizer_factory
from poi.eval.pose.pose_metric import PoseMetric


def get_config(task):
    class General:
        log_frequency = 10
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 32 if task == "train" else 1
        fp16 = False
        loader_worker = 8
        loader_collector = 2
        profile = None

    class KvstoreParam:
        kvstore = "nccl"
        batch_image = General.batch_image
        # gpus = [0, 1, 2, 3, 4, 5, 6, 7]
        gpus = [0]
        fp16 = General.fp16

    class NormalizeParam:
        # normalizer = normalizer_factory(type="syncbn", ndev=8, wd_mult=1.0)
        normalizer = normalizer_factory(type="local")

    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer

    class NeckParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        num_joints = 17
        num_deconv = 3
        num_deconv_filter = [256, 256, 256]
        num_deconv_kernel = [4, 4, 4]
        conv_kernel = 1

    class HeadParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image
        num_joints = 17
        heatmap_height = 64
        heatmap_width = 48

    class DatasetParam:
        if task == "train":
            image_set = ("coco_keypoints_train2017",)
        elif task in ["val", "test"]:
            image_set = ("coco_keypoints_val2017",)
        elif task == "export":
            image_set = None

    backbone = Backbone(BackboneParam)
    neck = Neck(NeckParam)
    head = Head(HeadParam)
    builder = Builder()
    if task == "train":
        train_sym = builder.get_train_symbol(backbone, neck, head)
        test_sym = None
    elif task in ["val", "test"]:
        train_sym = None
        test_sym = builder.get_test_symbol(backbone, neck, head)
    elif task == "export":
        train_sym = None
        test_sym = builder.get_export_symbol(backbone, neck, head)

    class ModelParam:
        train_symbol = train_sym
        test_symbol = test_sym

        from_scratch = False
        random = True
        memonger = False
        memonger_until = "stage3_unit21_plus"
        process_weight = None
        merge_bn = True
        batch_end_callbacks = None
        checkpoint_freq = 10

        class pretrain:
            prefix = "pretrained/resnet50_v1"
            epoch = 0
            fixed_param = None
            excluded_param = None

    class OptimizeParam:
        class optimizer:
            type = "adam"
            lr = 0.000035 / 32 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            lr_mode = None
            wd = 0.00001
            clip_gradient = None

        class schedule:
            begin_epoch = 0
            end_epoch = 150
            lr_iter = [80 * 4800 * 32 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       120 * 4800 * 32 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]
            lr_factor = 0.1

        warmup = None

    class TestParam:
        result_name = ["rec_id", "image_id", "affine", "max_idx", "max_val"]
        val_name = ["image_id", "keypoints", "score", "category_id"]
        test_name = ["image_id", "keypoints", "score", "category_id"]

        def process_roidb(x):
            return x

        def process_output(x, y):
            import numpy as np
            for out, r in zip(x, y):
                max_idx = np.array(out["max_idx"])
                max_val = np.array(out["max_val"])
                affine = np.array(out["affine"])
                inv_affine = np.linalg.inv(affine)
                affine_pts = np.ones((len(max_idx), 3))
                affine_pts[:, :2] = np.array(max_idx) * 4.0
                pts = np.dot(inv_affine, affine_pts.T).T[:, :2]
                out["keypoints"] = np.concatenate((pts, max_val), axis=1).reshape((-1,))
                vis_idx = max_val > 0.2
                if np.sum(vis_idx) > 0:
                    kpt_score = np.mean(max_val[vis_idx])
                else:
                    kpt_score = 0.0
                rescore = kpt_score * r["score"] if "score" in r else kpt_score
                out["score"] = rescore
                out["category_id"] = 1
            return x, y

        class model:
            prefix = "logs/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

        class coco:
            annotation = "data/coco_keypoints/annotations/person_keypoints_val2017.json"
            oks_nms = False

        test_metric = PoseMetric(coco) if task in ["val", "test"] else None

    class IOParam:
        # data processing
        class ReadParam:
            items = ["image", "affine"]

        class NormParam:
            mean = tuple(i * 255 for i in (0.485, 0.456, 0.406))  # RGB order
            std = tuple(i * 255 for i in (0.229, 0.224, 0.225))

        class FlipParam:
            joint_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

        class RescaleParam:
            scaling_factor = 0.25
            height = 256
            width = 192

        class RotateParam:
            rotation_p = 0.6
            rotation_factor = 30
            height = 256
            width = 192

        class CropParam:
            bbox_expand = 1.25
            height = 256
            width = 192

        class AffineParam:
            num_joints = 17
            height = 256
            width = 192

        class TargetParam:
            num_joints = 17
            height = 256
            width = 192
            heatmap_height = 64
            heatmap_width = 48
            sigma = 2

        class RenameParam:
            mapping = dict(image="data")

        from poi.core.image import (
            ReadRecord, RandFlip3DImageBboxJoint, Rescale3DMatrix, Rotate3DMatrix,
            Crop3DMatrix, Affine3DImageJoint, Affine3DImage, GenGaussianTarget, Norm2DImage,
            ConvertImageFromHWCToCHW, RenameRecord
        )
        from poi.core.sampler import RegularSampler, RandomSampler

        if task == "train":
            sampler = RandomSampler()
            transform = [
                ReadRecord(ReadParam),
                RandFlip3DImageBboxJoint(FlipParam),
                Rescale3DMatrix(RescaleParam),
                Rotate3DMatrix(RotateParam),
                Crop3DMatrix(CropParam),
                Affine3DImageJoint(AffineParam),
                GenGaussianTarget(TargetParam),
                Norm2DImage(NormParam),
                ConvertImageFromHWCToCHW(),
                RenameRecord(RenameParam)
            ]
            data_name = ["data"]
            label_name = ["target", "target_weight"]
        elif task in ["val", "test"]:
            sampler = RegularSampler()
            transform = [
                ReadRecord(ReadParam),
                Crop3DMatrix(CropParam),
                Affine3DImage(AffineParam),
                Norm2DImage(NormParam),
                ConvertImageFromHWCToCHW(),
                RenameRecord(RenameParam)
            ]
            data_name = ["data", "im_id", "rec_id", "affine"]
            label_name = []
        elif task == "export":
            sampler = None
            transform = [
                ReadRecord(ReadParam),
                Crop3DMatrix(CropParam),
                Affine3DImage(AffineParam),
                Norm2DImage(NormParam),
                ConvertImageFromHWCToCHW(),
                RenameRecord(RenameParam)
            ]
            data_name = ["data"]
            label_name = []

    from poi.core.metric import HeatmapAcc, HeatmapLoss

    class MetricParam:
        acc_metric = HeatmapAcc(
            "Acc",
            ["heatmap_blockgrad_output"],
            ["target", "target_weight"]
        )
        l2_metric = HeatmapLoss(
            "L2Loss",
            ["l2_loss_output"],
            ["target_weight"]
        )

        metric_list = [acc_metric, l2_metric]

    return General, KvstoreParam, DatasetParam, ModelParam, OptimizeParam, TestParam, \
        IOParam, MetricParam
