from poi.models.densepose_baseline.builder import Builder
from poi.models.common.backbone.resnet_v1 import ResNet50V1C5 as Backbone
from poi.models.densepose_baseline.builder import Neck as Neck
from poi.models.densepose_baseline.builder import Head as Head
from poi.ops.symbol.complicate import normalizer_factory
from poi.eval.densepose.densepose_metric import DensePoseMetric


def get_config(task):
    class General:
        log_frequency = 10
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 32 if task == "train" else 8
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
        num_masks = 14
        num_patches = 24
        num_deconv = 3
        num_deconv_filter = [256, 256, 256]
        num_deconv_kernel = [4, 4, 4]
        num_conv = 1
        num_conv_filter = [256]
        num_conv_kernel = [3]
        conv_kernel = 1

    class HeadParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        seg_grad_scale = 1.0
        index_grad_scale = 1.0
        u_grad_scale = 10.0
        v_grad_scale = 10.0
        heatmap_height = 64
        heatmap_width = 64

    class DatasetParam:
        if task == "train":
            image_set = ("coco_densepose_train",)
        elif task in ["val", "test"]:
            image_set = ("coco_densepose_minival",)
            # image_set = ("coco_densepose_minival", "coco_densepose_valminusminival")
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
            lr = 0.00035 / 32 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            lr_mode = None
            # momentum = 0.9
            wd = 0.00001
            clip_gradient = None

        class schedule:
            begin_epoch = 0
            end_epoch = 150
            lr_iter = [80 * 1220 * 32 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       120 * 1220 * 32 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]
            lr_factor = 0.1

        warmup = None

    class TestParam:
        result_name = ["rec_id", "image_id", "affine", "ann_index_lowres", "index_uv_lowres",
                       "u_lowres", "v_lowres"]
        val_name = ["image_id", "uv", "score", "category_id", "bbox"]
        test_name = ["image_id", "uv", "score", "category_id", "bbox"]

        def process_roidb(x):
            return x

        def process_output(x, y):
            import numpy as np
            import mxnet as mx
            from poi.ops.fuse.index_ops import resize_uv
            ctx = mx.gpu(KvstoreParam.gpus[0])
            for out, r in zip(x, y):
                ann_index_lowres = mx.nd.array(out["ann_index_lowres"], ctx=ctx)
                index_uv_lowres = mx.nd.array(out["index_uv_lowres"], ctx=ctx)
                u_lowres = mx.nd.array(out["u_lowres"], ctx=ctx)
                v_lowres = mx.nd.array(out["v_lowres"], ctx=ctx)
                gt_bbox = r["gt_bbox"]
                height = max(np.ceil(gt_bbox[3] - gt_bbox[1]).astype("int"), 1)
                width = max(np.ceil(gt_bbox[2] - gt_bbox[0]).astype("int"), 1)
                uv = resize_uv(
                    mx.nd, ann_index_lowres, index_uv_lowres, u_lowres, v_lowres, height, width
                )
                out["uv"] = uv.asnumpy().tolist()
                b = r["gt_bbox"].tolist()
                out["bbox"] = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                out["category_id"] = 1
                out["score"] = 1.0
                del out["ann_index_lowres"]
                del out["index_uv_lowres"]
                del out["u_lowres"]
                del out["v_lowres"]
            return x, y

        class model:
            prefix = "logs/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

        class coco:
            annotation = "data/coco_densepose/annotations/densepose_coco_2014_minival.json"

        test_metric = DensePoseMetric(coco) if task in ["val", "test"] else None

    class IOParam:
        # data processing
        class ReadParam:
            items = ["image", "affine"]

        class NormParam:
            mean = tuple(i * 255 for i in (0.485, 0.456, 0.406))  # RGB order
            std = tuple(i * 255 for i in (0.229, 0.224, 0.225))

        class FlipParam:
            mask_symmetry = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 13]
            index_symmetry = [1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12,
                              11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]
            uv_symmetry_file = "data/coco_densepose/UV_data/UV_symmetry_transforms.mat"

        class RescaleParam:
            scaling_factor = 0.25

        class RotateParam:
            rotation_p = 0.6
            rotation_factor = 30

        class CropParam:
            bbox_expand = 1.0
            height = 256
            width = 256

        class AffineParam:
            num_joints = 17
            height = 256
            width = 256

        class DPTargetParam:
            height = 256
            width = 256
            heatmap_height = 64
            heatmap_width = 64

        class RenameParam:
            mapping = dict(image="data")

        from poi.core.image import (
            ReadRecord, RandFlip3DImageBboxDensePose, Rescale3DMatrix, Rotate3DMatrix,
            Crop3DMatrix, GenDensePoseTarget, Affine3DImage, Norm2DImage,
            ConvertImageFromHWCToCHW, RenameRecord
        )
        from poi.core.sampler import RegularSampler, RandomSampler

        if task == "train":
            sampler = RandomSampler()
            transform = [
                ReadRecord(ReadParam),
                RandFlip3DImageBboxDensePose(FlipParam),
                Rescale3DMatrix(RescaleParam),
                Rotate3DMatrix(RotateParam),
                Crop3DMatrix(CropParam),
                GenDensePoseTarget(DPTargetParam),
                Affine3DImage(AffineParam),
                Norm2DImage(NormParam),
                ConvertImageFromHWCToCHW(),
                RenameRecord(RenameParam)
            ]
            data_name = ["data"]
            label_name = ["dp_masks", "dp_I", "dp_U", "dp_V", "dp_x", "dp_y"]
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

    from poi.core.metric import AccWithIgnore, CeWithIgnore, MakeLoss

    class MetricParam:
        seg_acc_metric = AccWithIgnore(
            "SegAcc",
            ["seg_loss_output"],
            ["dp_masks"]
        )
        seg_ce_metric = CeWithIgnore(
            "SegCE",
            ["seg_loss_output"],
            ["dp_masks"]
        )
        index_uv_acc_metric = AccWithIgnore(
            "IndexUVAcc",
            ["index_uv_loss_output"],
            ["dp_I"]
        )
        index_uv_ce_metric = CeWithIgnore(
            "IndexUVCE",
            ["index_uv_loss_output"],
            ["dp_I"]
        )
        u_metric = MakeLoss(
            "ULoss",
            ["u_loss_output"],
            ["dp_I"]
        )
        v_metric = MakeLoss(
            "VLoss",
            ["v_loss_output"],
            ["dp_I"]
        )

        metric_list = [seg_acc_metric, seg_ce_metric, index_uv_acc_metric, index_uv_ce_metric,
                       u_metric, v_metric]

    return General, KvstoreParam, DatasetParam, ModelParam, OptimizeParam, TestParam, \
        IOParam, MetricParam
