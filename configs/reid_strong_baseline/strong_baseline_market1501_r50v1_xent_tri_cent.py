from poi.models.reid_strong_baseline.builder import Builder
from poi.models.common.backbone.resnet_v1 import ResNet50V1C5X2 as Backbone
from poi.models.reid_strong_baseline.builder import BNNeck as Neck
from poi.models.reid_strong_baseline.builder import MultiHead as Head
from poi.ops.symbol.complicate import normalizer_factory
from poi.eval.reid.reid_metric import ReIDMetric


def get_config(task):
    class General:
        log_frequency = 10
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 64 if task == "train" else 8
        fp16 = False
        loader_worker = 8
        loader_collector = 1
        profile = None

    class KvstoreParam:
        kvstore = "local"
        batch_image = General.batch_image
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

    class HeadParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer

        class xent_loss:
            is_enable = True
            feature = "bn_branch"
            num_classes = 751
            grad_scale = 1.0
            smooth_alpha = 0.1

        class triplet_loss:
            is_enable = True
            feature = "pool_branch"
            margin = 0.3
            grad_scale = 1.0

        class center_loss:
            is_enable = True
            feature = "pool_branch"
            grad_scale = 1.0

        class get_feature:
            feature = "bn_branch"

    class DatasetParam:
        if task == "train":
            image_set = ("market1501_train",)
        elif task == "val":
            image_set = ("market1501_gallery", "market1501_query")
        elif task == "test":
            image_set = ("market1501_gallery", "market1501_query")
        elif task == "export":
            image_set = None
        else:
            raise ValueError("Task {} is not implemented.".format(task))

    backbone = Backbone(BackboneParam)
    neck = Neck(NeckParam)
    head = Head(HeadParam)
    builder = Builder()
    if task == "train":
        train_sym = builder.get_train_symbol(backbone, neck, head)
        test_sym = None
    elif task in ["val", "test"]:
        train_sym = None
        test_sym = builder.get_test_symbol(backbone, neck, head, fliptest=False)
    elif task == "export":
        train_sym = None
        test_sym = builder.get_export_symbol(backbone, neck, head, fliptest=False)

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
            lr = 0.00035 / 64 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            lr_mode = None
            # momentum = 0.9
            wd = 0.0005
            clip_gradient = None

        class schedule:
            begin_epoch = 0
            end_epoch = 150
            lr_iter = [30 * 200 * 64 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       100 * 200 * 64 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       130 * 200 * 64 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]
            lr_factor = 0.1

        class warmup:
            type = "gradual"
            lr = 0.00035 / 64 * len(KvstoreParam.gpus) * KvstoreParam.batch_image / 100
            iter = 10 * 200 * 64 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)

    class TestParam:
        result_name = ["rec_id", "im_id", "feature"]
        val_name = ["split", "pid", "cid", "feature"]
        test_name = ["split", "image_url", "feature"]

        class metric:
            gpus = KvstoreParam.gpus
            dist_type = "cosine"  # "euclidean", "cosine" or "reranking"
            max_rank = 50

        test_metric = ReIDMetric(metric)

        def process_roidb(x):
            return x

        def process_output(x, y):
            return x, y

        class model:
            prefix = "logs/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

    class IOParam:
        # data processing
        class SamplerParam:
            batch_image = General.batch_image
            num_instance = 4

        class ReadParam:
            items = ["image", "label"] if task == "train" else ["image"]

        class NormParam:
            mean = tuple(i * 255 for i in (0.485, 0.456, 0.406))  # RGB order
            std = tuple(i * 255 for i in (0.229, 0.224, 0.225))

        class PadParam:
            height = 276
            width = 148

        class RandCropParam:
            height = 256
            width = 128

        class ErasingParam:
            prob = 0.5
            min_ratio = 0.02
            max_ratio = 0.4
            aspect_ratio = 0.3
            mean = tuple(i * 255 for i in (0.485, 0.456, 0.406))

        class ResizeParam:
            height = 256
            width = 128

        class RenameParam:
            mapping = dict(image="data")

        from poi.core.image import (
            ReadRecord, Resize2DImage, RandFlip2DImage, PadHW2DImage, RandCropHW2DImage,
            RandErasing2DImage, Norm2DImage, ConvertImageFromHWCToCHW, RenameRecord
        )
        from poi.core.sampler import RegularSampler, RandomSampler, TripletSampler

        if task == "train":
            sampler = TripletSampler(SamplerParam)
            transform = [
                ReadRecord(ReadParam),
                Resize2DImage(ResizeParam),
                RandFlip2DImage(),
                PadHW2DImage(PadParam),
                RandCropHW2DImage(RandCropParam),
                RandErasing2DImage(ErasingParam),
                Norm2DImage(NormParam),
                ConvertImageFromHWCToCHW(),
                RenameRecord(RenameParam)
            ]
            data_name = ["data"]
            label_name = ["label"]
        elif task in ["val", "test"]:
            sampler = RegularSampler()
            transform = [
                ReadRecord(ReadParam),
                Resize2DImage(ResizeParam),
                Norm2DImage(NormParam),
                ConvertImageFromHWCToCHW(),
                RenameRecord(RenameParam)
            ]
            data_name = ["data", "im_id", "rec_id"]
            label_name = []
        elif task == "export":
            sampler = None
            transform = [
                ReadRecord(ReadParam),
                Resize2DImage(ResizeParam),
                Norm2DImage(NormParam),
                ConvertImageFromHWCToCHW(),
                RenameRecord(RenameParam)
            ]
            data_name = ["data"]
            label_name = []

    from poi.core.metric import AccWithIgnore, CeWithIgnore, MakeLoss

    class MetricParam:
        acc_metric = AccWithIgnore(
            "Acc",
            ["softmax_output"],
            ["label"]
        )
        ce_metric = CeWithIgnore(
            "CE",
            ["softmax_output"],
            ["label"]
        )
        triplet_metric = MakeLoss(
            "TripletLoss",
            ["triplet_loss_output"],
            ["label"]
        )
        center_metric = MakeLoss(
            "CenterLoss",
            ["center_loss_output"],
            ["label"]
        )

        metric_list = []
        if HeadParam.xent_loss.is_enable:
            metric_list += [acc_metric, ce_metric]
        if HeadParam.triplet_loss.is_enable:
            metric_list += [triplet_metric]
        if HeadParam.center_loss.is_enable:
            metric_list += [center_metric]

    return General, KvstoreParam, DatasetParam, ModelParam, OptimizeParam, TestParam, \
        IOParam, MetricParam
