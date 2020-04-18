from poi.models.attr_deepmar.builder import Builder
from poi.models.common.backbone.resnet_v1 import ResNet50V1C5 as Backbone
from poi.models.attr_deepmar.builder import BasicNeck as Neck
from poi.models.attr_deepmar.builder import MultiHead as Head
from poi.ops.symbol.complicate import normalizer_factory
from poi.eval.attr.attribute_metric import AttributeMetric


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

    class HeadParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer

        class xent_loss:
            is_enable = True
            feature = "pool_branch"
            num_classes = 54
            dropout = True
            dropout_p = 0.5
            pos_ratio = [
                0.312185, 0.009145, 0.402849, 0.547226, 0.035304, 0.137312, 0.777636, 0.073513,
                0.938281, 0.053575, 0.005809, 0.193025, 0.931609, 0.015798, 0.066782, 0.216241,
                0.077987, 0.047177, 0.231116, 0.115961, 0.283886, 0.030673, 0.041996, 0.097043,
                0.026866, 0.576898, 0.029790, 0.020370, 0.027612, 0.265263, 0.117766, 0.264969,
                0.201209, 0.084640, 0.014169, 0.228055, 0.083129, 0.021116, 0.070020, 0.026728,
                0.037502, 0.027749, 0.010087, 0.024413, 0.295955, 0.033872, 0.026885, 0.048668,
                0.022117, 0.010362, 0.017976, 0.024903, 0.132739, 0.010224
            ]
            grad_scale = 1.0

    class DatasetParam:
        if task == "train":
            image_set = ("rapv2_train",)
        elif task == "val":
            image_set = ("rapv2_val",)
        elif task == "test":
            image_set = ("rapv2_test",)
        elif task == "export":
            image_set = None
        else:
            raise ValueError("Stage {} is not implemented.".format(task))

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
            lr = 0.00035 / 64 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            lr_mode = None
            # momentum = 0.9
            wd = 0.0005
            clip_gradient = None

        class schedule:
            begin_epoch = 0
            end_epoch = 150
            lr_iter = [10 * 800 * 64 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       80 * 800 * 64 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       120 * 800 * 64 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]
            lr_factor = 0.1

        class warmup:
            type = "gradual"
            lr = 0.00035 / 64 * len(KvstoreParam.gpus) * KvstoreParam.batch_image / 100
            iter = 6 * 800 * 64 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)

    class TestParam:
        result_name = ["rec_id", "im_id", "logits"]
        val_name = ["split", "labels", "logits"]
        test_name = ["split", "labels", "logits"]

        class metric:
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

        test_metric = AttributeMetric(metric)

        def process_roidb(x):
            return x

        def process_output(x, y):
            for x_ in x:
                logits = []
                for logit in x_["logits"]:
                    logit_ = 1 if logit >= 0.5 else 0
                    logits.append(logit_)
                x_["logits"] = logits

            return x, y

        class model:
            prefix = "logs/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

    class IOParam:
        # data processing
        class SamplerParam:
            batch_image = General.batch_image

        class ReadParam:
            items = ["image", "labels"] if task == "train" else ["image"]

        class NormParam:
            mean = tuple(i * 255 for i in (0.485, 0.456, 0.406))  # RGB order
            std = tuple(i * 255 for i in (0.229, 0.224, 0.225))

        class PadParam:
            height = 244
            width = 244

        class RandCropParam:
            height = 224
            width = 224

        class ErasingParam:
            prob = 0.5
            min_ratio = 0.02
            max_ratio = 0.4
            aspect_ratio = 0.3
            mean = tuple(i * 255 for i in (0.485, 0.456, 0.406))

        class ResizeParam:
            height = 224
            width = 224

        class RenameParam:
            mapping = dict(image="data")

        from poi.core.image import (
            ReadRecord, Resize2DImage, RandFlip2DImage, PadHW2DImage, RandCropHW2DImage,
            RandErasing2DImage, Norm2DImage, ConvertImageFromHWCToCHW, RenameRecord
        )
        from poi.core.sampler import RegularSampler, RandomSampler, TripletSampler

        if task == "train":
            sampler = RandomSampler()
            transform = [
                ReadRecord(ReadParam),
                Resize2DImage(ResizeParam),
                RandFlip2DImage(),
                # PadHW2DImage(PadParam),
                # RandCropHW2DImage(RandCropParam),
                # RandErasing2DImage(ErasingParam),
                Norm2DImage(NormParam),
                ConvertImageFromHWCToCHW(),
                RenameRecord(RenameParam)
            ]
            data_name = ["data"]
            label_name = ["labels"]
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

    from poi.core.metric import MultiAccWithIgnore, MakeLoss

    class MetricParam:
        acc_metric = MultiAccWithIgnore(
            "Acc",
            ["fc1_output"],
            ["labels"]
        )
        ce_metric = MakeLoss(
            "CE",
            ["bce_loss_output"],
            ["labels"]
        )

        metric_list = [acc_metric, ce_metric]

    return General, KvstoreParam, DatasetParam, ModelParam, OptimizeParam, TestParam, \
        IOParam, MetricParam
