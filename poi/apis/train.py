import importlib
import os
import pprint
import mxnet as mx
import numpy as np
import pickle as pkl
from functools import reduce
from poi.utils.logging import set_logger
from poi.core.io import DataLoader
from poi.core.model import load_params
from poi.core.module import Module
from poi.core.callback import Speedometer, do_checkpoint
from poi.core.lr_scheduler import WarmupMultiFactorScheduler, LRSequential, AdvancedLRScheduler


def train_net(config, **kwargs):
    cfg = importlib.import_module(config.replace(".py", "").replace("/", "."))
    pGen, pKv, pData, pModel, pOpt, pTest, pIO, pMetric = cfg.get_config(stage="train")
    sym = pModel.train_symbol
    batch_size_per_gpu = pKv.batch_image
    batch_size = pKv.batch_image * len(pKv.gpus)

    # kvstore
    kvstore_type = "dist_sync" if os.environ.get("DMLC_ROLE") == "worker" else pKv.kvstore
    kv = mx.kvstore.create(kvstore_type)
    rank = kv.rank

    # logging
    log_path = os.path.join("logs", pGen.name, "log.txt")
    logger = set_logger(log_path, rank)

    # dataset
    image_sets = pData.image_set
    dbs = [pkl.load(open("data/cache/{}.db".format(i), "rb"), encoding="latin1")
           for i in image_sets]
    db = reduce(lambda x, y: x + y, dbs)
    train_data = DataLoader(
        db,
        sampler=pIO.sampler,
        transform=pIO.transform,
        data_name=pIO.data_name,
        label_name=pIO.label_name,
        batch_size=batch_size,
        use_mp=False,
        num_worker=pGen.loader_worker or 12,
        num_collector=pGen.loader_collector or 1,
        worker_queue_depth=2,
        collector_queue_depth=2
    )

    # infer shape
    in_shape = dict(train_data.provide_data + train_data.provide_label)
    in_shape = dict([(key, (batch_size_per_gpu,) + in_shape[key][1:]) for key in in_shape])
    _, inter_shape, _ = sym.get_internals().infer_shape(**in_shape)
    inter_shape_dict = zip(sym.get_internals().list_outputs(), inter_shape)
    param_shape_dict = [i for i in inter_shape_dict if not i[0].endswith("output")]
    inter_out_shape_dict = [i for i in inter_shape_dict if i[0].endswith("output")]
    _, out_shape, _ = sym.infer_shape(**in_shape)
    out_shape_dict = list(zip(sym.list_outputs(), out_shape))
    logger.info("params shape")
    logger.info(pprint.pformat(param_shape_dict))
    logger.info("internal output shape")
    logger.info(pprint.pformat(inter_out_shape_dict))
    logger.info("output shape")
    logger.info(pprint.pformat(out_shape_dict))

    # memonger
    if pModel.memonger:
        from poi.utils.memonger_v2 import search_plan_to_layer
        last_block = pModel.memoger_until or ""
        logger.info("do memonger up to {}".format(last_block))
        type_dict = {k: np.float32 for k in in_shape}
        sym = search_plan_to_layer(sym, last_block, 1000, type_dict=type_dict, **in_shape)

    # params
    model_prefix = os.path.join("logs", pGen.name, "checkpoint")
    begin_epoch = pOpt.schedule.begin_epoch
    pretrain_prefix = pModel.pretrain.prefix
    pretrain_epoch = pModel.pretrain.epoch
    if begin_epoch != 0:
        arg_params, aux_params = load_params(model_prefix, begin_epoch)
    elif pModel.from_scratch:
        arg_params, aux_params = dict(), dict()
    else:
        arg_params, aux_params = load_params(pretrain_prefix, pretrain_epoch)
    if pModel.process_weight is not None:
        pModel.process_weight(sym, arg_params, aux_params)
    if pModel.merge_bn:
        from poi.utils.graph_optimize import merge_bn
        sym, arg_params, aux_params = merge_bn(sym, arg_params, aux_params)
    if pModel.random:
        import time
        mx.random.seed(int(time.time()))
        np.random.seed(int(time.time()))
    init = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)
    init.set_verbosity(verbose=True)

    # module
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    ctx = [mx.gpu(int(i)) for i in pKv.gpus]
    fixed_param = pModel.pretrain.fixed_param
    excluded_param = pModel.pretrain.excluded_param
    mod = Module(
        sym, data_names=data_names, label_names=label_names, logger=logger, context=ctx,
        fixed_param=fixed_param, excluded_param=excluded_param
    )

    # metric
    eval_metrics = mx.metric.CompositeEvalMetric(pMetric.metric_list)
    # callback
    batch_end_callback = [Speedometer(train_data.batch_size, frequent=pGen.log_frequency)]
    batch_end_callback += pModel.batch_end_callbacks or []
    epoch_end_callback = do_checkpoint(model_prefix, frequent=pModel.checkpoint_freq or 1)
    # lr_schedule
    lr_mode = pOpt.optimizer.lr_mode or 'step'
    base_lr = pOpt.optimizer.lr * kv.num_workers
    lr_factor = pOpt.schedule.lr_factor or 0.1
    iter_per_epoch = len(train_data) // batch_size
    end_epoch = pOpt.schedule.end_epoch
    total_iter = iter_per_epoch * (end_epoch - begin_epoch)
    lr_iter = pOpt.schedule.lr_iter
    lr_iter = [total_iter + it if it < 0 else it for it in lr_iter]
    lr_iter = [it // kv.num_workers for it in lr_iter]
    lr_iter = [it - iter_per_epoch * begin_epoch for it in lr_iter]
    lr_iter_discount = [it for it in lr_iter if it > 0]
    current_lr = base_lr * (lr_factor ** (len(lr_iter) - len(lr_iter_discount)))
    logger.info("total iter {}".format(total_iter))
    logger.info("lr {}, lr_iters {}".format(current_lr, lr_iter_discount))
    logger.info("lr mode: {}".format(lr_mode))

    if pOpt.warmup and pOpt.schedule.begin_epoch == 0:
        logger.info(
            "warmup lr {}, warmup step {}".format(pOpt.warmup.lr, pOpt.warmup.iter))
        if lr_mode == "step":
            lr_scheduler = WarmupMultiFactorScheduler(
                step=lr_iter_discount,
                factor=lr_factor,
                warmup=True,
                warmup_type=pOpt.warmup.type,
                warmup_lr=pOpt.warmup.lr,
                warmup_step=pOpt.warmup.iter
            )
        elif lr_mode == "cosine":
            warmup_lr_scheduler = AdvancedLRScheduler(
                mode="linear",
                base_lr=pOpt.warmup.lr,
                target_lr=base_lr,
                niters=pOpt.warmup.iter
            )
            cosine_lr_scheduler = AdvancedLRScheduler(
                mode="cosine",
                base_lr=base_lr,
                target_lr=0,
                niters=(iter_per_epoch * (end_epoch - begin_epoch)
                        ) // kv.num_workers - pOpt.warmup.iter
            )
            lr_scheduler = LRSequential([warmup_lr_scheduler, cosine_lr_scheduler])
        else:
            raise NotImplementedError
    else:
        if lr_mode == "step":
            lr_scheduler = WarmupMultiFactorScheduler(step=lr_iter_discount, factor=lr_factor)
        elif lr_mode == "cosine":
            lr_scheduler = AdvancedLRScheduler(
                mode="cosine",
                base_lr=base_lr,
                target_lr=0,
                niters=iter_per_epoch * (end_epoch - begin_epoch) // kv.num_workers
            )
        else:
            lr_scheduler = None

    # optimizer
    if pOpt.optimizer.type == "sgd":
        optimizer_params = dict(
            momentum=pOpt.optimizer.momentum,
            wd=pOpt.optimizer.wd,
            learning_rate=current_lr,
            lr_scheduler=lr_scheduler,
            rescale_grad=1.0 / (len(ctx) * kv.num_workers),
            clip_gradient=pOpt.optimizer.clip_gradient
        )
    elif pOpt.optimizer.type == "adam":
        optimizer_params = dict(
            wd=pOpt.optimizer.wd,
            learning_rate=current_lr,
            lr_scheduler=lr_scheduler,
            rescale_grad=1.0,
            clip_gradient=pOpt.optimizer.clip_gradient
        )
    logger.info("optimizer: {}".format(pOpt.optimizer.type))
    logger.info("opt wd: {}".format(pOpt.optimizer.wd))
    logger.info("opt clip_grad: {}".format(pOpt.optimizer.clip_gradient))
    if pKv.fp16:
        optimizer_params["multi_precision"] = True
        optimizer_params["rescale_grad"] /= 128.0

    # profile
    profile = pGen.profile or False
    if profile:
        filename = os.path.join("logs", pGen.name, "profile.json")
        mx.profiler.set_config(profile_all=True, filename=filename)

    # start training
    mod.fit(
        train_data=train_data,
        eval_metric=eval_metrics,
        epoch_end_callback=epoch_end_callback,
        batch_end_callback=batch_end_callback,
        kvstore=kv,
        optimizer=pOpt.optimizer.type,
        optimizer_params=optimizer_params,
        initializer=init,
        allow_missing=True,
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=begin_epoch,
        num_epoch=end_epoch,
        profile=profile
    )

    logger.info("Training has done.")
    time.sleep(10)
    logger.info("Exiting")
