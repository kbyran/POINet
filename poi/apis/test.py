import os
import math
import pprint
import importlib
import pickle as pkl
from functools import reduce
from queue import Queue
from threading import Thread
import numpy as np
import mxnet as mx

from poi.utils.logging import set_logger
from poi.core.io import DataLoader
from poi.core.model import load_params
from poi.core.module import Module


def test_net(config, task="val", epoch=None, **kwargs):
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    cfg = importlib.import_module(config.replace(".py", "").replace("/", "."))
    pGen, pKv, pData, pModel, pOpt, pTest, pIO, pMetric = cfg.get_config(task=task)
    sym = pModel.test_symbol
    image_sets = pData.image_set

    # logging
    log_path = os.path.join("logs", pGen.name, "log_test.txt")
    logger = set_logger(log_path)

    # dataset
    dbs = [pkl.load(open("data/cache/{}.db".format(i), "rb"), encoding="latin1")
           for i in image_sets]
    db_all = reduce(lambda x, y: x + y, dbs)

    data_queue = Queue(100)
    result_queue = Queue()
    execs = []
    workers = []
    results = []
    split_size = 1000

    for index_split in range(int(math.ceil(len(db_all) / split_size))):
        logger.info(
            "evaluating [%d, %d)" % (index_split * split_size, (index_split + 1) * split_size))
        db = db_all[index_split * split_size:(index_split + 1) * split_size]
        db = pTest.process_roidb(db)

        num_pad = math.ceil(len(db) / pGen.batch_image) * pGen.batch_image - len(db)
        for i in range(num_pad):
            copy_i = db[-1].copy()
            copy_i["pad"] = True
            db.append(copy_i)

        for i, x in enumerate(db):
            x["rec_id"] = np.array(i, dtype=np.float32)
            x["im_id"] = np.array(x["im_id"], dtype=np.float32)

        data = DataLoader(
            db,
            sampler=pIO.sampler,
            transform=pIO.transform,
            data_name=pIO.data_name,
            label_name=pIO.label_name,
            batch_size=pGen.batch_image,
            use_mp=False,
            num_worker=4,
            num_collector=2,
            worker_queue_depth=2,
            collector_queue_depth=2,
            kv=None)

        logger.info("total number of images: {}".format(data.total_record))

        data_names = [k[0] for k in data.provide_data]

        if index_split == 0:
            arg_params, aux_params = load_params(
                pTest.model.prefix, epoch or pTest.model.epoch)
            if pModel.process_weight is not None:
                pModel.process_weight(sym, arg_params, aux_params)
            # merge batch normalization to speedup test
            if pModel.merge_bn:
                from poi.utils.graph_optimize import merge_bn
                sym, arg_params, aux_params = merge_bn(sym, arg_params, aux_params)
                sym.save(pTest.model.prefix + "_test.json")

            # infer shape
            in_shape = dict(data.provide_data + data.provide_label)
            in_shape = dict([(key, (pGen.batch_image,) + in_shape[key][1:]) for key in in_shape])
            _, inter_shape, _ = sym.get_internals().infer_shape(**in_shape)
            inter_shape_dict = list(zip(sym.get_internals().list_outputs(), inter_shape))
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

            for i in pKv.gpus:
                ctx = mx.gpu(i)
                mod = Module(sym, data_names=data_names, context=ctx)
                mod.bind(data_shapes=data.provide_data, for_training=False)
                mod.set_params(arg_params, aux_params, allow_extra=False)
                execs.append(mod)

        all_outputs = []

        if index_split == 0:
            def eval_worker(exe, data_queue, result_queue):
                while True:
                    batch = data_queue.get()
                    exe.forward(batch, is_train=False)
                    out = [x.asnumpy() for x in exe.get_outputs()]
                    result_queue.put(out)
            for exe in execs:
                workers.append(Thread(target=eval_worker, args=(exe, data_queue, result_queue)))
            for w in workers:
                w.daemon = True
                w.start()

        import time
        t1_s = time.time()

        def data_enqueue(loader, data_queue):
            for batch in loader:
                data_queue.put(batch)
        enqueue_worker = Thread(target=data_enqueue, args=(data, data_queue))
        enqueue_worker.daemon = True
        enqueue_worker.start()

        for _ in range(data.total_record // pGen.batch_image):
            r = result_queue.get()

            for i in range(pGen.batch_image):
                output_record = dict()
                for name_, r_ in zip(pTest.result_name, r):
                    output_record[name_] = r_[i].tolist()
                all_outputs.append(output_record)

        t2_s = time.time()
        logger.info("network uses: %.1f" % (t2_s - t1_s))

        # let user process all_outputs
        all_outputs.sort(key=lambda r: r["rec_id"])
        all_outputs, db = pTest.process_output(all_outputs, db)

        outputs_name = pTest.test_name if task == "test" else pTest.val_name
        for out, x in zip(all_outputs, db):
            if "pad" in x and x["pad"]:
                continue
            out_rec_id = out["rec_id"]
            x_rec_id = x["rec_id"]
            assert out_rec_id == x_rec_id, \
                "The record index for outputs should be the same as db. {} v.s. {}".format(
                    out_rec_id, x_rec_id)
            result = dict()
            for name in outputs_name:
                if name in x:
                    result[name] = x[name]
                else:
                    result[name] = out[name]
                if isinstance(result[name], np.ndarray):
                    result[name] = result[name].tolist()
                elif isinstance(result[name], mx.nd.NDArray):
                    result[name] = result[name].asnumpy().tolist()
            results.append(result)

        t3_s = time.time()
        logger.info("convert to list format uses: %.1f" % (t3_s - t2_s))

    if task == "test":
        import json
        json.dump(
            results,
            open("logs/{}/{}_result.json".format(pGen.name, pData.image_set[0]), "w"),
            sort_keys=True, indent=2)
        logger.info("test done.")
    else:
        val = pTest.test_metric
        val.process(results, logger)
        val.summarize()
        logger.info("val done.")
