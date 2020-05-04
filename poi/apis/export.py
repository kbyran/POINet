import os
import time
import importlib
import pprint
import numpy as np
import mxnet as mx

from poi.utils.logging import set_logger
from poi.core.model import load_params
from poi.infer.mx import MXModel
from poi.infer.onnx import ONNXModel
from poi.infer.trt import TRTModel


def mx2onnx(input_shapes, model_prefix, model_epoch, onnx_path):
    # load monkey patch for onnx converter
    from poi.utils.onnx_patch import onnx_mxnet
    sym = mx.sym.load(model_prefix + "_export.json")
    arg_params, aux_params = load_params(model_prefix, model_epoch)
    params = dict()
    params.update(arg_params)
    params.update(aux_params)
    converted_onnx_path = onnx_mxnet.export_model(
        sym, params, input_shapes, np.float32, onnx_path, False)
    return converted_onnx_path


def onnx2trt(input_shapes, onnx_path, trt_path):
    shape_str = "input:" + "x".join([str(i) for i in input_shapes[0]])
    command = "trtexec --onnx={} --shapes={} --saveEngine={} --verbose".format(
        onnx_path, shape_str, trt_path)
    os.system(command)
    return trt_path


def load_inputs(batch_size, transforms, input_names):
    inputs = {}
    input_records = [{"im_id": i, "image_url": "imgs/leo_512_256.jpg",
                      "gt_bbox": [0, 0, 256, 512]} for i in range(batch_size)]
    for r in input_records:
        for trans in transforms:
            trans(r)
    for name in input_names:
        inputs[name] = np.ascontiguousarray(np.stack([r["data"] for r in input_records]))
    return inputs


def allclose(dict_a, dict_b, logger=None):
    for name in dict_a:
        if name not in dict_b:
            logger.info("Output {} is not existed".format(name))
            return False
        if not np.allclose(dict_a[name], dict_b[name], rtol=1e-3, atol=1e-3):
            logger.info("Output {} is not equal within a tolerance".format(name))
            return False
    return True


def export_net(config, epoch=None, **kwargs):
    # init config
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    cfg = importlib.import_module(config.replace(".py", "").replace("/", "."))
    pGen, pKv, pData, pModel, pOpt, pTest, pIO, pMetric = cfg.get_config(task="export")
    sym = pModel.test_symbol
    sym.save(pTest.model.prefix + "_export.json")

    # logging
    log_path = os.path.join("logs", pGen.name, "log_export.txt")
    logger = set_logger(log_path)

    # load test images
    inputs = load_inputs(pGen.batch_image, pIO.transform, pIO.data_name)
    output_shapes = sym.infer_shape(**{k: v.shape for k, v in inputs.items()})[1]
    output_names = ["_".join(name.split("_")[:-1]) for name in sym.list_outputs()]
    model_info = {
        "input": [{"name": k, "shape": v.shape} for k, v in inputs.items()],
        "output": [{"name": k, "shape": s} for k, s in zip(output_names, output_shapes)]
    }
    logger.info(pprint.pformat(model_info))

    # convert onnx
    input_shapes = [v.shape for v in inputs.values()]
    onnx_path = pTest.model.prefix + ".onnx"
    onnx_path = mx2onnx(input_shapes, pTest.model.prefix, epoch or pTest.model.epoch, onnx_path)

    # convert tensorrt
    trt_path = pTest.model.prefix + ".trt"
    trt_path = onnx2trt(input_shapes, onnx_path, trt_path)
    logger.info("TensorRT model {} is saved".format(trt_path))

    model = MXModel(pTest.model.prefix, epoch or pTest.model.epoch, pKv.gpus[0],
                    pGen.batch_image, pIO.transform, pIO.data_name, None, logger)
    model.perf()
    mx_outputs = model.get_dummy_outputs()

    # test onnx
    model = ONNXModel(onnx_path, pGen.batch_image, pIO.transform, pIO.data_name, None, logger)
    model.perf()
    onnx_outputs = model.get_dummy_outputs()
    if allclose(mx_outputs, onnx_outputs, logger):
        logger.info("Allclosed for converting MXNet model into ONNX")
    else:
        logger.info("Not allclosed for converting MXNet model into ONNX")

    # test tensorrt
    model = TRTModel(trt_path, pKv.gpus[0],
                     pGen.batch_image, pIO.transform, pIO.data_name, None, logger)
    model.perf()
    trt_outputs = model.get_dummy_outputs()
    if allclose(mx_outputs, trt_outputs, logger):
        logger.info("Allclosed for converting MXNet model into TensorRT")
    else:
        logger.info("Not allclosed for converting MXNet model into TensorRT")
    model.close()
    time.sleep(10)
