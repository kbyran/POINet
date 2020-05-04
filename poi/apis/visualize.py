import os
import time
import json
import pprint
import importlib

from poi.utils.logging import set_logger
from poi.infer.mx import MXModel
from poi.infer.onnx import ONNXModel
from poi.infer.trt import TRTModel


def visualize_net(config, epoch=None, image=None, infer="mxnet", **kwargs):
    # init config
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    cfg = importlib.import_module(config.replace(".py", "").replace("/", "."))
    pGen, pKv, pData, pModel, pOpt, pTest, pIO, pMetric = cfg.get_config(task="export")

    # logging
    log_path = os.path.join("logs", pGen.name, "log_visualize.txt")
    logger = set_logger(log_path)

    # vis type
    vis_type = config.split("/")[1].split("_")[0]
    logger.info("Visualization type: {}".format(vis_type))

    # load json
    json_info = json.load(open(image))
    logger.info(pprint.pformat(json_info))

    if infer == "mxnet":
        pModel.test_symbol.save(pTest.model.prefix + "_export.json")
        model = MXModel(pTest.model.prefix, epoch or pTest.model.epoch, pKv.gpus[0],
                        pGen.batch_image, pIO.transform, pIO.data_name, None, logger)
    elif infer == "onnx":
        onnx_path = pTest.model.prefix + ".onnx"
        model = ONNXModel(onnx_path, pGen.batch_image, pIO.transform, pIO.data_name, None, logger)
    elif infer == "tensorrt":
        trt_path = pTest.model.prefix + ".trt"
        model = TRTModel(trt_path, pKv.gpus[0],
                         pGen.batch_image, pIO.transform, pIO.data_name, None, logger)

    if vis_type == "reid":
        from poi.vis.reid import vis_reid
        assert "image_a" in json_info and "image_b" in json_info
        output_a, output_b = model.predict([json_info["image_a"], json_info["image_b"]])
        vis_reid(output_a["image_url"], output_a["feature"],
                 output_b["image_url"], output_b["feature"])
    elif vis_type == "attr":
        from poi.vis.attr import vis_attr
        assert "image" in json_info
        output = model.predict([json_info["image"]])[0]
        vis_attr(output["image_url"], pTest.metric.attr_name, output["attribute"])
    elif vis_type == "pose":
        from poi.vis.pose import vis_pose
        assert "image" in json_info
        output = model.predict([json_info["image"]])[0]
        vis_pose(output["image_url"], output["affine"], output["max_idx"], output["max_val"],
                 kp_names=pTest.kp_name, skeletons=pTest.skeleton)
    elif vis_type == "densepose":
        from poi.vis.densepose import vis_densepose
        assert "image" in json_info
        output = model.predict([json_info["image"]])[0]
        vis_densepose(output["image_url"], output["gt_bbox"], output["ann_index_lowres"],
                      output["index_uv_lowres"], output["u_lowres"], output["v_lowres"])

    model.close()
    time.sleep(10)
