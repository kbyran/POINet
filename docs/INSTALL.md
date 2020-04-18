# Installation

## POI

Clone and install,

```
git clone https://github.com/kbyran/poi.git
cd poi && python3 setup.py install --user
```

## (Optional) ONNX and ONNX Runtime

`onnx==1.6.0` and `onnxruntime-gpu==1.2.0` is needed to export ONNX model.

```
pip3 install onnx==1.6.0 onnxruntime-gpu==1.2.0 --user
```

## (Optional) TensorRT

`tensorrt==7.0.0.11` is needed to export TensorRT model. Follow [TensorRT installation guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing) for details.
