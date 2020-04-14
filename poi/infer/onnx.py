from poi.infer.base import BaseModel
import numpy as np


class ONNXModel(BaseModel):
    def __init__(self, onnx_path, dummy_inputs, logger):
        BaseModel.__init__(self, dummy_inputs, logger)
        self.name = "ONNX"
        self.init_model(onnx_path)

    def init_model(self, onnx_path):
        import onnxruntime
        model = onnxruntime.InferenceSession(onnx_path)
        model.get_modelmeta()
        self.model = model
        self.input_names = model.get_inputs()[0].name
        self.logger.info("Warmup up...")
        self.dummy_predict_loops(10)

    def predict(self, **kwargs):
        outputs = self.model.run(None, kwargs)
        return {n.name: np.asarray(o) for n, o in zip(self.model.get_outputs(), outputs)}
