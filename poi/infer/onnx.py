from poi.infer.base import BaseModel
import numpy as np


class ONNXModel(BaseModel):
    def __init__(self, onnx_path,
                 batch_size=None, transforms=None, input_names=None,
                 base_record=None, logger=None):
        BaseModel.__init__(self, batch_size, transforms, input_names, base_record, logger)
        self.name = "ONNX"
        self.init_model(onnx_path)

    def init_model(self, onnx_path):
        import onnxruntime
        model = onnxruntime.InferenceSession(onnx_path)
        model.get_modelmeta()
        self.model = model
        self.input_names = model.get_inputs()[0].name
        self.logger.info("Warmup up...")
        self.inference_loops(10)

    def inference(self):
        outputs = self.model.run(None, self.inputs)
        return {n.name: np.asarray(o) for n, o in zip(self.model.get_outputs(), outputs)}
