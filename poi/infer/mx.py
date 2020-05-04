from poi.infer.base import BaseModel
import mxnet as mx
from poi.core.model import load_params


class MXModel(BaseModel):
    def __init__(self, model_prefix, model_epoch, ctx_id,
                 batch_size=None, transforms=None, input_names=None,
                 base_record=None, logger=None):
        BaseModel.__init__(self, batch_size, transforms, input_names, base_record, logger)
        self.name = "MXNet"
        self.init_model(model_prefix, model_epoch, ctx_id)

    def init_model(self, model_prefix, model_epoch, ctx_id):
        self.sym = mx.sym.load(model_prefix + "_export.json")
        arg_params, aux_params = load_params(model_prefix, model_epoch)
        ctx = mx.gpu(ctx_id)
        model = self.sym.simple_bind(ctx=ctx, grad_req='null', force_rebind=True,
                                     **{k: v.shape for k, v in self.inputs.items()})
        model.copy_params_from(arg_params, aux_params, allow_extra_params=True)
        self.model = model
        self.logger.info("Warmup up...")
        self.inference_loops(10)

    def inference(self):
        outputs = self.model.forward(is_train=False, **self.inputs)
        output_dict = {}
        for name, value in zip(self.sym.list_outputs(), outputs):
            name = "_".join(name.split("_")[:-1])
            value.wait_to_read()
            output_dict[name] = value.asnumpy()
        return output_dict
