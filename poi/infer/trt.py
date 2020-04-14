from poi.infer.base import BaseModel
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np


# a helper to host device storage
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, shape):
        self.host = host_mem
        self.device = device_mem
        self.shape = shape

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTModel(BaseModel):
    def __init__(self, trt_path, ctx_id, dummy_inputs=None, logger=None):
        BaseModel.__init__(self, dummy_inputs, logger)
        self.name = "TensorRT"
        self.init_model(trt_path, ctx_id)

    def init_model(self, trt_path, ctx_id):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        cuda.init()
        device = cuda.Device(ctx_id)
        self.ctx = device.make_context()
        with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        self.input_buffs = {}
        self.output_buffs = {}
        self.bindings = []
        self.stream = cuda.Stream()
        for name in engine:
            shape = engine.get_binding_shape(name)
            size = trt.volume(shape) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(name))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(name):
                self.input_buffs[name] = HostDeviceMem(host_mem, device_mem, shape)
            else:
                self.output_buffs[name] = HostDeviceMem(host_mem, device_mem, shape)

        self.model = engine.create_execution_context()
        self.logger.info("Warmup up...")
        self.dummy_predict_loops(10)

    def predict(self, **kwargs):
        for name in kwargs:
            # Transfer input data to page locked memory
            np.copyto(self.input_buffs[name].host, kwargs[name].flatten())
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(self.input_buffs[name].device,
                                   self.input_buffs[name].host, self.stream)
        # Run inference.
        self.model.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for name in self.output_buffs:
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(self.output_buffs[name].host,
                                   self.output_buffs[name].device, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return {name: np.reshape(self.output_buffs[name].host, self.output_buffs[name].shape)
                for name in self.output_buffs}

    def close(self):
        self.ctx.pop()
