import time


class BaseModel(object):
    def __init__(self, dummy_inputs=None, logger=None):
        self.name = "Base"
        self.num_loop = 100
        self.dummy_inputs = dummy_inputs
        self.logger = logger
        self.model = None
        self.last_outputs = None

    def init_model(self):
        pass

    def predict(self, inputs):
        pass

    def dummy_predict_loops(self, num_loop):
        for i in range(num_loop):
            outputs = self.predict(**self.dummy_inputs)
        self.dummy_outputs = outputs

    def perf(self):
        self.logger.info("Start to {} timing for {} loops...".format(self.name, self.num_loop))
        start = time.process_time()
        self.dummy_predict_loops(self.num_loop)
        cost_time = time.process_time() - start
        time_per_batch = cost_time / self.num_loop
        self.logger.info("Time used for each batch with shape of {}: {}".format(
            self.dummy_inputs["data"].shape, time_per_batch))

    def get_dummpy_outputs(self):
        return self.dummy_outputs
