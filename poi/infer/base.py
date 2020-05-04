import time
import logging
import numpy as np


BASE_RECORD = {
    "im_id": 0,
    "image_url": "imgs/leo_512_256.jpg",
    "gt_bbox": [0, 0, 256, 512]
}


class BaseModel(object):
    def __init__(self, batch_size=None, transforms=None, input_names=None,
                 base_record=None, logger=None):
        self.name = "Base"
        self.num_loop = 100
        assert batch_size is not None
        self.batch_size = batch_size
        assert transforms is not None
        self.transforms = transforms
        self.base_records = BASE_RECORD if base_record is None else base_record
        self.input_names = input_names
        self.init_inputs()
        # self.outputs = {}
        self.logger = logger if logger is not None else logging.getLogger()
        self.model = None

    def init_inputs(self):
        self.inputs = {}
        [trans(self.base_records) for trans in self.transforms]
        for name in self.input_names:
            self.inputs[name] = np.ascontiguousarray(
                np.stack([self.base_records[name]] * self.batch_size)
            )

    def init_model(self):
        pass

    def preprocess(self, records):
        for r in records:
            for trans in self.transforms:
                trans(r)
        for i, r in enumerate(records):
            for name in self.input_names:
                self.inputs[name][i] = r.pop(name)

    def inference(self):
        pass

    def postprocess(self, output, record):
        for i, r in enumerate(record):
            for k, v in output.items():
                r[k] = output[k][i]
        return record

    def predict(self, records):
        outputs = []
        for i in range(0, len(records), self.batch_size):
            record = records[i: i + self.batch_size]
            self.preprocess(record)
            print(record)
            output = self.inference()
            outputs += self.postprocess(output, record)
        return outputs

    def inference_loops(self, num_loop):
        for i in range(num_loop):
            outputs = self.inference()
        self.dummy_outputs = outputs

    def perf(self):
        self.logger.info("Start to {} timing for {} loops...".format(self.name, self.num_loop))
        start = time.process_time()
        self.inference_loops(self.num_loop)
        cost_time = time.process_time() - start
        time_per_batch = cost_time / self.num_loop
        self.logger.info("Time used for each batch with shape of {}: {}".format(
            self.inputs["data"].shape, time_per_batch))

    def get_dummy_outputs(self):
        return self.dummy_outputs

    def close(self):
        pass
