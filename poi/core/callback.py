import time
import logging
import mxnet as mx


class Speedometer(object):
    def __init__(self, batch_size, frequent=50, logger=None):
        self.batch_size = batch_size
        self.frequent = frequent
        self.logger = logging.getLogger() if logger is None else logger
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-" % (
                        param.epoch, count, speed)
                    for n, v in zip(name, value):
                        s += "%s=%f,\t" % (n, v)
                    self.logger.info(s)
                else:
                    self.logger.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                     param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


def do_checkpoint(prefix, frequent=1):
    def _callback(iter_no, sym, arg, aux):
        if (iter_no + 1) % frequent == 0:
            mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
    return _callback
