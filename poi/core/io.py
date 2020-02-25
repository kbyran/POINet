import mxnet as mx
from multiprocessing import Process
from queue import Queue
from threading import Thread
import numpy as np


class DataLoader(mx.io.DataIter):
    """
    DataLoader consists of the main process and two attached process,
    which manipulate image augmentations and batch collection.
    """

    def __init__(self, roidb, sampler, transform, data_name, label_name,
                 batch_size=1, use_mp=False, num_worker=None, num_collector=None,
                 worker_queue_depth=None, collector_queue_depth=None, kv=None, valid_count=None):
        super().__init__(batch_size=batch_size)

        if kv:
            (self.rank, self.num_worker) = (kv.rank, kv.num_workers)
        else:
            (self.rank, self.num_worker) = (0, 1)

        # infer properties from roidb
        self.roidb = roidb
        self.batch_sampler = sampler
        # data processing utilities
        self.transform = transform

        # decide data and label names
        self.data_name = data_name
        self.label_name = label_name

        # status variable for synchronization between get_data and get_label
        self._cur = 0
        self.total_index = self.batch_sampler(roidb)
        self.valid_count = valid_count if valid_count is not None else len(roidb)
        self.data = None
        self.label = None
        self.debug = False
        self.result = None

        # multi-worker settings
        self.use_mp = use_mp
        self.num_worker = num_worker
        self.num_collector = num_collector
        self.index_queue = Queue()
        self.data_queue = Queue(maxsize=worker_queue_depth)
        self.result_queue = Queue(maxsize=collector_queue_depth)
        # self.worker = None
        # self.num_collector = None

        # get first batch to fill in provide_data and provide_label
        self._worker_start()
        self._collector_start()
        self.load_first_batch()
        self.reset()

    @property
    def index(self):
        return self.total_index[:self.valid_count]

    @property
    def total_record(self):
        return len(self.index) // self.batch_size * self.batch_size

    def __len__(self):
        return self.total_record

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def _insert_queue(self):
        for i in range(0, len(self.index), self.batch_size):
            batch_index = self.index[i: i + self.batch_size]
            if len(batch_index) == self.batch_size:
                self.index_queue.put(batch_index)

    def _worker_start(self):
        if self.use_mp:
            self.workers = \
                [Process(target=self.worker, args=[self.roidb, self.index_queue, self.data_queue])
                 for _ in range(self.num_worker)]
        else:
            self.workers = \
                [Thread(target=self.worker, args=[self.roidb, self.index_queue, self.data_queue])
                 for _ in range(self.num_worker)]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def _collector_start(self):
        self.collectors = \
            [Thread(target=self.collector, args=[]) for _ in range(self.num_collector)]

        for c in self.collectors:
            c.daemon = True
            c.start()

    def reset(self):
        self._cur = 0
        self.total_index = self.batch_sampler(self.roidb)
        self._insert_queue()

    def iter_next(self):
        return self._cur + self.batch_size <= len(self.index)

    def load_first_batch(self):
        self.index_queue.put(range(self.batch_size))
        self.next()

    def load_batch(self):
        self._cur += self.batch_size
        result = self.result_queue.get()
        return result

    def next(self):
        if self.debug and self.result is not None:
            return self.result

        if self.iter_next():
            # print("[worker] %d" % self.data_queue.qsize())
            # print("[collector] %d" % self.result_queue.qsize())
            result = self.load_batch()
            self.data = result.data
            self.label = result.label
            self.result = result
            return result
        else:
            raise StopIteration

    def worker(self, roidb, index_queue, data_queue):
        while True:
            batch_index = index_queue.get()

            records = []
            for index in batch_index:
                roi_record = roidb[index].copy()
                for trans in self.transform:
                    trans(roi_record)
                records.append(roi_record)
            data_batch = {}
            for name in self.data_name + self.label_name:
                data_batch[name] = np.ascontiguousarray(np.stack([r[name] for r in records]))
            # for trans in self.batch_transform:
            #     trans(data_batch)
            data_queue.put(data_batch)

    def collector(self):
        while True:
            record = self.data_queue.get()
            data = [mx.nd.from_numpy(record[name], zero_copy=True) for name in self.data_name]
            label = [mx.nd.from_numpy(record[name], zero_copy=True) for name in self.label_name]
            # print([record[name] for name in self.data_name])
            # print([record[name] for name in self.label_name])
            # print([d[0] for d in data])
            # print([l for l in label])
            # print([(d.dtype, d.shape) for d in data])
            # print([(l.dtype, l.shape) for l in label])
            provide_data = [(k, v.shape) for k, v in zip(self.data_name, data)]
            provide_label = [(k, v.shape) for k, v in zip(self.label_name, label)]
            data_batch = mx.io.DataBatch(data=data,
                                         label=label,
                                         provide_data=provide_data,
                                         provide_label=provide_label)
            self.result_queue.put(data_batch)
