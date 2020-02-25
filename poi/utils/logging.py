import os
import functools
import logging
from datetime import datetime
from pytz import utc, timezone


@functools.lru_cache()
def set_logger(path, rank=0):
    def custom_time(*args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone("Asia/Shanghai")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()

    logger = logging.getLogger()

    # level
    logger.setLevel(logging.INFO)
    # format and datefmt
    fmt = "%(asctime)s %(message)s"
    date_fmt = "%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    formatter.converter = custom_time
    # filename, echo node has its own log file
    if rank != 0:
        path = "{}.rank{}".format(path, rank)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_handler = logging.FileHandler(filename=path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # stream, only for rank 0
    if rank == 0:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger
