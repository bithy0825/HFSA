import os
import time
import shutil
from loguru import logger
from tensorboardX import SummaryWriter


class Averager:
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer:
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return "{:.1f}h".format(t / 3600)
    elif t >= 60:
        return "{:.1f}m".format(t / 60)
    else:
        return "{:.1f}s".format(t)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip("/"))
    if os.path.exists(path):
        if remove and (
            basename.startswith("_")
            or input("{} exists, remove? (y/[n]): ".format(path)) == "y"
        ):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def setup_logger(log_file):
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True)
    logger.add(log_file, mode="a")
    return logger


def set_save_path(path, remove=True):
    ensure_path(path, remove=remove)
    log_file = os.path.join(path, "logger.log")
    logger_instance = setup_logger(log_file)
    writer = SummaryWriter(os.path.join(path, "tensorboard"))
    return logger_instance, writer
