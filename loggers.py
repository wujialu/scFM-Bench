# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import pickle
import gzip
from torch.utils.tensorboard import SummaryWriter
import copy
import pprint

logger = logging.getLogger()

def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(args):
    if args.enable_tensorboard:
        return TensorboardLogger(args)
    else:
        return Logger(args)


class Logger:
    def __init__(self, args):
        self.data = {}
        self.args = copy.deepcopy(vars(args))
        self.context = ""

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]

    def add_object(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def save(self, save_path, args):
        pickle.dump({'logged_data': self.data, 'args': self.args},
                    gzip.open(save_path, 'wb'))


class TensorboardLogger(Logger):
    def __init__(self, args):
        self.data = {}
        self.context = ""
        self.args = copy.deepcopy(vars(args))
        self.writer = SummaryWriter(log_dir=args.exp_dir)
        # print(self.args)
        pprint.pprint(self.args)
        self.writer.add_hparams(self.args, {})

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]
        self.writer.add_scalar(key, value, len(self.data[key]))

    def add_scalars(self, key, value, use_context=True):  #! X ~ multi-Y
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]
        self.writer.add_scalars(key, value, len(self.data[key]))

    def add_object(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def save(self, save_path):
        pickle.dump({'logged_data': self.data, 'args': self.args},
                    gzip.open(save_path, 'wb'))
        self.writer.flush()
