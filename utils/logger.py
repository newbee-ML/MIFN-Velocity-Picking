#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import os
import time
import logging


def MyLog(name, RootPath):
    logger = logging.getLogger(name)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(RootPath, '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def setup_logger(logpth):
    if not osp.exists(logpth):
        os.makedirs(logpth)
    logfile = 'MIFN-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


