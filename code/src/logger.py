# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import logging

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


def get_logger(params, log_file='ramen.log'):
    if os.path.exists(params.exp_path):
        print(f'| {params.exp_path} exists, consider to use a different experiment dir!')
    else:
        os.makedirs(params.exp_path)
    log_file = os.path.join(params.exp_path, log_file)

    logger = init_logger(log_file)
    return logger
