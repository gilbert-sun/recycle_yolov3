# -*- coding: utf-8 -*-
# !/usr/bin/env python3.6

import logging
import logging.handlers
import sys

def setup_custom_logger(name, log_name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d_%H:%M:%S')

    # mode="w" in test environment, change to "a" in production
    # maxBytes=1000000000 = 1 GB
    rotating_handler = logging.handlers.RotatingFileHandler("logs/" + log_name,
                                                             mode="a",
                                                             maxBytes=1000000000,
                                                             backupCount=5)
    rotating_handler.setFormatter(formatter)

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(rotating_handler)
    logger.addHandler(screen_handler)

    return logger
