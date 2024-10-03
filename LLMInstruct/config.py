# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

import os
import logging


def get_logging_level():
    return os.getenv("LOG_LEVEL", "INFO")


def configure_logging():
    logging_level = get_logging_level().upper()
    numeric_level = getattr(logging, logging_level, None)

    if not isinstance(numeric_level, int):
        raise Exception(f"Invalid log level: {numeric_level}")

    logging.basicConfig(
        level=numeric_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="[%(asctime)s] [%(process)s] [%(levelname)s] [%(module)s]: #%(funcName)s @%(lineno)d: %(message)s",
    )
    logging.info(f"Logging level: {logging_level}")
