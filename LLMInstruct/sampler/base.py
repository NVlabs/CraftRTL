# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

import random
import numpy as np
from abc import ABC, abstractmethod


class BaseSampler(ABC):

    @abstractmethod
    def sample(self):
        pass
