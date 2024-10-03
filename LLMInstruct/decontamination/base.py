# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

from abc import ABC, abstractmethod


class BaseFilter(ABC):

    @abstractmethod
    def validate(self) -> bool:
        # false -> remove
        pass
