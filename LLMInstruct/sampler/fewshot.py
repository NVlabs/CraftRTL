# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

from typing import Dict
from datasets import concatenate_datasets, Dataset

from LLMInstruct.sampler import BaseSampler
from LLMInstruct.utils import read_data


class FewShotSampler(BaseSampler):

    def __init__(self, dataset_name: str, seed: int = 87):
        
        self.seed = seed
        
        if isinstance(dataset_name, list):
            dataset_list = []
            for i in dataset_name:
                dataset_list.append(self._read(dataset_name))
            self.dataset = concatenate_datasets(dataset_list)
        else:
            self.dataset = self._read(dataset_name)
        
        self.dataset = self._filter(self.dataset)
        self.cnt = 0
        self._generator = self._create_generator()
        
    def _filter(self, dataset):
        return dataset
        
    def _create_generator(self):
        while True:
            self.cnt += 1
            yield self.dataset[self.cnt % len(self.dataset)]

    def sample(self) -> dict:
        return next(self._generator)
    
    def _read(self, dataset_name):
        if isinstance(dataset_name, str):
            return read_data('', dataset_name).shuffle(seed=self.seed)
        elif isinstance(dataset_name, Dataset):
            return dataset_name
        else:
            raise Exception(f"Not support {type(dataset_name)} data type.")



class LimitedFewShotSampler(FewShotSampler):
    def _filter(self, dataset: Dataset, max_length_map: Dict[str, int]):

        def filter_by_max_length(example: dict):
            for field, max_length in max_length_map.items():
                if field in example and len(example[field]) > max_length:
                    return False
            return True

        after_filter_dataset = self.dataset.filter(filter_by_max_length)
        print(f"FewShotSampler filtering {len(after_filter_dataset)}/{len(dataset)} after before.")
        return after_filter_dataset
