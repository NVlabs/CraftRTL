# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

from abc import abstractmethod
from multiprocessing import Pool
from typing import List
from datasketch import MinHash, MinHashLSH
from nltk import ngrams

from .base import BaseFilter


class SimilarityFilter(BaseFilter):
    def add(self):
        pass


class RougeFilter(SimilarityFilter):

    def __init__(self, instructions: List[str] = None):
        from rouge_score import rouge_scorer
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        self.instructions = [] if instructions is None else instructions

    def validate(self, threshold: float = 0.7) -> bool:
        with Pool(4) as p:
            rouge_scores = p.map(partial(scorer.score, inst), self.instructions)
        rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
        if max(rouge_scores) > threshold:
            return False
        return True

    def add(self, inst: str):
        self.instructions.append(inst)


class JaccardFilter(SimilarityFilter):

    def __init__(
        self,
        instructions: List[str] = None,
        num_perm: int = 128,
        threshold: float = 0.7,
    ):
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.instructions = [] if instructions is None else instructions

        for e, inst in enumerate(self.instructions):
            m = self.minhash(inst)
            self.lsh.insert(str(e), m, check_duplication=False)

    def minhash(self, inst: str, ngram_size=5):
        m = MinHash(num_perm=self.num_perm)
        for word in ngrams(inst, ngram_size): 
            m.update(" ".join(word).encode("utf8"))
        return m

    def validate(self, inst: str):
        m = self.minhash(inst)
        result = self.lsh.query(m)
        if not result:
            return True
        return False  # need to be remove

    def add(self, inst: str):
        m = self.minhash(inst)
        self.lsh.insert(hash(inst), m, check_duplication=False)
        self.instructions.append(inst)
