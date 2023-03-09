from typing import Callable
import numpy as np

from src.md.data.dataset import Dataset
from src.md.stats.f_classif import f_classif


class SelectKBest:
    def __init__(self, score_func: Callable = f_classif, k: int = 10):

        self.k = k
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':

        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:

        i = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[i]
        return Dataset(dataset.X[:, i], dataset.y, list(features), dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:

        self.fit(dataset)
        return self.transform(dataset)