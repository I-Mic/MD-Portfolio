from typing import Callable
import numpy as np

from src.md.data.dataset import Dataset
from src.md.stats.f_classif import f_classif

class SelectKBest:
    """
    A feature selection class that selects the k best features according to a given scoring function.

    Args:
        score_func (Callable): The scoring function used to evaluate the features. By default, the f_classif 
            function from the f_classif module is used.
        k (int): The number of best features to select. By default, k is set to 10.
    """
    def __init__(self, score_func: Callable = f_classif, k: int = 10):
        self.k = k
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        Fit the feature selector to the dataset and compute the scores for each feature.

        Args:
            dataset (Dataset): The dataset to fit the feature selector to.

        Returns:
            self: The fitted feature selector object.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the input dataset to contain only the k best features as selected by the feature selector.

        Args:
            dataset (Dataset): The dataset to select the best features from.

        Returns:
            Dataset: The transformed dataset containing only the k best features.
        """
        i = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[i]
        return Dataset(dataset.X[:, i], dataset.y, list(features), dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit the feature selector to the dataset and transform the input dataset to contain only the k best features.

        Args:
            dataset (Dataset): The dataset to fit the feature selector to and select the best features from.

        Returns:
            Dataset: The transformed dataset containing only the k best features.
        """
        self.fit(dataset)
        return self.transform(dataset)
