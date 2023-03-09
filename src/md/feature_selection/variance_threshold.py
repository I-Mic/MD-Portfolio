import numpy as np
from src.md.data.dataset import Dataset

class VarianceThreshold:
    def __init__(self, threshold: float = 0.0):

        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        self.threshold = threshold
        self.variance = None

    def fit(self, dataset:Dataset) -> 'VarianceThreshold':

        self.variance = np.nanvar(dataset.X,axis=0)
        return self

    def transform(self, dataset: Dataset) -> Dataset:

        features_mask = self.variance > self.threshold
        features = np.array(dataset.features)[features_mask]
        return Dataset(dataset.X, dataset.y, list(features), dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        
        self.fit(dataset)
        return self.transform(dataset)
    


