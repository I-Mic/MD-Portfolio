import numpy as np
from src.md.data.dataset import Dataset

class VarianceThreshold:
    """
    Feature selector that removes all features whose variance doesn't meet a given threshold.

    Parameters:
    -----------
    threshold: float, default=0.0
        The threshold below which the features will be removed.

    Attributes:
    -----------
    threshold: float
        The threshold below which the features will be removed.
    variance: ndarray, shape (n_features,)
        The variances of each feature in the input data.
    """

    def __init__(self, threshold: float = 0.0):
        """
        Initialize a VarianceThreshold object.

        Parameters:
        -----------
        threshold: float, default=0.0
            The threshold below which the features will be removed.
        """

        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        self.threshold = threshold
        self.variance = None

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Compute the variance of each feature in the input data and store them in the variance attribute.

        Parameters:
        -----------
        dataset: Dataset
            A dataset object containing the data and labels.

        Returns:
        --------
        self: VarianceThreshold
            The fitted object.
        """

        self.variance = np.nanvar(dataset.X, axis=0)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Reduce the input data to the features with variances greater than the threshold.

        Parameters:
        -----------
        dataset: Dataset
            A dataset object containing the data and labels.

        Returns:
        --------
        Dataset
            A new dataset object containing the selected features.
        """

        features_mask = self.variance > self.threshold
        features = np.array(dataset.features)[features_mask]
        return Dataset(dataset.X[:, features_mask], dataset.y, list(features), dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Compute the variance of each feature in the input data and reduce it to the features with variances greater than the threshold.

        Parameters:
        -----------
        dataset: Dataset
            A dataset object containing the data and labels.

        Returns:
        --------
        Dataset
            A new dataset object containing the selected features.
        """

        self.fit(dataset)
        return self.transform(dataset)
