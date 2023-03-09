from src.md.data.dataset import Dataset

import numpy as np
from typing import Tuple, Union
from scipy import stats


def f_regression(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray],Tuple[float, float]]:

    """
    Scoring function for regression problems.
    It computes pearson correlation coefficients between each feature and the target variable.

    Then, it computes the F-value using the following F-statistics formula:
    F = (R^2 / (1 - R^2)) * (n - 2)
    p-values are computed using scipy.stats.f.sf (survival function).
    """

    X = dataset.X
    y = dataset.y

    deg_of_freedom = y.size - 2

    corr_coef = []
    for i in range(X.shape[1]):
        r, _ = stats.pearsonr(X[:, i], y)
        corr_coef.append(r)

    corr_coef = np.array(corr_coef)

    corr_coef_squared = corr_coef ** 2
    F = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
    p = stats.f.sf(F, 1, deg_of_freedom)
    return F, p
    



