from src.md.data.dataset import Dataset

import numpy as np
from scipy import stats


def f_regression(dataset: Dataset):

    """
    Scoring function for regression problems.
    It computes pearson correlation coefficients between each feature and the target variable.

    Then, it computes the F-value using the following F-statistics formula:
    F = (R^2 / (1 - R^2)) * (n - 2)
    p-values are computed using scipy.stats.f.sf (survival function).
    
    Parameters:
    dataset (Dataset): A dataset object containing the data and labels.

    Returns:
    Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]: A tuple containing the F-values and p-values as numpy arrays or float values.
    """
    # Input features
    X = dataset.X
    # Target variable
    y = dataset.y

    # Degrees of freedom for the F-statistics calculation
    deg_of_freedom = y.size - 2

    # List to store the Pearson correlation coefficients
    corr_coef = []
    for i in range(X.shape[1]):
        # Compute the Pearson correlation coefficient between feature i and target variable
        r, _ = stats.pearsonr(X[:, i], y)
        corr_coef.append(r)

    # Convert the correlation coefficients to a numpy array
    corr_coef = np.array(corr_coef)

     # Square the correlation coefficients
    corr_coef_squared = corr_coef ** 2
    # Compute the F-values
    F = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
    # Compute the p-values using the F-distribution survival function
    p = stats.f.sf(F, 1, deg_of_freedom)

    # Return the F-values and p-values as numpy arrays or float values
    return F, p
    



