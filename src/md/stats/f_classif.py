from typing import Tuple, Union
import numpy as np
from scipy import stats
from src.md.data.dataset import Dataset

def f_classif(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
    """
    Performs one-way ANOVA test on a given dataset.

    Parameters:
    -----------
    dataset: Dataset
        A dataset object containing the data and labels.
    
    Returns:
    --------
    Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]
        A tuple containing the F statistic and p-value.
        - If the dataset contains multiple classes, the F statistic and p-value for each feature is returned as arrays.
        - If the dataset contains only one class, the F statistic and p-value for that class is returned as floats.
    """
    
    # Get unique class labels
    features = dataset.get_classes()

    # Split the data into separate groups based on the class labels
    groups = [dataset.X[dataset.y == f] for f in features]

    # Perform one-way ANOVA test on the groups
    F, p = stats.f_oneway(*groups)

    # Return the F statistic and p-value as a tuple
    if len(features) > 1:
        return F, p
    else:
        return float(F), float(p)
