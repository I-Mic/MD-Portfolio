from typing import Tuple, Union

import numpy as np
from scipy import stats

from src.md.data.dataset import Dataset


def f_classif(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray],Tuple[float, float]]:
    
    features = dataset.get_classes()
    groups = [dataset.X[dataset.y == f] for f in features]
    F, p = stats.f_oneway(*groups)
    return F, p