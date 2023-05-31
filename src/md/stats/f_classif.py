from scipy import stats
from src.md.data.dataset import Dataset

def f_classif(dataset: Dataset):
    """
    Performs the one-way ANOVA (Analysis of Variance) F-test on a dataset.

    Parameters:
        dataset (Dataset): The dataset containing the features and labels.

    Returns:
        tuple: The F statistic and p-value resulting from the F-test.

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
