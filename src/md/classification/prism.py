import numpy as np


class PRISM():
    """
    PRISM (PRojection Interpretation-based Sample Mining) algorithm for rule-based classification.

    Parameters:
        dataset: An instance of a dataset class containing X (features) and y (labels).

    Attributes:
        rules (list): A list of rules learned by the PRISM algorithm.
        dataset: The dataset used for training.

    """

    def __init__(self,dataset):

        self.rules = []
        self.dataset = dataset

    def fit(self, X,y):
        """
        Fits the PRISM model to the training data.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input features.
            y (array-like of shape (n_samples,)): The target labels.
        """

        if len(np.unique(y)) == 1:
            self.rules.append((None, y[0]))
            return
        
        while len(np.unique(y)) > 1:
            best_rule, best_score = None, -np.inf

            #Iterates each feature of the dataset
            for feature_idx, _ in enumerate(X.shape):
                unique_values = np.unique(X[:, feature_idx])
                covered = np.equal.outer(X[:, feature_idx], unique_values)
                #Creates a rule for each value of the feature
                for i, value in enumerate(unique_values):
                    rule = (feature_idx, value)
                    #Gets score of created rule
                    score = self.probability(y,covered[:, i])
                    #Checks if score improves with new rule
                    if score > best_score:
                        best_rule, best_score = rule, score
            #Saves the rule
            self.rules.append(best_rule)
            #Removes all samples covered by added rule
            X, y = self.remove_covered_by_rule(X, y, best_rule)

    def probability(self, y, covered):
        """
        Calculates the probability of the most frequent class label within the covered samples.

        Parameters:
            y (array-like of shape (n_samples,)): The target labels.
            covered (array-like of shape (n_samples,)): A boolean array indicating the covered samples.

        Returns:
            float: The probability of the most frequent class label within the covered samples.
        """

        targets = y[covered]
        _ ,counts = np.unique(targets,return_counts=True)
        return np.argmax(counts) / len(targets)

    def remove_covered_by_rule(self, X, y, rule):
        """
        Removes the samples covered by a given rule from the dataset.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input features.
            y (array-like of shape (n_samples,)): The target labels.
            rule (tuple): The rule specifying the feature index and value.

        Returns:
            array-like of shape (n_samples, n_features): The updated input features after removing covered samples.
            array-like of shape (n_samples,): The updated target labels after removing covered samples.
        """

        feature_idx, value = rule
        not_covered = X[:, feature_idx] != value
        return X[not_covered], y[not_covered]

    def predict(self, X):
        """
        Predicts the target labels for the input features.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input features.

        Returns:
            array-like of shape (n_samples,): The predicted target labels.
        """
        
        predictions = np.zeros(X.shape[0])
        for rule in self.rules:
            feature_idx, value = rule
            covered = X[:, feature_idx] == value

            target_covered = self.dataset.y[self.dataset.X[:, feature_idx] == value]
            uniques, counts = np.unique(target_covered,return_counts=True)
            most_common = uniques[np.argmax(counts)]
            predictions[covered] = most_common

        return predictions