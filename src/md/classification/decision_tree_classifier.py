import numpy as np

class DecisionTreeClassifier:
    """
    Decision tree classifier.

    Parameters:
        max_depth (int or None, optional): The maximum depth of the tree. Defaults to None.
        pruning (bool, optional): Whether to perform pruning using reduced error pruning. Defaults to False.
        criterion (str, optional): The criterion to use for determining the best split at each node. Possible values are 'gini', 'entropy', and 'gain'. Defaults to 'gini'.

    Methods:
        fit(X, y):
            Builds the decision tree classifier.

            Parameters:
                X (array-like of shape (n_samples, n_features)): The input samples.
                y (array-like of shape (n_samples,)): The target values.

        predict(X):
            Predicts the target values for the input samples.

            Parameters:
                X (array-like of shape (n_samples, n_features)): The input samples.

            Returns:
                array-like of shape (n_samples,): The predicted target values.

        score(X, y):
            Returns the mean accuracy on the given test data and labels.

            Parameters:
                X (array-like of shape (n_samples, n_features)): The input samples.
                y (array-like of shape (n_samples,)): The true labels.

            Returns:
                float: The mean accuracy.

    Attributes:
        max_depth (int or None): The maximum depth of the tree.
        tree (tuple or int or None): The decision tree structure represented as a tuple or an integer if the node is a leaf.
        pruning (bool): Whether pruning is performed.
        criterion (str): The criterion used for determining the best split.

    """

    def __init__(self, max_depth=None, pruning = False,criterion = 'gini'):
        """
        Initializes the DecisionTreeClassifier.

        Parameters:
            max_depth (int or None, optional): The maximum depth of the tree.
            pruning (bool, optional): Whether to perform pruning using reduced error pruning.
            criterion (str, optional): The criterion to use for determining the best split at each node. 
        """

        self.max_depth = max_depth
        self.tree = None
        self.pruning = pruning
        self.criterion = criterion

    def _best_criteria(self,feature, y, criterion):
        """
        Computes the best criteria value for determining the best split at each node.

        Parameters:
            feature (array-like of shape (n_samples,)): The feature values.
            y (array-like of shape (n_samples,)): The target values.
            criterion (str): The criterion to use for determining the best split.

        Returns:
            float: The best criteria value.

        Raises:
            Exception: If an invalid criterion is provided.
        """

        if criterion == 'gini':
            return self._gini_index(y)
        elif criterion == 'entropy':
            return self._entropy(y)
        elif criterion == 'gain':
            return self._gain_ratio(feature,y)
        else:
            raise Exception("Invalid Criteria!") 

    def _gini_index(self, y):
        """
        Computes the Gini index impurity measure.

        Parameters:
            y (array-like of shape (n_samples,)): The target values.

        Returns:
            float: The Gini index impurity measure.
        """

        n_samples = len(y)
        _, counts = np.unique(y, return_counts=True)
        impurity = 1 - np.sum((counts / n_samples) ** 2)
        return impurity

    def _entropy(self,y):
        """
        Computes the entropy impurity measure.

        Parameters:
            y (array-like of shape (n_samples,)): The target values.

        Returns:
            float: The entropy impurity measure.
        """

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _gain_ratio(self, feature, y):
        """
        Computes the gain ratio for a feature.

        Parameters:
            feature (array-like of shape (n_samples,)): The feature values.
            y (array-like of shape (n_samples,)): The target values.

        Returns:
            float: The gain ratio.
        """

        n = len(y)
        values, counts = np.unique(feature, return_counts=True)
        entropy_before = self._entropy(y)
        split_info = - np.sum((counts / n) * np.log2(counts / n))
        entropy_after = entropy_before
        for value in values:
            subset_labels = y[feature == value]
            entropy_after -= self._entropy(subset_labels)
        return entropy_after / split_info if split_info != 0 else 0


    def _best_split(self, X, y):
        """
        Finds the best split point for the given data.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input samples.
            y (array-like of shape (n_samples,)): The target values.

        Returns:
            tuple: A tuple containing the index of the best feature, the best threshold value, and the best gain.

        """

        best_feature = None
        best_threshold = None
        best_gain = -np.inf

        # Loop over features and thresholds
        for feature_idx in range(X.shape[1]):
            feature = X[:, feature_idx]
            thresholds = np.unique(feature)

            for threshold in thresholds:
                # Calculate gain for split
                left_indices = feature <= threshold
                right_indices = feature > threshold

                left_y = y[left_indices]
                left_X = feature[left_indices ]
                right_y = y[right_indices]
                right_X = feature[right_indices]

                if len(left_y) == 0 or len(right_y) == 0 or len(left_X) == 0 or len(right_X) == 0:
                    continue

                gain = self._best_criteria(feature,y,self.criterion) - ((len(left_y) / len(y)) * self._best_criteria(left_X,left_y,self.criterion) 
                                                                        + (len(right_y) / len(y)) * self._best_criteria(right_X,right_y,self.criterion))

                # Update best split if necessary
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        """
        Recursively builds the decision tree.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input samples.
            y (array-like of shape (n_samples,)): The target values.
            depth (int): The current depth of the tree.

        Returns:
            tuple or int or None: The decision tree structure represented as a tuple or an integer if the node is a leaf.

        """

        if depth == self.max_depth:
            return 

        if len(np.unique(y)) == 1:
            return y[0]

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        # Majority voting
        if best_gain == 0:
            return np.argmax(np.bincount(y))

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth+1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth+1)

        node = (best_feature, best_threshold, left_tree, right_tree)

        if self.pruning:
            pruned_node = self._post_pruning(X, y, node)
            if pruned_node is not None:
                return pruned_node

        return node

    def _post_pruning(self, X, y, node):
        """
        Performs post-pruning on the decision tree.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input samples.
            y (array-like of shape (n_samples,)): The target values.
            node (tuple or int): The current node to prune.

        Returns:
            tuple or int or None: The pruned decision tree structure represented as a tuple or an integer if the node is a leaf.

        """

        #Reduced Error pruning
        if isinstance(node, tuple):
            feature, threshold, left_subtree, right_subtree = node

            # Prune left subtree
            pruned_left_subtree = self._post_pruning(X[X[:, feature] <= threshold], y[X[:, feature] <= threshold], left_subtree)

            # Prune right subtree
            pruned_right_subtree = self._post_pruning(X[X[:, feature] > threshold], y[X[:, feature] > threshold], right_subtree)

            # Calculate error before pruning
            predictions = self.predict(X)
            error_before = np.sum(predictions != y)

            # Calculate error after pruning
            if isinstance(pruned_left_subtree, int) and isinstance(pruned_right_subtree, int):
                predictions[X[:, feature] <= threshold] = pruned_left_subtree
                predictions[X[:, feature] > threshold] = pruned_right_subtree
                error_after = np.sum(predictions != y)
            else:
                error_after = error_before

            # Prune if error is reduced
            if error_after <= error_before:
                if isinstance(pruned_left_subtree, int) and isinstance(pruned_right_subtree, int):
                    return np.argmax(np.bincount(y))
                else:
                    return (feature, threshold, pruned_left_subtree, pruned_right_subtree)

        return node

    def fit(self, X, y):
        """
        Builds the decision tree classifier.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input samples.
            y (array-like of shape (n_samples,)): The target values.
        """

        self.tree = self._build_tree(X, y, 0)

    def predict(self, X):
        """
        Predicts the target values for the input samples.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input samples.

        Returns:
            array-like of shape (n_samples,): The predicted target values.
        """

        predictions = np.zeros(len(X))

        for i in range(len(X)):
            node = self.tree

            while isinstance(node, tuple):
                if X[i][node[0]] <= node[1]:
                    node = node[2]
                else:
                    node = node[3]

            predictions[i] = node

        return predictions
    
    def score(self, x, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input samples.
            y (array-like of shape (n_samples,)): The true labels.

        Returns:
            float: The mean accuracy.
        """
        
        return np.mean(x == y)

