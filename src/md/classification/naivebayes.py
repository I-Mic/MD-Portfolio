import numpy as np

class NaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Initialize a NaiveBayes instance.
        
        Parameters:
        -----------
        alpha : float, optional (default=1.0)
            Additive smoothing parameter. A larger value of alpha results in more smoothing.
        """
        self.alpha = alpha
        
    def fit(self, X, y):
        """
        Fit a Naive Bayes model to the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training input samples.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Store unique classes
        self.classes = np.unique(y)
        # Number of features (vocabulary size)
        self.vocab_size = X.shape[1]
        # Initialize word counts and class counts
        self.word_counts = np.zeros((len(self.classes), self.vocab_size))
        self.class_counts = np.zeros(len(self.classes))
        # Initialize prior probabilities
        self.before = np.zeros(len(self.classes))
        
       # Iterate over each class
        for i, c in enumerate(self.classes):
            # Select samples belonging to the current class
            X_c = X[y == c]
            # Calculate word counts for the current class
            self.word_counts[i] = np.sum(X_c, axis=0)
            # Calculate class counts
            self.class_counts[i] = np.sum(X_c)
            # Calculate prior probability of the current class
            self.before[i] = X_c.shape[0] / X.shape[0]
        
            
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test input samples.
        
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels.
        """

        # Initialize an array to store the log-likelihood values for each sample and class
        likelihood = np.zeros((X.shape[0], len(self.classes)))
        
        # Iterate over each class
        for i, c in enumerate(self.classes):
            # Calculate the log-likelihood for the current class
        
            # Compute the log prior probability of the current class
            prob = np.log(self.before[i])
             # Compute the log likelihood of each feature given the current class
            prob += np.sum(np.log((self.word_counts[i] + self.alpha) / (self.class_counts[i] + self.alpha * self.vocab_size)) * X, axis=1)
            # Store the log-likelihood values for the current class
            likelihood[:, i] = prob
        
        # Determine the predicted class for each sample by selecting the class with the highest log-likelihood
        return self.classes[np.argmax(likelihood, axis=1)]
    
    def score(self, x, y):
        return np.mean(x == y)
