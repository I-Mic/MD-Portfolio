import numpy as np

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.vocab_size = X.shape[1]
        self.word_counts = np.zeros((len(self.classes), self.vocab_size))
        self.class_counts = np.zeros(len(self.classes))
        self.before = np.zeros(len(self.classes))
        
        # Get sum of words classes and prob before
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.word_counts[i] = np.sum(X_c, axis=0)
            self.class_counts[i] = np.sum(X_c)
            self.before[i] = X_c.shape[0] / X.shape[0]
            
    def predict(self, X):
        #Get log prob of each class for each instance
        likelihood = np.zeros((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            prob = np.log(self.before[i])
            prob += np.sum(np.log((self.word_counts[i] + self.alpha) / (self.class_counts[i] + self.alpha * self.vocab_size)) * X, axis=1)
            likelihood[:, i] = prob
        
        # Returns class with greater log prob
        return self.classes[np.argmax(likelihood, axis=1)]
    
    def score(self, x, y):
        return np.mean(x == y)