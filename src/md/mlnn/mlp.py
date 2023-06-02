import numpy as np
from scipy import optimize

def sigmoid(x):
    """
    Compute the sigmoid function.

    Args:
        x (float or array-like): Input value(s) to the sigmoid function.

    Returns:
        float or array-like: Result of applying the sigmoid function to the input value(s).
    """
    return 1 / (1 + np.exp(-x))

class MLP:
    """
    Multi-Layer Perceptron (MLP) class for binary classification.

    Attributes:
        X (numpy.ndarray): Input features matrix of shape (m, n), where m is the number of samples and n is the number of features.
        y (numpy.ndarray): Target values vector of shape (m,).
        h (int): Number of hidden nodes in the MLP.
        W1 (numpy.ndarray): Weight matrix of the first layer of shape (h, n+1), where n+1 is the number of input features including the bias term.
        W2 (numpy.ndarray): Weight matrix of the second layer of shape (1, h+1), where h+1 is the number of hidden nodes including the bias term.
    Methods:
        setWeights(w1, w2): Set the weight matrices of the MLP.
        costFunction(w): Compute the cost function.
        build_model(X, y): Build the MLP model by optimizing the weights.
        predict(instance): Predict the output for a single instance.

    """

    def __init__(self, hidden_nodes=2):
        """
        Initialize the MLP object.

        Args:
            hidden_nodes (int, optional): Number of hidden nodes in the MLP. Defaults to 2.
        """
        self.X = None
        self.y = None
        self.h = hidden_nodes
        self.W1 = None
        self.W2 = None

    def setWeights(self, w1, w2):
        """
        Set the weight matrices of the MLP.

        Args:
            w1 (numpy.ndarray): Weight matrix of the first layer of shape (h, n+1).
            w2 (numpy.ndarray): Weight matrix of the second layer of shape (1, h+1).
        """
        self.W1 = w1
        self.W2 = w2

    def costFunction(self, w=None):
        """
        Compute the cost function.

        Args:
            w (numpy.ndarray, optional): Weight matrix to be used in the cost function.
                                        If not provided, the MLP's current weights will be used.

        Returns:
            float: Value of the cost function.
        """
        if w is not None:
            # if w is provided, update the MLP weights
            self.W1 = w[:self.h * self.X.shape[1]].reshape([self.h, self.X.shape[1]])
            self.W2 = w[self.h * self.X.shape[1]:].reshape([1, self.h+1])

        # get the number of samples
        m = self.X.shape[0]

        # dot product between x and the transpose of the weight of the first layer
        Z2 = np.dot(self.X, self.W1.T)
        # apply the sigmoid activation function to Z2
        A2 = np.hstack((np.ones([Z2.shape[0], 1]), sigmoid(Z2)))
        # dot product between A2 and the transpose of the weight of the second layer
        Z3 = np.dot(A2, self.W2.T)

        # get predictions
        predictions = sigmoid(Z3)
        # Calculate the squared error
        sqe = (predictions - self.y.reshape(m,1)) ** 2

        # get cost function value
        res = np.sum(sqe) / (2 * m)
        return res


    def build_model(self, X, y, random = False):
        """
        Build the MLP model by optimizing the weight matrices.

        Args:
            X (numpy.ndarray): Input features matrix.
            y (numpy.ndarray): Target values matrix.

        Returns:
            None
        """
        self.X, self.y = X, y

        # bias term
        self.X = np.hstack((np.ones([self.X.shape[0], 1]), self.X))

        if not random:
            # initialize the weights with zeros.
            self.W1 = np.zeros([self.h, self.X.shape[1]])
            self.W2 = np.zeros([1, self.h + 1])

        else:
            # initialize the weights with random values.
            np.random.seed(42)
            self.W1 = np.random.randint(low=-100, high= 100,size=(self.h, self.X.shape[1]))
            self.W2 = np.random.randint(low=-100, high= 100,size=(1, self.h + 1))

        # get number of elements in W1 and W2.
        size = self.h * self.X.shape[1] + self.h + 1

        # generate random weight values
        initial_w = np.random.rand(size)

        # minimize the cost function
        result = optimize.minimize(lambda w: self.costFunction(w), initial_w, method='BFGS',
                                options={"maxiter": 1000, "disp": False})

        # get optimized weights
        weights = result.x

        # update the weights of both layers
        self.W1 = weights[:self.h * self.X.shape[1]].reshape([self.h, self.X.shape[1]])
        self.W2 = weights[self.h * self.X.shape[1]:].reshape([1, self.h + 1])


    def predict(self, instance):
        """
        Predict the output for a single instance.

        Args:
            instance (array-like): Input features of a single instance.

        Returns:
            float: Predicted output value for the instance.
        """
        x = np.empty([self.X.shape[1]])
        # bias term
        x[0] = 1
        # feature values from the instance input
        x[1:] = np.array(instance[:self.X.shape[1] - 1])

        # dot product between the weight of the first layer and x
        z2 = np.dot(self.W1, x)
        a2 = np.empty([z2.shape[0]+1])
        # add bias term for a2
        a2[0] = 1
        a2[1:] = sigmoid(z2)
        # dot product between the weight of the second layer and a2
        z3 = np.dot(self.W2, a2)

        # get predicted output values
        predictions = sigmoid(z3)
        return predictions
    
