"""
Train a decision tree that computes the logical AND function using entropy as the splitting crite-
rion. How does it compare to the Perceptron solution? Draw possible decision boundaries for
both a trained decision tree and Perceptron.

The decision tree's accuracy is 1.0, while the perception solution's accuracy is also 1.0. In that sense, 
both do a pretty good job classifying this data properly. 

The possible decision boundaries are attached in a separate file named possible_decision_boundaries.pdf 
"""
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

'''This is the decision tree portion of the assignment.'''
X_train = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
y_train = np.array([0, 0, 0, 1])

#%% Fit Decision tree model
from sklearn.tree import DecisionTreeClassifier

entropy_model = DecisionTreeClassifier(criterion='entropy',
max_depth=4,
random_state=1)
entropy_model.fit(X_train, y_train)

#%% Check prediction accuracy on test set
predictions = entropy_model.predict(X_train)
accuracy = accuracy_score(y_train, predictions)
print(accuracy)


'''This is the perceptron solution'''
random_state = 1
class Perceptron:

    def __init__(self, learning_rate=0.1, iterations=100, random_state=1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.random_state = random_state
    
    def fit(self, x, y):
        weight_init = np.random.RandomState(self.random_state)
        self.weight = weight_init.normal(loc=0.0, scale=0.01, size=x.shape[1])
        self.bias = np.float64(0.)
        self.errors = []
        self.predictions = [] # Added by Kevin

        for i in range(self.iterations):
            errors = 0
            for xi, target in zip(x, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weight += update * xi
                self.bias += update
                errors += int(update != 0.0)
            self.errors.append(errors)
            self.predictions.append(self.predict(x)) # Added by Kevin
        return self
    
    def net_input(self, X):
        return np.dot(X, self.weight) + self.bias
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


perceptron = Perceptron(0.1, 10, random_state)

perceptron.fit(X_train, y_train)

predictions = perceptron.predict(X_train)
accuracy = accuracy_score(y_train, predictions)
print(accuracy)