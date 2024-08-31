import numpy as np
import matplotlib.pyplot as plt  
'''
Train a Perceptron to the entire Pima Indians data set (available in the Files > Data Sets section on
Canvas). Do not split the data into training and test for this problem!
a) Experiment with different learning rates and report the highest (training set) classification
accuracy you can obtain.
b) For the learning rate that gives the highest classification accuracy, plot the number of mis-
classification errors against the number of epochs, similar to Figure 2.7 from the textbook
(shown below for reference).
Submit your code in a file named pimaPerceptron.py.
'''# -*- coding: utf-8 -*-
"""
Fitting a Perceptron to a logical OR and XOR function from Lecture 3 on
2024/09/03.
Perceptron class is slightly modified from ch02.py accompanying Machine
Learning with PyTorch and Scikit-Learn by Raschka, Liu, and Mirjalili
Example is taken from Chapter 3 of Machine Learning: An Algorithmic
Perspective (2nd Edition) by Marsland
@author: Kevin S. Xu
"""
# Random state for initializing Perceptron weights
random_state = 1
class Perceptron:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []
        self.predictions_ = [] # Added by Kevin

        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            self.predictions_.append(self.predict(X)) # Added by Kevin
        return self
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
#%% Train Perceptron on pima data 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])
ppn_or = Perceptron(eta=0.1, n_iter=10, random_state=random_state)
ppn_or.fit(X, y_or)
plt.plot(range(1, len(ppn_or.errors_) + 1), ppn_or.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('OR data')
print(*ppn_or.predictions_, sep='\n')
#%% Train Perceptron on XOR data
y_xor = np.array([0, 1, 1, 0])
ppn_xor = Perceptron(eta=0.1, n_iter=50, random_state=random_state)
ppn_xor.fit(X, y_xor)
plt.plot(range(1, len(ppn_xor.errors_) + 1), ppn_xor.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('XOR data')
print(*ppn_xor.predictions_, sep='\n')
#%% Train Perceptron on XOR data in 3-D
X_3d = np.c_[X, np.array([[0], [0], [0], [1]])]
ppn_xor_3d = Perceptron(eta=0.1, n_iter=20, random_state=random_state)
ppn_xor_3d.fit(X_3d, y_xor)
plt.plot(range(1, len(ppn_xor_3d.errors_) + 1), ppn_xor_3d.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('XOR 3-D data')
print(*ppn_xor_3d.predictions_, sep='\n')