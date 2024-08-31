import numpy as np 
import matplotlib.pyplot as plt

'''
Train a Perceptron to the entire Pima Indians data set (available in the Files > Data Sets section on
Canvas). Do not split the data into training and test for this problem!
a) Experiment with different learning rates and report the highest (training set) classification
accuracy you can obtain.

The highest classification accuracy came from some of the lowest learning rates. 

b) For the learning rate that gives the highest classification accuracy, plot the number of mis-
classification errors against the number of epochs, similar to Figure 2.7 from the textbook
(shown below for reference).

Please find the code below. It will plot the small learning rate, with the x axis being #epochs
and the y axis being #updates. 
Submit your code in a file named pimaPerceptron.py.
'''
# Random state for initializing Perceptron weights
random_state = 1
class Perceptron:

    def __init__(self, eta=0.0001, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
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
    
#Train Perceptron on pima data

#turn csv file into np array 
X = np.genfromtxt('C:/Machine Learning/pima-indians-diabetes.csv', delimiter=',', skip_header=1)

#target values 
target_values = X[:, -1]
ppn_pima = Perceptron(eta=0.9, n_iter=10, random_state=random_state)
ppn_pima.fit(X, target_values)
plt.plot(range(1, len(ppn_pima.errors_) + 1), ppn_pima.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Pima data')
print(*ppn_pima.predictions_, sep='\n')

plt.show()