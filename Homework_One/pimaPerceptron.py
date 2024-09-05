import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


'''
Train a Perceptron to the entire Pima Indians data set (available in the Files > Data Sets section on
Canvas). Do not split the data into training and test for this problem!
a) Experiment with different learning rates and report the highest (training set) classification
accuracy you can obtain.

The highest classification accuracy came from any learning rate between 0.6 and 0.9. The accuracy at those points
was reported as 1.0. 

b) For the learning rate that gives the highest classification accuracy, plot the number of mis-
classification errors against the number of epochs, similar to Figure 2.7 from the textbook
(shown below for reference).

Please find the code below. It will plot one of the learning rates, with the x axis being #epochs
and the y axis being #updates. 
Submit your code in a file named pimaPerceptron.py.
'''
# Random state for initializing Perceptron weights
random_state = 1
class Perceptron:

    def __init__(self, learning_rate=0.1, iterations=150, random_state=1):
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
    
#Train Perceptron on pima data

#turn csv file into np array 
X = np.genfromtxt('C:/Machine Learning/pima-indians-diabetes.csv', delimiter=',', skip_header=1)

#features
features = X[:, :-1]
print(np.shape(features))

#target values 
target_values = X[:, -1]
print(np.shape(target_values))

scaler = StandardScaler()

x_scale = scaler.fit_transform(features)

ppn_pima = Perceptron(learning_rate=0.6, iterations=150, random_state=random_state)
ppn_pima.fit(x_scale, target_values)
plt.plot(range(1, len(ppn_pima.errors) + 1), ppn_pima.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Pima data')

plt.show()

predictions = ppn_pima.predict(x_scale)
print(np.shape(predictions))
accuracy = accuracy_score(target_values, predictions)
print(accuracy)