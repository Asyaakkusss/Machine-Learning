'''
The parity problem returns 1 if the number of inputs that are 1 is even, and 0 otherwise. Can a
Perceptron learn this problem for 3 inputs? Design the network and train it to the entire data set.
Submit the table of input and target combinations, a diagram of the network structure, and the
highest training set accuracy you are able to obtain with a Perceptron along with the weights that
achieve this accuracy. Submit your code in a file named parity_perceptron.py.  


Answer: No, the perceptron cannot learn this problem for 3 inputs because the problem is not linearly separable
in the case of 3 inputs. 

Tables for input and target combinations: 

  Inputs          Targets
-----------       -------
|a   b   c|       |--y--|
|---------|       |-----|
|0   0   0|       |  0  |
|0   0   1|       |  0  |
|0   1   0|       |  0  |
|0   1   1|       |  1  |
|1   0   0|       |  0  |
|1   0   1|       |  1  |
|1   1   0|       |  1  |
|1   1   1|       |  0  |
-----------       -------

Diagram of network structure: see ProblemTwoNetwork.pdf

TODO: ask professor about your perceptron and whether you should play around with weights and put them as an input. Is he 
interested in original/initial weights or all of them in general? 

Highest training set accuracy obtained: 0.75. The initial weights were [ 0.01624345 -0.00611756 -0.00528172]. The learning 
rate was 0.0001. 

'''
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score

#feature
random_state = 1
class Perceptron:

    def __init__(self, learning_rate=0.0001, iterations=150, random_state=1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.random_state = random_state
    
    def fit(self, x, y):
        weight_init = np.random.RandomState(self.random_state)
        self.weight = weight_init.normal(loc=0.0, scale=0.01, size=x.shape[1])
        print("initial weight: ", self.weight)
        self.bias = np.float64(0.)
        self.errors = []
        self.predictions = [] # Added by Kevin

        for i in range(self.iterations):
            errors = 0
            for xi, target in zip(x, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weight += update * xi
                print("subsequent weights: ", self.weight)
                self.bias += update
                errors += int(update != 0.0)
            self.errors.append(errors)
            self.predictions.append(self.predict(x)) # Added by Kevin
        return self
    
    def net_input(self, X):
        return np.dot(X, self.weight) + self.bias
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    


#creating the X and y inputs 
X=np.array([[0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1]])

y = np.array([0, 0, 0, 1, 0, 1, 1, 0])


perceptron = Perceptron(0.0001, 50, random_state)

perceptron.fit(X, y)


plt.plot(range(0, len(perceptron.errors)), perceptron.errors, marker = 'o')

plt.xlabel("Epochs")
plt.ylabel("Number of Updates")
plt.title("Parity Problem: Convergence Investigation")

plt.show()

predictions = perceptron.predict(X)
accuracy = accuracy_score(y, predictions)
print(accuracy)


