'''
Train a polynomial regression on the data in trainPoly.csv using ordinary least squares (OLS)
for each value of maximum degree 𝑑 = 1, ... , 9. Do not first standardize or use any other feature
scaling! Use the trained model to make and evaluate predictions on the data in testPoly.csv.
In both files, the first column contains the input, and the second column contains the prediction
target. Submit your code in a file named polyFit.py.
a) Plot the mean squared error (MSE) for the training data and the test data on the same axes
as a function of maximum degree 𝑑. Include axis labels and a legend in your plot. Explain
the trend in the plot.
b) Plot the normalized squared magnitude of the weight vector ‖𝒘‖2/𝑑 on a log scale as
function of 𝑑. Include axis labels and a legend in your plot. Explain the trend in the plot.
c) Create the same two plots using ridge regression (L2 penalty) with regularization strength
𝛼 = 10−6 instead of OLS and compare the results with OLS.
'''

#imports
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

#open csv and put things into np array
testPoly = np.genfromtxt('Homework_Four/testPoly.csv', delimiter=',', skip_header=1)
trainPoly = np.genfromtxt('Homework_Four/trainPoly.csv', delimiter=',', skip_header=1)

print(np.shape(testPoly))
print(np.shape(trainPoly))

testPoly_X = testPoly[:, 0].reshape(-1,1)
testPoly_Y = testPoly[:, 1].reshape(-1,1)

trainPoly_X = trainPoly[:, 0].reshape(-1,1)
trainPoly_Y = trainPoly[:, 1].reshape(-1,1)

#create arrays for test and train mean squared errors 
arr_testmse = []
arr_trainmse = []

#training and testing 
for deg in range(1,9): 
    poly = PolynomialFeatures(deg);
    train_poly = poly.fit_transform(trainPoly_X)
    test_poly = poly.transform(testPoly_X)


    model = LinearRegression(); 
    model.fit(train_poly, trainPoly_Y)

    #prediction step for train 
    train_prediction = model.predict(train_poly)
    train_meansqerr = mean_squared_error(trainPoly_Y, train_prediction)
    arr_trainmse.append(train_meansqerr)

    #prediction step for test 
    test_prediction = model.predict(test_poly)
    test_meansqerr = mean_squared_error(testPoly_Y, test_prediction)
    arr_testmse.append(test_meansqerr)

iterations_array = np.arange(1, 9)

plt.plot(iterations_array, arr_trainmse, label="Training Data Mean Squared Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Training Data Mean Squared Error with OLS")
plt.show()
plt.plot(iterations_array, arr_testmse, label ="Test Data Mean Squared Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Test Data Mean Squared Error with OLS")
plt.show()