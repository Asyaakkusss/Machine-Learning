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