'''
he Auto-Mpg data set (available in the Files > Data Sets section on Canvas) contains attributes
about a variety of cars, such as the number of cylinders, horsepower, and weight. It also contains
information on the estimated city fuel economy in miles per gallon (mpg).
Your task is to predict the mpg given the other attributes. Split the data 50/50 into training and test
sets using train_test_split() with random_state=1 to provide a fixed seed. Try the fol-
lowing regression models, tuning hyperparameters as necessary:
• Elastic net penalized linear regression with polynomial features.
• Support vector regression (both linear and nonlinear).
• Random forest regression.
• K-nearest neighbors regression.
Report the maximum test set 𝑅2 you are able to obtain with each regression model. Submit your
code as regression_autompg.py.
Hint: The last column of the data set contains the name of the car model, which has no predictive
value. You can ignore the column when loading the data into a numPy array by using the following
line:
np.loadtxt('auto-mpg-missing-data-removed.txt', comments='"')
This causes numPy to ignore all text inside double quotes.
'''