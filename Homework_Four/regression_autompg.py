'''
The Auto-Mpg data set (available in the Files > Data Sets section on Canvas) contains attributes
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

Answer:

1. Random Forest Regression: 0.7025374494908679
2. K-nearest neighbors regression: 0.43389041538710194
3. SVR Linear: 0.4782529186714335
4. SVR Gaussian: 0.004707850737426522
5. SVR Polynomial: 
6. Elastic Net: 0.5602096449942575

The maximum test R**2 I was able to achieve was random forest overall. 

'''
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score


data = np.loadtxt('Homework_Four/auto-mpg-missing-data-removed.txt', comments='"')
print(np.shape(data))

features = data[:, :7]
targets = data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.5, random_state = 1)

#create polynomial feature transformer for elastic net with polynomial features 
poly = PolynomialFeatures(degree=3) 

x_train_poly = poly.fit_transform(x_train)

x_test_poly = poly.transform(x_test) 


#model = RandomForestRegressor()
#model = KNeighborsRegressor(n_neighbors=100)
#model = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=.001)
#model = SVR(kernel='linear', C=0.001, gamma='auto', epsilon=0.04)
model = ElasticNet(alpha=10, l1_ratio=1)  # Adjust alpha and l1_ratio as needed

model.fit(x_train_poly, y_train)

prediction = model.predict(x_test_poly)

r_squared = r2_score(y_test, prediction)
print(r_squared)