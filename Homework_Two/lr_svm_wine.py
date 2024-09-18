'''
Fit a logistic regression and SVM to the wine data set. Try out different kernels for the SVM,
tuning hyperparameters, such as the Gaussian kernel width, manually. In the same manner as
Homework 1, first pre-process the data by standardizing each attribute, and then split the data
50/50 into training and test sets using train_test_split() with random_state=1 to provide
a fixed seed. Report the highest test set accuracy you are able to obtain for each classifier. Submit
your code in a file named lr_svm_wine.py.

Trying different kernels: 

'poly': The major things that were tuned manually for this kernel were the the degree and coef. The coef
did not change the accuracy much unless it was set to 0. In that case, the model became much less accurate. 
As the degree went up, the accuracy went down. The range with the best accuracy were degrees 1-5. The highest
test set accuracy obtained for the poly classifier was 0.9775280898876404. 

'linear': C was the major thing to tweak in this model. When C was between 0 and 1, the model tended to be less 
accurate. Making it very large did not make much of a difference. The highest test set accuracy obtained for the 
linear classifier was 0.9775280898876404. 

'rbf' (aka Gaussian): A C of one yielded a high accuracy. The highest test set accuracy obtained for the gaussian 
classifier was 1.0 with C = 1. If C was very large or very small (between 0 and 1), the accuracy obtained went down
significantly. Changing gamma from "scale" to "auto" did not change the accuracy much, if at all. 

'sigmoid': A C of one yielded a high accuracy. The highest test set accuracy obtained for the sigmoid classififier was 
0.9887640449438202 with C = 1. If C was very large of very small (between 0 and 1), the accuracy obtained went down
significantly. Changing gamma from "scale" to "auto" did not change the accuracy much, if at all. 
'''

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd 
from sklearn.linear_model import LogisticRegression

#read in csv file in the form of a pandas data frame 
data_frame = pd.read_csv("/home/asyaakkus/Machine-Learning/wine.data.csv")

#separate the x data from the label (or y) data before standardizing 
x_data = data_frame.iloc[:, 1:].values
y_data = data_frame.iloc[:, 0].values
#standardizing: (value - mean)/SD. Use sklearn StandardScalar, since what is being described 
#is z-score standardization 

sc = StandardScaler()

#the standardized x_data we will be using going forward 
std_data = sc.fit_transform(x_data)

#splitting data using train_test_split() w/ random_state = 1
X_train, X_test, y_train, y_test = train_test_split(std_data, y_data, test_size=0.5, random_state = 1)

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

'''Fitting SVM model to the dataset'''
from sklearn.svm import SVC

#linear
#svm = SVC(kernel='linear', C=5, random_state=1)

#poly
#svm = SVC(kernel='poly', degree=5, coef0=1, gamma='scale', random_state=1)

#rbf (Gaussian)
#svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=1)

#sigmoid
svm = SVC(kernel='sigmoid', gamma='scale', C=1, random_state=1)

svm.fit(X_train_std, y_train)

#%% Check prediction accuracy on test set
predictions = svm.predict(X_test_std)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

'''Fitting logistic regression to the dataset'''
log = LogisticRegression(C=10.**10, multi_class='ovr')
log.fit(X_train_std, y_train)
predictions = log.predict(X_test_std)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
