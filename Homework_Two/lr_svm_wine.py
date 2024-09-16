'''
Fit a logistic regression and SVM to the wine data set. Try out different kernels for the SVM,
tuning hyperparameters, such as the Gaussian kernel width, manually. In the same manner as
Homework 1, first pre-process the data by standardizing each attribute, and then split the data
50/50 into training and test sets using train_test_split() with random_state=1 to provide
a fixed seed. Report the highest test set accuracy you are able to obtain for each classifier. Submit
your code in a file named lr_svm_wine.py.
'''

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd 


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
print(np.shape(X_train_std))
X_test_std = sc.transform(X_test)

# combine train and test for visualize purpose
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

#%% Fit SVM model
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)


#%% Check prediction accuracy on test set
predictions = svm.predict(X_test_std)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
