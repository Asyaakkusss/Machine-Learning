import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

'''
When applying naïve Bayes to continuous (real-valued) attributes, there are two typical ap-
proaches:
• Discretize the attributes using a histogram and then treat the bins of the histogram as cate-
gories. The continuous data is thus transformed to counts of categories (bins). In the sim-
plest case, we can use 2 bins for each attribute, which corresponds to simply choosing a
threshold for each feature. If an attribute is greater than the threshold, then set it to 1; set it
to 0 otherwise. This is often referred to as Bernoulli naïve Bayes because each feature
follows a Bernoulli distribution.
• Fit a continuous probability distribution directly to the attributes. Typically, the Gaussian
distribution is used for each attribute, so this approach is called Gaussian naïve Bayes.
Fitting a Gaussian distribution involves estimating the mean and variance of the distribu-
tion from the attributes.
Since naïve Bayes assumes that attributes are conditionally independent given the class (target),
applying naïve Bayes involves fitting one distribution (either Bernoulli or Gaussian) per class per
attribute. Both the Bernoulli and Gaussian naïve Bayes classifiers are implemented in the
naive_bayes module in the scikit-learn Python package.
a) Fit a Gaussian naïve Bayes classifier to the wine data set (available in the Files > Data Sets
section on Canvas). First pre-process the data by standardizing, i.e., subtracting the mean
and dividing by the standard deviation for each attribute. Split the data 50/50 into training
and test sets using train_test_split() with random_state=1 to provide a fixed
seed. Report the test set accuracy.

The test set accuracy is 0.9550561797752809.

b) Fit a Bernoulli naïve Bayes classifier to the wine data set using the same pre-processing
and train/test split. For a Bernoulli naïve Bayes classifier, you will have to specify also the
threshold to use, which corresponds to the binarize parameter in the BernoulliNB
class. Report the highest test set accuracy you are able to obtain for different thresholds.
How does it compare to Gaussian naïve Bayes?
Submit your code for both methods in a file named wineNaiveBayes.py.
'''

#read in csv file in the form of a pandas data frame 
data_frame = pd.read_csv("C:/Machine Learning/wine.data.csv")

#separate the x data from the label (or y) data before standardizing 
x_data = data_frame.iloc[:, 1:].values
y_data = data_frame.iloc[:, 0].values
#standardizing: (value - mean)/SD. Use sklearn StandardScalar, since what is being described 
#is z-score standardization 

scaler = StandardScaler()

#the standardized x_data we will be using going forward 
std_data = scaler.fit_transform(x_data)

#splitting data using train_test_split() w/ random_state = 1
x_train, x_test, y_train, y_test = train_test_split(std_data, y_data, test_size=0.5, random_state = 1)

#we now fit the Gaussian naive Bayes classifier to this data 
model = GaussianNB()
model.fit(x_train, y_train)

prediction = model.predict(x_test)

test_accuracy = accuracy_score(y_test, prediction)

print(test_accuracy)