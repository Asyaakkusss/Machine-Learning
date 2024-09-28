'''
Train a k-nearest neighbor classifier to the Pima Indians data. Split the data 50/50 into training and
test sets using train_test_split() with random_state=1 to provide a fixed seed. Try out
different approaches for feature scaling and selection to see their effects on the test set classifica-
tion accuracy. Also experiment with different values for the number of neighbors 𝑘. Report the
highest test set accuracy you are able to obtain. Submit your code in a file named knn_pima.py.

Feature scaling approaches: 
I tried out the StandardScaler, the MinMaxScaler, and RobustScaler. Out of the three, the best accuracy I 
was able to obtain was through the StandardScaler at 0.7630208333333334. 

Feature selection approaches: 
I tried out the elastic net, L1, and L2 feature selectors. The highest accuracy I was able to get was from 
L1 and L2 regularization at 0.765625. 

Number of neighbors: 
When it comes to the number of neighbors, the highest test accuracy obtained is 0.7630208333333334 when 40 
of the nearest neighbors are taken into account. 
'''

from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

#get your Pima data 

#turn csv file into np array 
x = np.genfromtxt('/home/asyaakkus/Machine-Learning/pima-indians-diabetes.csv', delimiter=',', skip_header=1)

#features
X = x[:, :-1]

#target values 
y = x[:, -1]

#Splitting data into 50% training and 50% test data and standardizing features 
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.5, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#feature selection section 
lr_en = LogisticRegression(penalty='elasticnet', C=0.1, solver='saga', l1_ratio=0.5, max_iter=10000, random_state=0)

lr_l1 = LogisticRegression(penalty='l1', C=0.1, solver='saga', max_iter=10000, random_state=0)

lr_l2 = LogisticRegression(penalty='l1', C=0.1, solver='saga', max_iter=10000, random_state=0)

selector = SelectFromModel(lr_en)

selector.fit(X_train_std, y_train)

X_train_sel = selector.transform(X_train_std)
X_test_sel = selector.transform(X_test_std)

#making the knn classifier 
knn = KNeighborsClassifier(n_neighbors=40,
p=2,
metric='minkowski')
knn.fit(X_train_sel, y_train)

#check prediction accuracy 
predictions = knn.predict(X_test_sel)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
