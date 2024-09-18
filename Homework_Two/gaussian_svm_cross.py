'''
a) What is the maximum training set accuracy achievable by a linear SVM? Justify your an-
swer!

The maximum training set accuracy that can be achieved by a linear SVM in this situation is 
0.75. This is because no matter where the line is drawn, 2 data points will always be on the wrong
side of it in the best case scenario. There are 8 data points, and 2/8 = 0.25. 1-0.25 = 0.75, so
at best only 75% of the points will be properly placed. 

b) Find a kernel function for an SVM that perfectly separates the two classes and show that it
does so.

The gaussian kernel/rbf perfectly separate them. If you run the script below, you see an accuracy 
of 1.0 when using the rbf kernel. The accuracy is a lot lower when the linear kernel is used, for 
example (around 0.5). 

c) Can a Gaussian kernel SVM perfectly separate the two classes? If so, fit one to the training
data and plot its decision region. Submit your plotted decision region. Also submit your
code in a file named gaussian_svm_cross.py.

TODO (QUESTION 5): figure out part c 
'''
import numpy as np 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt 

#test set: 
data = np.array([[-2,0],
                 [-1,0],
                 [0,1],
                 [0,2],
                 [1,0],
                 [2,0],
                 [0,-1],
                 [0,-2]])
#y set: let x = 1 and o = 0
y_data = np.array([
                 [0],
                 [1],
                 [1],
                 [0],
                 [1],
                 [0],
                 [1],
                 [0],
])

svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=1)


svm.fit(data, y_data)

#%% Check prediction accuracy on test set
predictions = svm.predict(data)
accuracy = accuracy_score(y_data, predictions)
print(accuracy)

#plot decision region 
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

lab = svm.predict(np.c_[xx.ravel(), yy.ravel()])
lab = lab.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, lab, alpha=0.8)
plt.scatter(data[:, 0], data[:, 1], c=y_data, edgecolors='k', marker='o', label='Data points')
plt.title('Decision Boundary of SVM with Gaussian Kernel')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.grid()
plt.show()