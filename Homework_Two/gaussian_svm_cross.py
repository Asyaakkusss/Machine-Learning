'''
a) What is the maximum training set accuracy achievable by a linear SVM? Justify your an-
swer!

The maximum training set accuracy that can be achieved by a linear SVM in this situation is 
0.75. This is because no matter where the line is drawn, 2 data points will always be on the wrong
side of it in the best case scenario. There are 8 data points, and 2/8 = 0.25. 1-0.25 = 0.75, so
at best only 75% of the points will be properly placed. 

b) Find a kernel function for an SVM that perfectly separates the two classes and show that it
does so.
c) Can a Gaussian kernel SVM perfectly separate the two classes? If so, fit one to the training
data and plot its decision region. Submit your plotted decision region. Also submit your
code in a file named gaussian_svm_cross.py.
'''

