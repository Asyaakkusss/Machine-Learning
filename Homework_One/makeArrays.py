import numpy as np 
'''
Problem 1

a) Make an array a of size 6 × 4 where every element is a 2.

b) Make an array b of size 6 × 4 that has 3 on the leading diagonal and 1 everywhere else.
(You can do this without loops.)

c) Can you multiply these two matrices together? Why does a * b work, but not
np.dot(a,b)?

d) Compute np.dot(a.transpose(),b) and np.dot(a,b.transpose()). Why are
the results different shapes?

Submit your code for all 4 parts in a file named makeArrays.py. You should be able to solve
parts (a) and (b) using one line of code each.
'''
array_a = np.ndarray(
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2]) #array of 6x4 with each element being a 2

array_b = np.ndarray(
    [3, 1, 1, 1],
    [2, 3, 1, 1],
    [1, 1, 3, 1],
    [1, 1, 1, 3],
    [1, 1, 1, 1],
    [1, 1, 1, 1])