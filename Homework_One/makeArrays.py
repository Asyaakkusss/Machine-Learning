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

#part a 
array_a = np.array(
    [[2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2]]) 

#part b 
array_b = np.array(
    [[3, 1, 1, 1],
    [2, 3, 1, 1],
    [1, 1, 3, 1],
    [1, 1, 1, 3],
    [1, 1, 1, 1],
    [1, 1, 1, 1]]) 

#part c
'''The * operation is element wise multiplication of matrices, while the np.dot operation 
takes the dot product. This is why the former works but the latter does not. In order for
the dot product to be successful, the number of rows in one matrix must equal the number of
columns in the other. In this case, because they are the same dimension, this is not possible
and hence the dot product fails. Element-wise multiplication, on the other hand, relies on both
matrices to be the same dimension and hence the * operation works properly in this case.'''

#part d 
print(np.dot(array_a.transpose(),array_b))
print(np.dot(array_a,array_b.transpose()))

'''For the first option: The array_a becomes a 4x6 matrix and the dot product of a 4x6 and 
6x4 matrix is taken. Hence, the first option makes a 4x4 matrix per the rules of dot products. 
The number of rows in A (4) matches the number of columns in B (4). 
   For the second option: The array_a stays a 6x4 matrix and the dot product of a 6x4 and 
4x6 matrix is taken when the transpose of array_b turns its dimensions into something different.
The number of rows in A (6) matches the number of columns in B (6). Hence, the first option
makes a 6x6 matrix per the rules of dot products.'''