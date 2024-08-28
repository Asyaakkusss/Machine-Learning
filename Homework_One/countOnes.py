import numpy as np
'''
Problem 2

Write a function that consists of a set of loops that run through a 2-D numPy array and count the
number of ones in it. Do the same thing using the np.where() function (use info(np.where)
to find out how to use it). Name your functions countOnesLoop() and countOnesWhere(),
respectively, and submit your code in a file named countOnes.py.
'''

def countOnesLoop(matrix): 
    ii = np.shape(matrix)[0]
    jj = np.shape(matrix)[1]
    i = 0
    j = 0
    counter = 0; 
    for i in range(ii): 
        for j in range(jj): 
            if (matrix[i][j] == 1):
                counter+=1 
    print(counter)

array_a = np.array(
    [[1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 1]]) 

countOnesLoop(array_a)