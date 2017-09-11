
# coding: utf-8

import numpy as np
from numpy import *
import math
import sys

def loadDataSet(filename):
    f = open(filename)
    lines = f.readlines()
    dataSet = []
    labels = []
    for i in lines:
        i = i.strip()
        cols = i.split("\t")

        dataSet.append(list(map(lambda a:float(a), cols[0:-1])))
        labels.append(float(cols[-1]))
    f.close()
    return dataSet, labels

def sigmoid(num):
    return 1.0 / (1+math.exp(-num))

# loss function, forcast => y_hat, real=>labels and y
def loss(forcast, real):
   return -(real * math.log(forcast) + (1-real)*log(1-forcast)) 


def sigmoidOnEle(matrix):
    m,n = shape(matrix)
    arr = []
    for i in range(m):
        rows = []
        for j in range(n):
            sig = sigmoid(matrix[i,j])
            rows.append(sig)
        arr.append(rows)

    return mat(arr)

"""
    alpha: learning rate is 0.001
    maxCycle: loop times is 500
    matrix rule: martix_A's columns equals matrix_B's rows
"""
def linearReg(dataSet, labels):
    dataSetMat = mat(dataSet)
    labelMat = mat(labels).transpose()
    alpha = 0.001
    maxCycles = 1000
    m,n = shape(dataSet)
    weights = zeros((n,1))
    b = 0
    for i in range(maxCycles):
        # this is  Z set()
        z = np.dot(dataSetMat, weights)+b
        # this is A set()
        y_hat = sigmoidOnEle(z)
        error_martrix = y_hat - labelMat
        dz = (1.0 / m) * np.dot(dataSetMat.transpose(),error_martrix)
        weights = weights - alpha * dz
        db = (1.0 /m) * np.sum(error_martrix)
        b = b - alpha * db

    return weights, b


if __name__ == '__main__':

    filename = sys.argv[1]
    dataSet, labels = loadDataSet(filename)
    w,b= linearReg(dataSet, labels)
    print(w)
    print(b)
    
    # some test
    #test_data = mat(dataSet[0:10])
    #test_labels = mat(labels[0:10])

    #y_hat = np.dot(test_data,w) + b 

   
