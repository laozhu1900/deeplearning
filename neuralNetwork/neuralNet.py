# coding:utf-8

import numpy as np
from numpy import *
import random
# eg:data: x:[[1,2],[3,4],[5,6]], x1=[1,2],x2=[3,4],x3=[5,6]

def loadDataSet(filename):
    f = open(filename)
    lines = f.readlines()

    dataSet = []
    labels = []
    for i in lines:
        i = i.strip()
        cols = i.split("\t")

        dataSet.append(cols[:-1])
        labels.append(cols[-1])

    return dataSet, labels

def relu(z):
    if z >0:
        return z
    else:
        retrun 0

def reluOnMatEle(matrix):
    m,n = shape(matrix)
    arr = []
    for i in range(m):
        rows = []
        for j in range(n):
            rl = relu(matrix[i,j])
            rows.append(sig)
        arr.append(rows)

    return arr

'''
  rows: echo row match x1's weights
  columns: column's num is hidden-layer's num
'''
def initWeights(rows, columns):
    matrix = np.random.random((rows, columns))
    return matrix

'''
 some functions: z = WT*X+b
                 a = g(z), g(x) is relu function

      | x1 x2 x3 |      | - w1 - |
  X = | x1 x2 x3 |  WT =| - w2 - | 
      | x1 x2 x3 |      | - w3 - |
'''
def neuralNet(dataSet, labels):
    dataSetMat = mat(dataSet).transpose()
    labelMat = mat(labels)
    # m => x1.....xm
    # n => len(w1)
    
    '''
     z_set: to save each layer's z
     forword neural network
     backprop neural network
    '''
    z_set = []
    a_set = []

    m, n = shape(dataSetMat) 
    w_1 = initWeights(m+1,n)
    b_1 = np.random.random((rows, 1))
    
    layer_1_z = np.dot(dataSetMat, w_1.transpose())+b_1
    layer_1_a = reluOnMatEle(layer_1_z)
    z_set.append(layer_1_z)
    a_set.append(layer_1_a)

    w_2 = initWeights(m+1,n)
    b_2 = np.random.random((rows, 1))
    
    layer_2_z = np.dot(layer_1_a, w_2.transpose())+b_2
    layer_2_a = reluOnMatEle(layer_2_z)
    z_set.append(layer_2_z)
    a_set.append(layer_2_a)

    # backprop neural network
    
    dz_2 = layer_2_a - labelMat
    dw_2 = (1.0 / m) * dz_2 * a_set[1].transpose()
    db_2 = (1.0 / m) * np.sum(dz_2,axis=1,keepdims = True)
    
    dz_1 = layer_1_a - labelMat
    dw_1 = (1.0 / m) * dz_1 * a_set[0].transpose()
    db_1 = (1.0 / m) * np.sum(dz_2,axis=1,keepdims = True)
    

    

        






    

