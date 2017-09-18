# coding:utf-8

import numpy as np
from numpy import *
import random
import sys
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
        return 0

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
    matrix = np.random.rand((rows, columns)) * 0.01
    return matrix

'''
 some functions: z = WT*X+b
                 a = g(z), g(x) is relu function

      | x1 x2 x3 |      | - w1 - |
  X = | x1 x2 x3 |  WT =| - w2 - | 
      | x1 x2 x3 |      | - w3 - |
  m => x1.....xm
  n => len(w1)
 
  zList: to save each layer's z
  forword neural network
  backprop neural network

'''


# layer: the layers of neural network,    maxCycles:  max loops for training neural network
def neuralNet(dataSet, labels, layer, maxCycles):
    
    alpha = 0.01

    listA = []
    listZ = []

    #dWList = []
    #dBList = []

    listW = []
    listB = []


    # init echo lays' weight and b
    for i in range(layer):
        listB.append(initWeights(m+1,n))
        listW.append(np.random.random((rows, 1)))


    # dataSet and real values
    dataSetMat = mat(dataSet).transpose()
    labelMat = mat(labels)
    

    # the main function for neural network
    # recursive function for neural network
    def loopNN(dataSetMat, labelMat, loopLayer):

        m,n = shape(dataSetMat)
        
        # forword neural network
        z = np.dot(dataSetMat, w_1.transpose())+b_1
        a = reluOnMatEle(listZ[layer-loopLayer])
        listZ.append(z)
        listA.append(a)
   
   
        # backprop neural network
        dZ = a - labelMat 
        dW = (1.0 / m) * dZ * a.transpose()
        dB =  (1.0 / m) * np.sum(dZ,axis=1,keepdims = True)
        
        listW[layer - loopLayer] = listW[layer - loopLayer] - alpha * dW
        listB[layer - loopLayer] = listB[layer - loopLayer] - alpha * dB

        
        #dWList.append(dW)
        #dBList.append(dB)

        loopLayer -=1

        if(loopLayer>0):

            return loopNN(a, labelMat, loopLayer)
        else:
            return listW, listB

    
    for i in range(maxCycles):

        listW, listB = loopNN(dataSetMat, labelMat, layer)

    return listW, listB


if __name__ == '__main__':
    
    filename = sys.argv[1]
    
    dataSetMat, labelMat = loadDataSet(filename)

    print(neuralNet(dataSetMat, labelMat))


