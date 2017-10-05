
# coding:utf-8
import numpy as np

'''
    how to make a neural network
    some rules
        loss funcation
        activation funcation
        softmax regression

    some hyper-parameters
        W
        B
        L2 regularization
        learning rate


'''
allLayers=3

def loadDataSet(filename):
    f = open(filename)
    lines = f.readlines()

    dataSet = []
    labels = []

    for i in lines:
        i = i.strip()
        cols = i.split('\t')

        dataSet.append(cols[:-1])
        labels.append(cols[-1])

    return np.mat(dataSet), np.mat(labels)



"""

    layers: the layers of neural network
    filename: the dataset for training
"""

def buildModel(layers=3, dataSet, labels, maxCycles=20000):

    # init every layers' weight and B
    model = []
    rows, columns = dataSet.shape
    listA = []
    listZ = []
    for i in range(layers):
        W = np.random.randn(columns, rows) / np.sqrt(columns)
        B = np.zeros(1, columns)
        model.append((W,B))
    
    def forwardPropagation(dataSet, layers):
        
        m, n = dataSet.shape
        (W, B) = model[allLayers - layers]
        # Forward propagation
        Z = dataSet.dot(W) + B
        A = np.tanh(Z)
        listZ.append(Z)
        listA.append(A)
        
        layers -= 1
        if layers > 0:
            return forwardPropagation(Z, layers)
        else:
            expScores = np.exp(Z)
            probs = expScores / np.sum(expScores, axis=1, keepdims=True)
            return probs

    delta = forwardPropagation(dataSet, layers)

    def backPropagation(delta):
        delta[range(rows), y] -= 1



        

        


