''' @author - ALOJOECEE '''

import numpy as np

#initialize hyperparameters
inputlayersize = 2
hiddenlayersize = 3
outputlayersize = 1
w1 = None #Enter test weights for input to hidden layer
w2 = None #Enter test weights for hidden to output layer
array = np.array(([3, 5], [5, 10], [3, 2]), dtype=float)
array2 = np.array(([75], [90], [56]), dtype=float)
#scale inputs and target outputs
sd1 = array/np.max(array, axis=0)
sd2 = array2/100

#feedforward
def forward(x):
    z1 = np.dot(x, w1) #weighted sum of the input to hidden layer
    def sigmoid(z):
        #compute sigmoid
        return 1/(1+np.exp(-z))
    a1 = sigmoid(z1) #apply non-linear activation function sigmoid
    z2 = np.dot(a1, w2) #weighted sum of hidden to outpur layer
    a2 = sigmoid(z2) 
    return a2 #output