# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:15:58 2018

@author: ALOJOECEE
"""
import numpy as np

class NeuralNet():
    def __init__(self, x):
        #record hyperparameters
        self.inputlayersize = 2
        self.hiddenlayersize = 3
        self.outputlayersize = 1
        self.x = x
        #generate weights as arrays
        self.w1 = np.random.randn(self.inputlayersize, self.hiddenlayersize)    
        self.w2 = np.random.randn(self.hiddenlayersize, self.outputlayersize)
    #create sigoid activation function
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    def forward(self):
        #dot product of input array against weights
        self.z1 = np.dot(self.x, self.w1)
        #hidden layer
        self.a1 = self.sigmoid(self.z1) #sigmoid activation function of dot output
        self.z2 = np.dot(self.a1, self.w2)
        #output layer
        self.a2 = self.sigmoid(self.z2)
        return self.a2 #a2 is the Yhat
    
#array of input data
array = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
#convert to standardized unit
std = array/np.max(array, axis=0)
#observe forward propagation
NN = NeuralNet(std)
print(NN.forward())
    
        
        
        
        
        
        
