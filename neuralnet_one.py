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
        #global output
        self.a2 = self.sigmoid(self.z2)
        return self.a2 #a2 is the Yhat
        #sigmoid derivativent function
    def sigmoidderiv(self, err):
        return err*(1-err)
    #backpropagation for training data
    def backward(self):
        #error
        self.error = s1 - self.a2
        #error against sigmoid derivative of output
        self.deltaone = self.error*self.sigmoidderiv(self.a2)
        self.error2 = np.dot(self.deltaone, self.w2.T)
        #new weights for hidden layer
        self.w2 += np.dot(self.a1.T, self.deltaone)

        self.deltatwo = self.error2*self.sigmoidderiv(self.a1)
        #new weights for input layer
        self.w1 += np.dot(sw.T, self.deltatwo)
    def train(self):
    	self.forward
    	self.backward
    def weights(self):
    	return self.w1, self.w2																				
 
scores = np.array(([75], [89], [66]), dtype=float)
sw = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
#convert to standardized unit
s1 = sw/np.max(sw, axis=0)
s2 = scores/100

for i in range(100):
	NeuralNet(s1).train()																																	
	cost = str(np.mean(np.square(s2 - NeuralNet(s1).forward())))
	print("* Cost function \n", cost)
	print("Weights \n", NeuralNet(s1).weights())
	print("Predicted output \n", NeuralNet(s1).forward(), "\n Target output \n", s2)
    
    




