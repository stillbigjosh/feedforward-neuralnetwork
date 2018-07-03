import numpy as np
import matplotlib.pyplot as plt

#initialize hyperparameters
inputlayersize = 2
hiddenlayersize = 3
outputlayersize = 1
w1 = np.random.randn(inputlayersize, hiddenlayersize)
w2 = np.random.randn(hiddenlayersize, outputlayersize)
array = np.array(([3, 5], [5, 10], [3, 2]), dtype=float)
array2 = np.array(([75], [90], [56]), dtype=float)
#scale inputs and target outputs
sd1 = array/np.max(array, axis=0)
sd2 = array2/100
clist = []

#feedforward
def forward(x):
    global a1, a2, z1, z2
    z1 = np.dot(x, w1)
    def sigmoid(z):
        #compute sigmoid
        return 1/(1+np.exp(-z))
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2)
    a2 = sigmoid(z2)
    return a2
def sigmoidderiv(sig):
    #derivative of sigmoid function
    return sig*(1-sig)
#iteration process for feed forward, backpropagation and subsequent training - The Epoch
for i in range(400):
    print("*Predicted output \n", forward(sd1))
    print("Target output \n", sd2)
    cost = np.mean(np.square(sd2 - forward(sd1)))
    print("Cost function \n", cost)
    error = sd2 - a2
    d1 = error*sigmoidderiv(a2)
    error2 = np.dot(d1, w2.T)
    d2 = error2*sigmoidderiv(a1)
    w1 += np.dot(sd2.T, d2)
    w2 += np.dot(a1.T, d1)
    print("Weights for input layer \n", w1)
    print("Weights for hidden layer \n", w2)
    clist.append(cost)
#gradient descent scatter graph
x = [a for a in range(20)]
y = clist
plt.xlabel("Epoch")
plt.ylabel("Cost function")
plt.title("Training sample")
plt.plot(x, y)
plt.show()













