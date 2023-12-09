# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:44:40 2023

@author: ybbfo
"""
import numpy as np
import matplotlib.pyplot as plt


def calc_error_tanh(weights, inputs, outputs):
    """
    Calculate the error between the predicted and actual values using tanh activation function.

    Parameters:
    weights (numpy.ndarray): The weight vector for the neural network.
    inputs (numpy.ndarray): The input data.
    outputs (numpy.ndarray): The actual output data.

    Returns:
    numpy.ndarray: The error between the predicted and actual values.
    """
    weights = weights.reshape((16,1))
    outputs = outputs.reshape((inputs.shape[0],1))
    inputs = inputs.reshape((inputs.shape[0],3))
    value = (weights[0] * np.tanh(weights[1]*inputs[:, 0] + weights[2]*inputs[:, 1] + weights[3]*inputs[:, 2] + weights[4]) 
             + weights[5] * np.tanh(weights[6]*inputs[:, 0] + weights[7]*inputs[:, 1] + weights[8]*inputs[:, 2] + weights[9]) 
             + weights[10] * np.tanh(weights[11]*inputs[:, 0] + weights[12]*inputs[:, 1] + weights[13]*inputs[:, 2] + weights[14]) 
             + weights[15])
    value = value.reshape((inputs.shape[0],1))
    error_vector = value - outputs
    return error_vector

def initializeData(N=500, Gamma=1):
    # Generate N random points in R^3 within the range [-Gamma, Gamma] for each component
    x = np.random.uniform(-Gamma, Gamma, (N, 3))

    # Calculating y(n) = x1 * x2 + x3 for each point
    y = x[:, 0] * x[:, 1] + x[:, 2]
    
    return x, y
def calc_loss(error_vector, weights, lam):
    """
    Calculate the loss including the regularization term.

    Parameters:
    error_vector (ndarray): Error vector.
    weights (ndarray): Weight matrix of the neural network.
    lam (float): Regularization parameter.

    Returns:
    float: Computed loss value.
    """
    loss = np.sum(error_vector**2) + lam * np.sum(weights**2)
    return loss


weights = np.array([[ 2.91746289e+02],
 [-3.68262181e-02],
 [-6.01811213e-02],
 [ 2.16888645e-05],
 [-6.59674911e-01],
 [ 8.89104662e+02],
 [ 1.60334488e-02],
 [ 1.88524179e-04],
 [ 1.12366687e-03],
 [-9.19443364e-03],
 [ 2.82658624e+02],
 [-3.76820643e-02],
 [ 6.11409413e-02],
 [-1.63576764e-05],
 [ 6.58478115e-01],
 [ 1.36539702e+01]])


loss_vs_gamma = []
for i in np.array([1 * i for i in range(1,11)]):
    
    testX,testY = initializeData(N=500,Gamma =i)
    error = calc_error_tanh(weights, testX,testY)
    loss_vs_gamma.append(calc_loss(error, weights, 1e1))

print(loss_vs_gamma)
plt.figure(figsize=(12, 6))
plt.plot(np.array([1 * i for i in range(1,11)]), loss_vs_gamma, marker='o', linestyle='-')
plt.xlabel('gamma')
plt.ylabel('loss for test data')
plt.title('Loss vs Gamma on Test Data')
plt.grid(True)
plt.show()
