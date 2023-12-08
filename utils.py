# -*- coding: utf-8 -*-
"""
Created on Thu Dec 7 17:37:48 2023

@author: ybbfo
"""

import numpy as np

def relu(input_array):
    """
    Apply the Rectified Linear Unit (ReLU) function element-wise to the input array.

    Parameters:
    input_array (ndarray): Input array or scalar to which the ReLU function is applied.

    Returns:
    ndarray: An array where each element is transformed by the ReLU function.
    """
    return np.maximum(0, input_array)


def relu_derivative(input_array):
    """
    Compute the derivative of the ReLU function element-wise for the input array.

    Parameters:
    input_array (ndarray): Input array for which the derivative of ReLU is computed.

    Returns:
    ndarray: An array where each element is the derivative of ReLU for the corresponding element in the input array.
    """
    return (input_array > 0).astype(float)

def calc_error_relu(weights, inputs, outputs):
    """
    Calculate the error using the ReLU activation function in a simple neural network model.

    Parameters:
    weights (ndarray): Weight matrix of the neural network.
    inputs (ndarray): Input feature matrix.
    outputs (ndarray): Actual output vector.

    Returns:
    ndarray: The error vector computed as the difference between predicted and actual values.
    """
    weights = weights.reshape((16,1))
    outputs = outputs.reshape((inputs.shape[0],1))
    inputs = inputs.reshape((inputs.shape[0],3))
    value = (weights[0] * relu(weights[1]*inputs[:, 0] + weights[2]*inputs[:, 1] + weights[3]*inputs[:, 2] + weights[4]) 
             + weights[5] * relu(weights[6]*inputs[:, 0] + weights[7]*inputs[:, 1] + weights[8]*inputs[:, 2] + weights[9]) 
             + weights[10] * relu(weights[11]*inputs[:, 0] + weights[12]*inputs[:, 1] + weights[13]*inputs[:, 2] + weights[14]) 
             + weights[15])
    value = value.reshape((inputs.shape[0],1))
    error_vector = value - outputs
    return error_vector

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

def gradient_f_w_relu(weights, input_sample):
    """
    Compute the gradient of the network function with respect to weights using ReLU activation.

    Parameters:
    weights (ndarray): Weight vector.
    input_sample (ndarray): Single input data point.

    Returns:
    ndarray: Gradient of the network function with respect to the weights.
    """
    weights = weights.flatten()

    # Intermediate values in the neural network
    z1 = weights[1]*input_sample[0] + weights[2]*input_sample[1] + weights[3]*input_sample[2] + weights[4]
    a1 = relu(z1)
    z2 = weights[6]*input_sample[0] + weights[7]*input_sample[1] + weights[8]*input_sample[2] + weights[9]
    a2 = relu(z2)
    z3 = weights[11]*input_sample[0] + weights[12]*input_sample[1] + weights[13]*input_sample[2] + weights[14]
    a3 = relu(z3)
    # Gradients with respect to each weight
    gradients = np.array([
        a1,
        weights[0] * relu_derivative(z1) * input_sample[0],
        weights[0] * relu_derivative(z1) * input_sample[1],
        weights[0] * relu_derivative(z1) * input_sample[2],
        weights[0] * relu_derivative(z1),
        a2,
        weights[5] * relu_derivative(z2) * input_sample[0],
        weights[5] * relu_derivative(z2) * input_sample[1],
        weights[5] * relu_derivative(z2) * input_sample[2],
        weights[5] * relu_derivative(z2),
        a3,
        weights[10] * relu_derivative(z3) * input_sample[0],
        weights[10] * relu_derivative(z3) * input_sample[1],
        weights[10] * relu_derivative(z3) * input_sample[2],
        weights[10] * relu_derivative(z3),
         # Gradient of bias term is 1
        1.0  
    ])

    return gradients

def jacobian_relu(weights, inputs):
    """
    Compute the Jacobian matrix for a neural network using ReLU activation function.

    Parameters:
    weights (ndarray): Weight vector of the neural network.
    inputs (ndarray): Input feature matrix.

    Returns:
    ndarray: Jacobian matrix.
    """
    jac = np.zeros((len(inputs), len(weights)))
    for i in range(inputs.shape[0]):
        jac[i] = gradient_f_w_relu(weights, inputs[i])
    return jac

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

def gradient_f_w_tanh(weights, input_sample):
    """
    Calculate the gradient of the function f with respect to weights w using tanh activation function.
    
    Parameters:
    weights (numpy.ndarray): The weight vector for the neural network.
    input_sample (numpy.ndarray): A single input sample.

    Returns:
    numpy.ndarray: The gradient of f with respect to w.
    """
    # Calculating the tanh components
    weights = weights.flatten()
    tanh1 = np.tanh(weights[1]*input_sample[0] + weights[2]*input_sample[1] + weights[3]*input_sample[2] + weights[4])
    tanh2 = np.tanh(weights[6]*input_sample[0] + weights[7]*input_sample[1] + weights[8]*input_sample[2] + weights[9])
    tanh3 = np.tanh(weights[11]*input_sample[0] + weights[12]*input_sample[1] + weights[13]*input_sample[2] + weights[14])
    # Derivative of tanh
    dtanh1 = 1 - tanh1**2
    dtanh2 = 1 - tanh2**2
    dtanh3 = 1 - tanh3**2

    # Gradient calculation
    grad = np.array([
        tanh1,
        weights[0] * dtanh1 * input_sample[0],
        weights[0] * dtanh1 * input_sample[1],
        weights[0] * dtanh1 * input_sample[2],
        weights[0] * dtanh1,
        tanh2,
        weights[5] * dtanh2 * input_sample[0],
        weights[5] * dtanh2 * input_sample[1],
        weights[5] * dtanh2 * input_sample[2],
        weights[5] * dtanh2,
        tanh3,
        weights[10] * dtanh3 * input_sample[0],
        weights[10] * dtanh3 * input_sample[1],
        weights[10] * dtanh3 * input_sample[2],
        weights[10] * dtanh3,
        1
    ])

    return grad

def jacobian_tanh(weights, inputs):
    """
    Compute the Jacobian matrix for the neural network using tanh activation function.

    This function calculates the Jacobian matrix of the neural network 
    output with respect to the weights, for all input samples in inputs.

    Parameters:
    weights (numpy.ndarray): The weight vector for the neural network.
    inputs (numpy.ndarray): The input data.

    Returns:
    numpy.ndarray: The Jacobian matrix.
    """
    jac = np.zeros((len(inputs), len(weights)))
    for i in range(inputs.shape[0]):
        jac[i] = gradient_f_w_tanh(weights, inputs[i])
    return jac