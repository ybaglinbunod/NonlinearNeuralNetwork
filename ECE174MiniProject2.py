# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:05:56 2023

@author: ybbfo
"""

# Import section
import numpy as np
import matplotlib.pyplot as plt
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
def levenMarqAlg_ReLU(weights, inputs, outputs, iterations=500):
    current_weights = weights
    identity_matrix = np.identity(weights.shape[0])
    lam = 1e-3 
    beta = 1.1
    alpha = 0.9
    loss_weights = []
    for i in range(iterations):
        current_jacobian = jacobian_relu(current_weights, inputs)
        current_error = calc_error_relu(current_weights, inputs, outputs)
        weights_update = current_weights - np.linalg.inv(current_jacobian.T @ current_jacobian + identity_matrix * lam) @ current_jacobian.T @ current_error
        
        l2_current_error = current_error.T @ current_error
        next_error = calc_error_relu(weights_update, inputs, outputs)
        l2_next_error = next_error.T @ next_error

        if np.all(np.abs(weights_update - current_weights) < 1e-13):
            break

        if l2_next_error < l2_current_error:
            lam = alpha * lam
            current_weights = weights_update
        else:
            lam = beta * lam

        loss_weights.append(calc_loss(current_error, current_weights, lam))

        if i % 50 == 0:
            print(i)

    print("final lambda value:", lam)
    return current_weights, loss_weights

def levenMarqAlg(weights, inputs, outputs, iterations=500, lam=1e-3, alpha=0.9, beta=1.1):
    current_weights = weights
    identity_matrix = np.identity(weights.shape[0])

    # lam, alpha, and beta are now parameters
    loss = []
    for i in range(iterations):
        current_jacobian = jacobian_tanh(current_weights, inputs)
        current_error = calc_error_tanh(current_weights, inputs, outputs)
        weights_update = current_weights - np.linalg.inv(current_jacobian.T @ current_jacobian + identity_matrix * lam) @ current_jacobian.T @ current_error
        
        l2_current_error = current_error.T @ current_error
        next_error = calc_error_tanh(weights_update, inputs, outputs)
        l2_next_error = next_error.T @ next_error

        if np.all(np.abs(weights_update - current_weights) < 1e-7):
            break

        if l2_next_error < l2_current_error:
            lam = alpha * lam
            current_weights = weights_update
        else:
            lam = beta * lam
    
    loss.append(calc_loss(current_error, current_weights, lam))
    return current_weights, loss


def initializeData(N=500, Gamma=1):
    # Generate N random points in R^3 within the range [-Gamma, Gamma] for each component
    x = np.random.uniform(-Gamma, Gamma, (N, 3))

    # Calculating y(n) = x1 * x2 + x3 for each point
    y = x[:, 0] * x[:, 1] + x[:, 2]
    
    return x, y


import numpy as np

def ablation_study(inputs, outputs, param_ranges, eval_function, iterations=500):
    """
    Conducts an ablation study on the levenMarqAlg model with specified hyperparameters.
    
    Parameters:
        inputs: Input data for the model.
        outputs: Output data for the model.
        param_ranges: A dictionary with hyperparameters and their ranges.
        eval_function: A function that evaluates the model's performance.
        iterations: Number of iterations for the levenMarqAlg model (default 500).
        
    Returns:
        A dictionary with hyperparameter combinations and their corresponding performance.
    """
    results = {}
    for lam in param_ranges['lam']:
        for alpha in param_ranges['alpha']:
            for beta in param_ranges['beta']:
                for w in param_ranges['w']:
                    # Run the levenMarqAlg model with the current set of hyperparameters
                    final_weight,loss = levenMarqAlg(w, inputs, outputs, iterations, lam, alpha, beta)
                    # Evaluate the model
                    performance = loss
                    # Record the performance
                    params = (lam, alpha, beta, w)
                    results[params] = performance
    return results

# Example usage
param_ranges = {
    'lam': np.array([1 * (10 ** -i) for i in range(10)]),
    'alpha': np.array([ 0.1*i for i in range(1,10)]),
    'beta': np.array([ 1.1+ 0.5*i for i in range(10)]),
    'w': [np.ones((16,1)), np.zeros((16,1)), np.random.normal(0, .1, (16, 1)), 
          np.random.normal(0, .05, (16, 1)), np.random.normal(0, .2, (16, 1)),
          np.random.normal(0, .5, (16, 1)), np.random.normal(0, 1, (16, 1)),
          np.random.normal(0, 2, (16, 1)), np.random.normal(0, 3, (16, 1)),
          np.random.normal(5, .1, (16, 1))]  # Replace with actual initial weights
}

def your_eval_function(inputs, outputs, final_w):
    # Define how you evaluate the model. For example, calculate and return the error.
    pass

trainX, trainY = initializeData()
# Example call
results = ablation_study(trainX, trainY, param_ranges)

"""
def main():
    Main function to run the program.
    # Initialize N = 500 points
    
    final_loss = []
    w_tanh = np.random.normal(0, .1, (16, 1))
    w_tanh, losses_tanh = levenMarqAlg(w_tanh, trainX, trainY, iterations=1000)

    print(losses_tanh[-5:])
    final_loss.append(losses_tanh[-1])

    print(final_loss)

if __name__ == "__main__":
    main()
"""