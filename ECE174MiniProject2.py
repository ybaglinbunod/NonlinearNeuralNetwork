# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:05:56 2023

@author: ybbfo
"""

# Import section
import numpy as np
import matplotlib.pyplot as plt
from utils import *

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

def levenMarqAlg(weights, inputs, outputs, iterations=500):
    current_weights = weights
    identity_matrix = np.identity(weights.shape[0])
    lam = 1e-3
    beta = 1.1
    alpha = 0.9
    loss_weights = []
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

        loss_weights.append(calc_loss(current_error, current_weights, lam))

        if i % 2500 == 0:
            print(i, end=' ')
    
    print("final lambda value:", lam)
    return current_weights, loss_weights

def main():
    """
    Main function to run the program.
    """
    
    # Initialize N = 500 points
    trainX, trainY = initializeData()
    final_loss = []
    vals = np.arange(0.1, 2.1, 0.1)
    for val in vals:
        w_tanh = np.random.normal(0, val / 5, (16, 1))
        w_tanh, losses_tanh = levenMarqAlg(w_tanh, trainX, trainY, iterations=5000)
        print("standard deviation:", val / 5)
        plot_values(losses_tanh)
        print(losses_tanh[-5:])
        final_loss.append(losses_tanh[-1])

    print(final_loss)

if __name__ == "__main__":
    main()
