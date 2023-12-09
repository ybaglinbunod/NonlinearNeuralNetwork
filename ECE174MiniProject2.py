# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:05:56 2023

@author: ybbfo
"""

# Import section
import numpy as np
import matplotlib.pyplot as plt

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


def levenMarqAlg(weights, inputs, outputs, iterations=500, lam=1e1, alpha=0.1, beta=10):
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

        if np.all(np.abs(weights_update - current_weights) < 1e-4):
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




def ablation_study(inputs, outputs, param_ranges, iterations=50):
    results = {}
    x = 0
    we = []
    for seed_number in param_ranges['seed_number']:
        np.random.seed(seed_number)
        w = np.random.rand(16,1)
        we.append(w)
        for lam in param_ranges['lam']:
        
        

            
            final_weight, loss = levenMarqAlg(w, inputs, outputs, iterations, lam)
            performance = np.mean(loss)
            # Use a simple identifier like w_index instead of the array itself
            params = (lam, seed_number)  
            results[params] = performance
            #print(seed_number, end=' ')
            x+=1
            
        print("seed done",seed_number)
    return results, we
 

param_ranges = {
    'lam': np.array([1 * (10 ** -i) for i in range(10)]),
    'seed_number': np.array([10*i for i in range(11)])
}



import seaborn as sns

def create_heatmap(results, param_ranges):
    # Adjusting the dimensions of the performance matrix
    performance_matrix = np.zeros((len(param_ranges['lam']), len(param_ranges['seed_number'])))

    for i, lam in enumerate(param_ranges['lam']):
        for j, seed_number in enumerate(param_ranges['seed_number']):
            # Assuming 'beta' is a fixed parameter and not affecting the matrix dimensions
            key = (lam, seed_number)
            loss = results.get(key, np.nan)

            # Check if performance is a sequence and take mean if it is
            if isinstance(loss, (list, np.ndarray)):
                loss = np.mean(loss)

            performance_matrix[i, j] = loss

    # Creating the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(performance_matrix, xticklabels=param_ranges['seed_number'], yticklabels=param_ranges['lam'], annot=True, cmap='viridis')
    plt.xlabel('Seed Number')
    plt.ylabel('Lambda')
    plt.title('Loss Heatmap (darker color indicates lower values)')
    plt.show()

# Example usage:
# results = {(1.0, 0): 7.61, (1.0, 10): 6.29, ...}
# param_range = {'lam': [1.0, 0.1, ...], 'seed_number': [0, 10, ...], 'beta': [0.5]}
# create_heatmap(results, param_ranges)


# Create heatmaps for each w configuration

trainX,trainY = initializeData()
results,weights = ablation_study(trainX,trainY, param_ranges,iterations = 1000)

print(min(results))
create_heatmap(results, param_ranges)
min_key = min(results, key=results.get)
index_of_key = results.index(min_key)
print(weights[index_of_key])
print("Key with the minimum value:", min_key)
print("Minimum Value:", results[min_key])

#final_loss = []
#w_tanh = np.ones((16,1))
#w_tanh, losses_tanh = levenMarqAlg(w_tanh, trainX, trainY, iterations=1000)


#PLOT LOSS VS Iterations 
print("loss vs iterations")
w = weights[index_of_key]
w_tanh, losses_tanh = levenMarqAlg(weights, trainX, trainY, iterations=1000, lam=1e1, alpha=0.1, beta=10)

# Plotting
iterations = range(len(losses_tanh))
testX,testY = initializeData(N=100)
print(w_tanh)
print(len(losses_tanh))
plt.figure(figsize=(12, 6))
plt.plot(iterations, losses_tanh, marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss per Iteration')
plt.grid(True)
plt.show()



#Plot the loss vs different gammas
loss_vs_gamma = []
for i in np.array([.2 * i for i in range(1,100,2)]):
    
    testX,testY = initializeData(N=500,Gamma =i)
    error = calc_error_tanh(weights, testX,testY)
    loss_vs_gamma.append(calc_loss(error, weights, 1e-4))

plt.figure(figsize=(12, 6))
plt.plot(np.array([.2 * i for i in range(1,100,2)]), loss_vs_gamma, marker='o', linestyle='-')
plt.xlabel('gamma')
plt.ylabel('loss for test data')
plt.title('Loss vs Gamma on Test Data')
plt.grid(True)
plt.show()
#for j in np.array([1*10**-i for i in range(10)]):
    

"""
j = 1e-4
loss_vs_gamma = []
w_tanh = np.random.rand(16,1)
w_tanh, losses_tanh = levenMarqAlg(w_tanh, trainX, trainY, iterations=1000, lam = j)
#print("1")
print("wee",w_tanh)
for i in np.array([1 * i for i in range(1,11)]):
    testX,testY = initializeData(N=500,Gamma =i)
    error = calc_error_tanh(w_tanh, testX,testY)
    
    #print("wee",w_tanh)
    loss_vs_gamma.append(calc_loss(error, w_tanh, j)/100)
    #loss_vs_gamma.append(mean_absolute_error(error))
print(loss_vs_gamma[0])
plt.figure(figsize=(12, 6))
plt.plot(np.array([1 * i for i in range(1,11)]), loss_vs_gamma, marker='o', linestyle='-')
plt.xlabel('gamma')
plt.ylabel('loss for test data')
plt.title('Loss vs Gamma on Test Data, Lambda = {}'.format(j))
plt.grid(True)
plt.show()
"""
print("end")
   
    
    
    
    
    