# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:05:56 2023

@author: ybbfo
"""

# Import section
import numpy as np


# Function definitions
def initializeData(N = 500):
    mean = 0
    std = 1
    x = np.random.normal(mean, std, (N, 3))

    # Normalizing each point so that max(|x1|, |x2|, |x3|)
    #is brought to 1
    max_value = np.max(x)
    x = x / max_value


    # Calculating y(n) = x1 * x2 + x3 for each point
    y = x[:, 0] * x[:, 1] + x[:, 2]
    
    return x,y


def calc_error(w,x,y):
    #f is predicted, y is actual
    w = w.reshape((16,1))
    y = y.reshape((x.shape[0],1))
    x = x.reshape((x.shape[0],3))
    value = (w[0] * np.tanh(w[1]*x[:, 0] + w[2]*x[:, 1] + w[3]*x[:, 2] + w[4]) 
             + w[5] * np.tanh(w[6]*x[:, 0] + w[7]*x[:, 1] + w[8]*x[:, 2] + w[9]) 
             + w[10] * np.tanh(w[11]*x[:, 0] + w[12]*x[:, 1] + w[13]*x[:, 2] + w[14]) 
             + w[15])
    value = value.reshape((x.shape[0],1))
    r_n = value - y
    return r_n

def gradient_f_w(w, xi):
    """
    Calculate gradient of f_w with respect to w

    """

    # Calculating the tanh components
    tanh1 = float(np.tanh(w[1]*xi[0] + w[2]*xi[1] + w[3]*xi[2] + w[4]))
    tanh2 = float(np.tanh(w[6]*xi[0] + w[7]*xi[1] + w[8]*xi[2] + w[9]))
    tanh3 = float(np.tanh(w[11]*xi[0] + w[12]*xi[1] + w[13]*xi[2] + w[14]))
    # Derivative of tanh
    dtanh1 = 1 - tanh1**2
    dtanh2 = 1 - tanh2**2
    dtanh3 = 1 - tanh3**2

    # Gradient calculation
    grad = np.array([
        tanh1,
        float(w[0] * dtanh1 * xi[0]),
        float(w[0] * dtanh1 * xi[1]),
        float(w[0] * dtanh1 * xi[2]),
        float(w[0] * dtanh1),
        tanh2,
        float(w[5] * dtanh2 * xi[0]),
        float(w[5] * dtanh2 * xi[1]),
        float(w[5] * dtanh2 * xi[2]),
        float(w[5] * dtanh2),
        tanh3,
        float(w[10] * dtanh3 * xi[0]),
        float(w[10] * dtanh3 * xi[1]),
        float(w[10] * dtanh3 * xi[2]),
        float(w[10] * dtanh3),
        1
    ])

    return grad


def jacobian(w,x):
    jac = np.zeros((len(x),len(w)))
    #print(jac.shape)
    #print(gradient_f_w(w, x[0,:]).shape)
    for i in range(x.shape[0]):
        jac[i] = gradient_f_w(w,x[i])
    return jac


def levenMarqAlg(w,x,y, k =500):
    wk = w
    identity = np.identity(w.shape[0])
    lam = 0.000001
    beta = 1.1
    alpha = 0.9
    for i in range(k):
        #print("calculating")
        wk = w
        curJac = jacobian(wk, x)
        r_wk = calc_error(wk, x, y)
        w = wk - np.linalg.inv(curJac.T @ curJac 
            + identity*lam)@curJac.T @r_wk
        #calculate ||r(wk+1)||^2 and ||r(wk)||^2 
        l2rwk = r_wk.T@r_wk
        r_wk_add1 = calc_error(w, x, y)
        l2rwk_add1 = r_wk_add1.T@r_wk_add1
        if l2rwk_add1<l2rwk:
            lam = alpha*lam
            wk = w
            #print("here")
        else:
            #print("there")
            lam = beta*lam
            wk = wk
        #print("new w:", wk!=w)
        if i%100 == 0:
            print(i)
    print("final lambda value:",lam)
    return wk
    
def main():
    """
    Main function to run the program.
    """
    #initialize N = 500 points
    trainX,trainY = initializeData()
    w = np.ones((16,1))
    w = levenMarqAlg(w,trainX,trainY, k =1000)
    print(w)
    z = calc_error(w, trainX, trainY)
    print(np.mean(z))
    testX,testY = initializeData()
    e = calc_error(w, testX, testY)
    print(np.mean(e))
# Conditional to check if script is running as main

if __name__ == "__main__":
    main()
