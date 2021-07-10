# import modules we want to use
import derivatives as der
import activationFunctions as activate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os,glob

# cost function
def cost_function(T_values,Y_values,lamda,w):
    norm_w1 = (np.sum(np.square(w[0])))
    norm_w2 =(np.sum(np.square(w[1])))
    arr = ( np.log(Y_values) * T_values)
    final_summation = np.sum(arr)
    
    result = final_summation - ( (0.5*lamda) * ( norm_w1 + norm_w2 ) )    # Σ logynk * tnk - λ ||W||^2
    return result

# train neural network
def NeuralNetwork(X, W1, W2, hiddenLayerSize, outLayerSize, activation_h):
    
    x = np.copy(X)
    var1 = x.dot(W1.T)

    array = activate.activate_function_z(activation_h, var1)
    add_one = np.ones((array.shape[0],1))
   
    Z = np.concatenate((add_one,array), axis=1)
    var2 = Z.dot(W2.T)
    y = activate.softmax(var2)
    
    return y, x, Z

# StochasticGradientAscent with 1 hiddenLayer of size M.
# this function returns the cost, and the derivatives of W1,W2
def stochastic_grad_ascent(X, T, W1 , W2, M, Κ, activation_h, lamda):
    
    ynk,x,z = NeuralNetwork(X, W1, W2, M, Κ, activation_h)
    E = cost_function(T, ynk, lamda,[W1,W2])
    grad = der.crossEntropy_softmax_gradient(ynk, T)
    dE_dW1 = der.W1_derivative(grad,W2[:,1:], activation_h ,x, W1, lamda)
    dE_dW2 = der.W2_derivative(grad, z, lamda, W2)
               
    return dE_dW1, dE_dW2, E

# StochasticGradientAscent with miniBatches of size = batch_size ( 100 or 200 ).
# we apply a modification in l_rate as l_rate = l_rate / batch_size
# For each epoch, we shuffle the data for better training.
def mini_batches_SGA(X, T, activation_h, hiddenLayers, classesK, learning_rate, lamda, epochs, batch_size):
    
    train_x = np.copy(X)    # trying not to modify our training data
    
    add_one = np.ones((train_x.shape[0],1))
    train_x = np.concatenate((add_one,train_x), axis=1)     # adding the bias 
    
    X_Y = np.concatenate((train_x,T), axis=1)     # concat X with Y values
     
    N = train_x.shape[0] / batch_size
    
    # weights initialization using Xavier
    lowerW1 = -(np.sqrt(2/train_x.shape[1]))
    upperW1 = (np.sqrt(2/train_x.shape[1]))
    W1 = np.random.uniform(lowerW1,upperW1,(hiddenLayers,train_x.shape[1]))
    lowerW2 = -(np.sqrt(2/hiddenLayers+1))
    upperW2 = (np.sqrt(2/hiddenLayers+1))
    W2 = np.random.uniform(lowerW2, upperW2,(classesK,hiddenLayers+1))
    
    learning_rate = learning_rate / batch_size
    
   
    
    COSTS = []
    for e in range(epochs):
        np.random.shuffle(X_Y) #shuffle our data
        temp_costs = []
        batch = 0
        for i in range(int(N)):
            batch_x = X_Y[batch:batch + batch_size][:,:train_x.shape[1]]     # take each batch
            batch_y = X_Y[batch:batch + batch_size][:,train_x.shape[1]:]
            dW1, dW2, batchError =  stochastic_grad_ascent(batch_x, batch_y, W1, W2, hiddenLayers, classesK, activation_h, lamda)
            batch += batch_size
            W1 += learning_rate*(dW1)
            W2 += learning_rate*(dW2)
            temp_costs.append(batchError)
         
        COSTS.append( (1/batch_size) * np.sum(temp_costs) )     # save current epoch cost from minibatches with (1/m) * Σcost(batch_i)
        

        
    return W1, W2, COSTS     

# Gradcheck function
def gradcheck(X, t, lamda, batches, M, K, activation_h):
    
    train_x = np.copy(X)    # trying not to modify our training data
    
    add_one = np.ones((train_x.shape[0],1))
    train_x = np.concatenate((add_one,train_x), axis=1)     # adding the bias
    
    # weights initialization using Xavier
    lowerW1 = -(np.sqrt(2/train_x.shape[1]))
    upperW1 = (np.sqrt(2/train_x.shape[1]))
    W1 = np.random.uniform(lowerW1,upperW1,(M,train_x.shape[1]))
    lowerW2 = -(np.sqrt(2/M+1))
    upperW2 = (np.sqrt(2/M+1))
    W2 = np.random.uniform(lowerW2, upperW2,(K,M+1))
    
    epsilon = 1e-6

    _list = np.random.randint(train_x.shape[0], size=5)
    x_sample = np.array(train_x[_list, :])
    t_sample = np.array(t[_list, :])
    
    savedW1 = np.copy(W1)
    savedW2 = np.copy(W2)
    
    
    EW1, EW2, Ew = stochastic_grad_ascent(x_sample, t_sample, W1 , W2, M, K, activation_h, lamda)
    numerical_gradw1 = np.zeros(EW1.shape)
    numerical_gradw2 = np.zeros(EW2.shape)
    
    
    for k in range(numerical_gradw1.shape[0]):
        for d in range(numerical_gradw1.shape[1]):
            
           
            w1_tmp = np.copy(savedW1)
            w1_tmp[k, d] += epsilon
            _, _, e_plus = stochastic_grad_ascent(x_sample, t_sample, w1_tmp , savedW2, M, K, activation_h, lamda)

            
            w1_tmp = np.copy(savedW1)
            w1_tmp[k, d] -= epsilon
            _, _, e_minus = stochastic_grad_ascent(x_sample, t_sample, w1_tmp , savedW2, M, K, activation_h, lamda)
            
            # ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numerical_gradw1[k, d] = (e_plus - e_minus) / (2 * epsilon)
    
    for k in range(numerical_gradw2.shape[0]):
        for d in range(numerical_gradw2.shape[1]):
            
            
            w2_tmp = np.copy(savedW2)
            w2_tmp[k, d] += epsilon
            _, _, e_plus = stochastic_grad_ascent(x_sample, t_sample, savedW1 , w2_tmp, M, K, activation_h, lamda)

            
            w2_tmp = np.copy(savedW2)
            w2_tmp[k, d] -= epsilon
            _, _, e_minus = stochastic_grad_ascent(x_sample, t_sample, savedW1 ,w2_tmp, M, K, activation_h, lamda)
            
            # ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numerical_gradw2[k, d] = (e_plus - e_minus) / (2 * epsilon)
            
    return EW1, numerical_gradw1, EW2, numerical_gradw2