import forward_propagation as fp
import backward_propagation as bp
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import time

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size=1,use_batchnorm=False,param=None,dropout_keep_prob=1):
    """
    Implements a L-layer neural network. All layers but the last should have the ReLU
    activation function, and the final layer will apply the sigmoid activation function. The
    size of the output layer should be equal to the number of labels in the data. Please
    select a batch size that enables your code to run well (i.e. no memory overflows while
    still running relatively fast)
    :param X: the input data, a numpy array of shape (width(no_features),height(no_examples))
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param layers_dims: a list containing the dimensions of each layer, including the input
    :param learning_rate:
    :param num_iterations:
    :param batch_size: the number of examples in a single training batch.
    :return:
    parameters - the parameters learnt by the system during the training
    costs - the values of the cost function (calculated by the compute_cost function)
    One value is to be saved after each 100 training iterations

    Hint: initialize -> L_model_forward -> compute_cost -> L_model_backward -> update parameters
    """
    costs = []
    if not param:
        parameters = fp.initliaize_parameters(layers_dims)
    else:
        parameters = param
        
    for idx in range (0, num_iterations):
#        if batch_size:
        AL_full = np.zeros(Y.shape)
        num_batches = int(len(Y)/batch_size)
        for ii in range(0,num_batches):
            cur_batch_idx = ii*batch_size
            if cur_batch_idx+batch_size > len(Y):
                batch_X = X[:,cur_batch_idx:len(Y)]
                batch_Y = Y[cur_batch_idx:len(Y)]
            else:
                batch_X = X[:,cur_batch_idx:cur_batch_idx+batch_size]
                batch_Y = Y[cur_batch_idx:cur_batch_idx+batch_size]
            
            AL, caches = fp.L_model_forward(batch_X, parameters, use_batchnorm,dropout_keep_prob)
            grads = bp.L_model_backward(AL, batch_Y, caches,dropout_keep_prob)
            parameters = bp.update_parameters(parameters, grads, learning_rate)
            AL_full[cur_batch_idx:cur_batch_idx+batch_size] = AL.T
            
        cost = fp.compute_cost(AL_full.T, Y)
        print(idx)
        if (idx+1) % 100 == 0:  # One value per 100 iterations
            costs.append(cost)
            print(cost)
    return parameters, costs





def predict(X, Y, parameters):
    """
    The function receives an input data and the true labels and calculates the accuracy of
    the trained neural network on the data.
    :param X: the input data, a numpy array of shape (height*width, number_of_examples)
    :param Y:the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :return:
    :param accuracy - the accuracy measure of the neural net on the provided data
    """
    m = Y.shape[0]
    A, caches = fp.L_model_forward(X.T, parameters)
    
    predicted = np.argmax(A.T,axis=1).astype(int)
    acc = np.sum((predicted.T == Y.argmax(axis=1)) / m)
    print("Accuracy: " + str(acc))
    return acc


