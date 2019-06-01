import numpy as np
import forward_propagation as fp
import warnings

def linear_backward(dZ, cache):
    """
    Implements the linear part of the backward propagation process for a single layer
    :param dZ: the gradient of the cost with respect to the linear output of the current layer
    :param cache: tuple of values (A_prev, W, b) coming from the forward propagation
    :return:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1),
    same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache['A'],cache['W'],cache['B']
    m = A_prev.shape[0] 
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ) 
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer. The function
    first computes dZ and then applies the linear_backward function.
    :param dA: post activation gradient of the current layer
    :param cache: – contains both the linear cache and the activations cache
    :param activation:
    :return:
    dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1),
    same shape as A_prev
    dW – Gradient of the cost with respect to W (current layer l), same shape as W
    db – Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache['linear_cache'],cache['activation_cache']
    if activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
    else:
        dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ,linear_cache)
    return dA_prev, dW, db


def relu_backward (dA, activation_cache):
    """
    Implements backward propagation for a ReLU unit
    :param dA: the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)
    :return: dZ – gradient of the cost with respect to Z
    """
    Z = activation_cache
    dA[Z <= 0] = 0
    return dA

# =============================================================================
# fix softmax - need to remove
# =============================================================================
def sigmoid_backward (dA, activation_cache):
    """
    Implements backward propagation for a sigmoid unit
    :param dA: the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)
    :return: dZ – gradient of the cost with respect to Z
    """
    Z = activation_cache
    sig, _ = fp.sigmoid(Z)
    return (dA * (sig * (1 - sig)))


def softmax_backward (dA, activation_cache):
    """
    Implements backward propagation for a softmax unit
    :param dA: the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)
    :return: dZ – gradient of the cost with respect to Z
    """
    Z = activation_cache
    soft,_=fp.softmax(Z)
    return soft-dA



def L_model_backward(AL, Y, caches,dropout_keep_prob=1):
    """
    Implement the backward propagation process for the entire network.
    :param AL:  - the probabilities vector, the output of the forward propagation (L_model_forward)
    :param Y: the true labels vector
    :param caches: - list of caches containing for each layer: a) the linear cache; b) the activation cache
    :return: Grads - a dictionary with the gradients
    grads["dA" + str(l)] = ...
    grads["dW" + str(l)] = ...
    grads["db" + str(l)] = ...
    """
    grads = dict()
    cur_layer = len(caches)-1
    Y = Y.T

    dA_prev, dW, db = linear_activation_backward(Y, caches[cur_layer],"softmax")
    
    # **** dropout ****
    if dropout_keep_prob!=1:
        dA_prev=dropout_backward(dA_prev, caches[cur_layer-1]['D'], dropout_keep_prob)
    # **** dropout ****
    
    grads["dA" + str(cur_layer)] = dA_prev 
    grads["dW" + str(cur_layer+1)] = dW 
    grads["db" + str(cur_layer+1)] = db 
    for cur_layer in range(cur_layer-1,-1,-1): 
        dA_prev, dW, db = linear_activation_backward(dA_prev, caches[cur_layer], 'relu')
        
        # **** dropout ****
        if dropout_keep_prob!=1 and cur_layer!=0:
            dA_prev=dropout_backward(dA_prev, caches[cur_layer-1]['D'], dropout_keep_prob)
        # **** dropout ****
        grads["dA" + str(cur_layer)] = dA_prev
        grads["dW" + str(cur_layer+1)] = dW
        grads["db" + str(cur_layer+1)] = db
    return grads





def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent
    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :param grads: a python dictionary containing the gradients (generated by L_model_backward)
    :param learning_rate: the learning rate used to update the parameters (the “alpha”)
    :return: parameters – the updated values of the parameters object provided as input
    """
    
    for idx , value in parameters.items():
        if idx == 0:
            continue
        W, b = parameters[idx]
        W = W - learning_rate*grads["dW"+str(idx)]
        b = b - learning_rate*grads["db"+str(idx)]
        parameters[idx] = (W,b)
    return parameters

# =============================================================================
#  Bonus - dropout   
# =============================================================================
    
def dropout_backward(dA, activation_cache, keep_prob):
    D=activation_cache
    dA = np.multiply(D, dA) 
    dA = dA/keep_prob
    return dA






