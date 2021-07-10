import numpy as np


def activation_function_h1(a):
    return np.log(1 + np.exp(a))


def activation_function_h2(a):
    return np.tanh(a)


def activation_function_h3(a):
    return np.cos(a)


def activate_function_z(user_choice, number):
    if user_choice == "h1":
        return activation_function_h1(number)
    elif user_choice == "h2":
        return activation_function_h2(number)
    elif user_choice == "h3":
        return activation_function_h3(number)

    
def softmax( x, ax=1 ):

    m = np.max( x, axis=ax, keepdims=True )
    p = np.exp( x - m )
    return ( p / np.sum(p,axis=ax,keepdims=True) )
