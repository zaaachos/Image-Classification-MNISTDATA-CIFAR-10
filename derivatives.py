import numpy as np


def der_activation_function_h1(a):
    return 1/(1+np.exp(-a))


def der_activation_function_h2(a):
    return 1-np.power(np.tanh(a),2)


def der_activation_function_h3(a):
    return -np.sin(a)


def activate_derivative_h(choice, a):
    if choice == "h1":
        return der_activation_function_h1(a)
    elif choice == "h2":
        return der_activation_function_h2(a)
    elif choice == "h3":
        return der_activation_function_h3(a)


def crossEntropy_softmax_gradient(Y, T):
    result = np.array(T - Y)
    return result


def W1_derivative(vector, W2, dh, X, W1, lamda):
    first = vector.dot(W2)
    second = first*activate_derivative_h(dh , X.dot(W1.T))
    final = second.T.dot(X)
    return final - lamda * W1    # apply the normalization


def W2_derivative(vector, Z, lamda, W2):
    first = (vector.T).dot(Z)
    second = -lamda * W2   # apply the normalization
    return first + second


