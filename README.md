# Image-Classification-MNISTDATA-CIFAR-10

## Description
Academic project which refers to implementation of Mini-Batch Stochastic Gradient Ascent (SGA). 
The implementation consists of a MultiLayerPerceptron (MLP) with one hidden layer with M hidden units. The Neural Network was trained upon [MNIST](http://yann.lecun.com/exdb/mnist/) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets and it tried to predict the image.

## Dependencies
* Python
* NumPy
* Matplotlib
* cPickle 


## Scores
The MLP was trained with different hyperparameters.
* Learning_rates = = [ 0.01, 0.001]
* λ = [ 0.1, 0.5 ]
* epochs = [ 10, 20, 30]
* HiddenLayers (M) = [100, 200, 300]
* activation_h = [h1. h2. h3]
* batch_size = 200

In MNIST DATA we got accuracy > 85.6% with best the acc = 98.14%.

![MNIST PREDICITONS](https://github.com/zaaachos/Image-Classification-MNISTDATA-CIFAR-10/blob/main/predictions_images/mnist.png)

In CIFAR-10 DATA we got accuracy > 34.5% with best the acc = 45.81%.

![CIFAR-10 PREDICITONS](https://github.com/zaaachos/Image-Classification-MNISTDATA-CIFAR-10/blob/main/predictions_images/cifar-10.png)

You can observe every single run in corresponding Excel file.




