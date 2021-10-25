# Neural Network

A barebones neural network written in C++ without any dependencies.

## Calculating weights and biases

Run `main.cpp` in C++ to find the weights and biases. Final weights and biases are printed into `weight.txt` and `bias.txt`. Testing data from [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset).

## Digit Recognition

Using the weights and biases from the earlier C++ program, paste the weights and biases to the top of `html/nn2.js`. The website in `html/index.html` will allows users to write digits and the network will predict the written number.

## Possible Improvements

* Use cross-entropy loss function and softmax activation
* Use matrices to calculate multiple inputs at once
* Better random weight initialization

## Additional Links

* [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)