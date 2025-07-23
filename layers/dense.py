from layers.layer import Layer
import numpy as np


class Dense(Layer):
   def __init__(self, input_size, output_size):
      #? Generate a matrix of size 3*2 filled with random values from 0 to 1
      self.weights = np.random.randn(output_size, input_size)
      self.bias = np.random.randn(output_size,  1)
   
   def forward(self, input):
      self.input = input
      return np.dot(self.weights, self.input) + self.bias
   
   def backward(self, output_gradient, learning_rate):
      #? here .T is for transposing the matrix (dE/dY) or (dE/dB) -> output gradient . Xt, weight_gradient = dE/dW
      weight_gradient = np.dot(output_gradient, self.input.T)
      self.weights -= learning_rate * weight_gradient
      self.bias -= learning_rate * output_gradient
      #? here we have to return dE/dx = Wt . dE/dY
      return np.dot(self.weights.T, output_gradient)
