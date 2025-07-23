from layers.layer import Layer
import numpy as np

class Activation(Layer):
   #? activation_prime is derivative of activation is a function
   def __init__(self, activation, activation_prime):
      super().__init__()
      self.activation = activation
      self.activation_prime = activation_prime
   
   def forward(self, input):
      self.input = input
      return self.activation(input)
   
   def backward(self, output_gradient, learning_rate):
      return np.multiply(output_gradient, self.activation_prime(self.input))