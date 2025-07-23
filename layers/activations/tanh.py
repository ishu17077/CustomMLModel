from layers.activation import Activation
import numpy as np

class Tanh(Activation):
   def __init__(self):
      #? this is just tangent hyperbolic function
      tanh = lambda x: np.tanh(x)
      #? just derivative of tan x i.e sec^2(x) = 1 - tan^2(x)
      tanh_prime = lambda x: 1 - tanh(x)**2
      super().__init__(tanh, tanh_prime)