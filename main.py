import numpy as np
from layers.dense import Dense
from layers.activations.tanh import Tanh
from loss.mse import mse, mse_prime


# Reshaping to the needs of XOR operator ques and ans
X = np.reshape([[0,0],[1,0], [0,1], [1,1]],(4,2,1))
Y = np.reshape([[0], [1], [1], [0]],(4,1,1))

#?  Network of Layers
network = [
   Dense(2,3),
   Tanh(),
   Dense(3,1),
   Tanh()
]


epochs = 10000
learning_rate = 0.1

for e in range(epochs):
   error = 0
   #? zip() function is used to map x to y: ([0,0],0),....
   for x,y in zip(X,Y):
      output = x
      for layer in network:
         output = layer.forward(output)

      #* Error
      error += mse(y, output)

      #* Backward
      grad = mse_prime(y, output)

      for layer in reversed(network):
         grad = layer.backward(grad, learning_rate)

   # * Error

   error /= len(Y)
   print("%d %d, error = %f" %(e+1, epochs, error))