import numpy as np

def mse(y_true, y_pred):
   #? y_pred and y_true are arraylike and substracted then squared then get mean error
   return np.mean(np.power(y_pred - y_true, 2))

def mse_prime(y_true, y_pred):
   return 2 * (y_pred - y_true)/ np.size(y_true)
