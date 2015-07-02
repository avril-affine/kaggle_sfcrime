import numpy as np

def sigmoid(x):
    h = 1.0 / (1.0 + np.exp(-1.0 * x))
    return h
