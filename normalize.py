import numpy as np

def normalize(X, mu, sigma):
    """Returns a normalized matrix.
    Subtracts average and divides by standard deviation.
    """
    return (X - mu) / sigma
