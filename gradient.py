import numpy as np

def gradient(Theta, X, Y, lam):
    """Returns the gradient of Theta."""

    m = len(X)
    n = len(X[0])
    k = len(Y[0])
    Theta = np.reshape(Theta, (n, k))
    h = 1.0 / (1.0 + np.exp(np.dot(-1.0 * X, Theta)))
    grad = (1.0/m) * np.dot(X.T, (h - Y))
    reg = (lam / m) * Theta
    reg[0, :] = np.zeros(k)
    grad = grad + reg
    grad = np.ndarray.flatten(grad)
    return grad
