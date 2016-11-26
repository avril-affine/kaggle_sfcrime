import numpy as np
from sigmoid import sigmoid

def gradient(Theta, X, Y, lam):
    """Returns the gradient of Theta."""

    m = len(X)
    n = len(X[0])
    k = len(Y[0])
    k_h = (n + k) // 2

    Theta1 = np.reshape(Theta[0:(n+1)*k_h], (n+1, k_h))
    Theta2 = np.reshape(Theta[(n+1)*k_h:], (k_h+1, k))

    delta1 = np.zeros_like(Theta1)
    delta2 = np.zeros_like(Theta2)

    for t in range(m):
        # compute a2, a3
        a1 = np.append([1], X[t,:])         # 1 x n + 1
        a2 = sigmoid(np.dot(a1, Theta1))    # 1 x k_h
        a2 = np.append([1], a2)
        a3 = sigmoid(np.dot(a2, Theta2))    # 1 x k

        # compute deltas
        d3 = a3 - Y[t]                      # 1 x k
        g_z2 = np.dot(a2.T, 1.0 - a2)
        d2 = np.dot(np.dot(d3, Theta2.T), g_z2) # 1 x k_h
        d2 = np.reshape(d2[1:], (1, k_h))

        a1 = np.reshape(a1, (1, n + 1))
        a2 = np.reshape(a2, (1, k_h + 1))
        d3 = np.reshape(d3, (1, k))
        delta1 = delta1 + (1.0 / m) * np.dot(a1.T, d2)
        delta2 = delta2 + (1.0 / m) * np.dot(a2.T, d3)

    # regularize
    reg1 = Theta1
    reg1[0, :] = np.zeros(k_h)
    delta1 = delta1 + (lam / m) * reg1
    reg2 = Theta2
    reg2[0, :] = np.zeros(k)
    delta2 = delta2 + (lam / m) * reg2

    # flatten and concatenate arrays
    delta1 = np.ndarray.flatten(delta1)
    delta2 = np.ndarray.flatten(delta2)
    grad = np.append(delta1, delta2)
    return grad
