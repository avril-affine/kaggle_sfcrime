import numpy as np

def costFunction(Theta, X, Y, lam):
    """Returns cost of Theta using logistic regression"""

    m = len(X)
    n = len(X[0])
    k = len(Y[0])
    Theta = np.reshape(Theta, (n, k))
    h = 1.0 / (1.0 + np.exp(np.dot(-1.0 * X, Theta)))
    J = -(1.0/m) * (np.dot(np.log(h).T,  Y) + \
        np.dot(np.log(1.0 - h).T, (1.0 - Y)))
    Theta2 = np.dot(Theta.T,Theta)
    Theta2[0, :] = np.zeros(k)
    J = J + (lam / 2.0 / m) * Theta2
    J = J.sum()
    print 'cost =', J
    return J
