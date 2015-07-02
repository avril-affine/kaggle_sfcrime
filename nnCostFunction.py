import numpy as np
from sigmoid import sigmoid

def costFunction(Theta, X, Y, lam):
    """Returns cost of Theta using logistic regression"""

    m = len(X)
    n = len(X[0])
    k = len(Y[0])
    k_h = (n + k) // 2      #average of features and categories
    Theta1 = np.reshape(Theta[0:(n+1)*k_h], (n+1, k_h))
    Theta2 = np.reshape(Theta[(n+1)*k_h:], (k_h+1, k))

    one = np.ones(m)
    one = np.reshape(one, (m, 1))
    a1 = np.concatenate((one, X), axis=1)
   
    #compute inputs to hidden layer
    a2 = sigmoid(np.dot(a1, Theta1))
    a2 = np.concatenate((one, a2), axis=1)

    #compute output layer
    a3 = sigmoid(np.dot(a2, Theta2))

    #compute cost
    J = -(1.0/m) * (np.dot(np.log(a3).T,  Y) + \
        np.dot(np.log(1.0 - a3).T, (1.0 - Y)))
    J = J.sum()

    #compute regularization term
    Theta1_sq = np.dot(Theta1.T, Theta1)
    Theta1_sq[0, :] = np.zeros(k_h)
    Theta2_sq = np.dot(Theta2.T, Theta2)
    Theta2_sq[0, :] = np.zeros(k)
    J = J + (lam / 2.0 / m) * (Theta1_sq.sum() + Theta2_sq.sum())
    print 'cost =', J
    return J
