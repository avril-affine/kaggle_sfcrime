from getdata import getData
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs
from nnCostFunction import costFunction
from sigmoid import sigmoid
from normalize import normalize
from nnGradient import gradient
import numpy as np

def main():
    """Main function for sf-crime machine learning
    From training data try to predict the category of crime
    given the date and location.
    """

    again = True

    while again:
        p = float(raw_input('Percent of data to train on: '))
        ran = raw_input('Shuffle data?(y/n) ')
        if ran == 'y' or ran == 'Y':
            ran = True
        else:
            ran = False

        # setup matrices from train.csv file
        out = getData('train.csv', perc=p, rand=ran)
        X = np.array(out['X'])
        Y = out['Y']
        X_test = np.array(out['X_test'])
        Y_test = out['Y_test']
        crimes = out['crimes']

        # calculate mean and standard deviation
        mu = np.mean(X)
        sigma = np.std(X)
        X = normalize(X, mu, sigma)
        X_test = normalize(X_test, mu, sigma)
        
        # get dimensions of matrices 
        m = len(X)
        n = len(X[0])
        k = len(Y[0])
        k_h = (n + k) // 2
        print 'Dimensions: m =', m, 'n =', n, 'k =', k, 'k_h =', k_h

        # randomly initialize Theta
        epsilon = 0.15
        Theta1 = np.random.rand(n, k_h)
        Theta1 = Theta1 * 2 * epsilon - epsilon
        Theta2 = np.random.rand(k_h, k)
        Theta2 = Theta2 * 2 * epsilon - epsilon
        one = np.ones(k_h)
        one = np.reshape(one, (1, k_h))
        Theta1 = np.concatenate((one, Theta1), axis=0)
        one = np.ones(k)
        one = np.reshape(one, (1, k))
        Theta2 = np.concatenate((one, Theta2), axis=0)
        Theta1 = np.ndarray.flatten(Theta1)
        Theta2 = np.ndarray.flatten(Theta2)
        Theta = np.append(Theta1, Theta2)

        # minimize costFunction of Theta
        new_lam = True
        while new_lam:
            lam = float(raw_input('Enter lambda: '))
            xopt = fmin_bfgs(costFunction, Theta, 
                             fprime=gradient, args=(X,Y,lam)
                             )
            Theta1 = np.reshape(xopt[0:(n+1)*k_h], (n + 1, k_h))
            Theta2 = np.reshape(xopt[(n+1)*k_h:], (k_h + 1, k))

            # accuracy against training set
            m = len(X)
            one = np.ones(m)
            one = np.reshape(one, (m, 1))
            a1 = np.concatenate((one, X), axis=1)
            a2 = sigmoid(np.dot(a1, Theta1))
            a2 = np.concatenate((one, a2), axis=1)
            test = sigmoid(np.dot(a2, Theta2))
            correct = 0
            for i in range(len(test)):
                j = np.argmax(test[i])
                if j == np.argmax(Y[i]):
                    correct += 1
            print 'Training set accuracy =', 100.0 * correct / len(test)

            # if there is a test matrix test accuracy of Theta
            if len(X_test) > 0:
                m = len(X_test)
                one = np.ones(m)
                one = np.reshape(one, (m, 1))
                a1 = np.concatenate((one, X_test), axis=1)
                a2 = sigmoid(np.dot(a1, Theta1))
                a2 = np.concatenate((one, a2), axis=1)
                test = sigmoid(np.dot(a2, Theta2))
                correct = 0
                for i in range(len(test)):
                    j = np.argmax(test[i])
                    if j == np.argmax(Y_test[i]):
                        correct += 1
                print 'Test set accuracy =', 100.0 * correct / len(test)
            new_lam = raw_input('Different lambda?(y/n) ')
            if new_lam == 'y' or new_lam == 'Y':
                new_lam = True
            else:
                new_lam = False

        sub = raw_input('Create submission file?(y/n) ')
        if sub == 'y' or sub == 'Y':
            # create predictions for kaggle test data set
            out = getData('test.csv', perc=1.0, test=True)
            X_test = out['X']
            X_test = normalize(X_test, mu, sigma)
            m = len(X_test)
            one = np.ones(m)
            one = np.reshape(one, (m, 1))
            a1 = np.concatenate((one, X_test), axis=1)
            a2 = sigmoid(np.dot(a1, Theta1))
            a2 = np.concatenate((one, a2), axis=1)
            ans = sigmoid(np.dot(a2, Theta2))

            # write to submission csv file
            sub_file = raw_input('Enter submission file name: ')
            f = open(sub_file, 'w')
            header ='Id'
            for c in crimes:
                header += ',' + c
            f.write(header + '\n')
            for i in range(len(ans)):
                f.write(str(i) + ',' + ','.join(map(str, ans[i])) + '\n')
            f.close()

        again = raw_input('Run again? (y/n) ')
        if again == 'y' or again == 'Y':
            again = True
        else:
            again = False

if __name__ == '__main__':
    main()
