from getdata import getData
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs
from costFunction import costFunction
from normalize import normalize
from gradient import gradient
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
        print 'Dimensions: m = ', m, ' n = ', n, ' k = ', k

        # create Theta
        Theta = np.zeros((n + 1, k))
        one = np.ones(k)
        Theta[0,:] = one
        one = np.ones(m)
        one = np.reshape(one, (m, 1))
        X = np.concatenate((one, X), axis=1)

        # minimize costFunction of Theta
        new_lam = True
        while new_lam:
            lam = float(raw_input('Enter lambda: '))
            xopt = fmin_bfgs(costFunction, Theta, 
                             fprime=gradient, args=(X,Y,lam)
                             )
            xopt = np.reshape(xopt, (n + 1, k))

            # accuracy against training set
            test = 1.0 / (1.0 + np.exp(np.dot(-X, xopt)))
            correct = 0
            for i in range(len(test)):
                j = np.argmax(test[i])
                if j == np.argmax(Y[i]):
                    correct += 1
            print 'Training set accuracy =', 100.0 * correct / len(test)

            # if there is a test matrix test accuracy of Theta
            if len(X_test) > 0:
                one = np.ones(len(X_test))
                one = np.reshape(one, (len(X_test), 1))
                X_t = np.concatenate((one, X_test), axis=1)
                test = 1.0 / (1.0 + np.exp(np.dot(-X_t, xopt)))
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

        ans = raw_input('Create submission file?(y/n) ')
        if ans == 'y' or ans == 'Y':
            # create predictions for kaggle test data set
            out = getData('test.csv', perc=1.0, test=True)
            X_test = out['X']
            X_test = normalize(X_test, mu, sigma)
            one = np.ones(len(X_test))
            one = np.reshape(one, (len(X_test), 1))
            X_test = np.concatenate((one, X_test), axis=1)

            # write to submission csv file
            ans = (1.0 / (1.0 + np.exp(np.dot(-X_test, xopt))))
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
