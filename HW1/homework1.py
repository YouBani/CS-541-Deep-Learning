import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson


def problem_1a(A, B):
    return A + B


def problem_1b(A, B, C):
    return np.dot(A, B) - C


def problem_1c(A, B, C):
    return (A * B) + C.T


def problem_1d(x, y):
    return np.inner(x, y)


def problem_1e(A, x):
    return np.linalg.solve(A, x)


def problem_1f(A, x):
    return (np.linalg.solve(A.T, x.T)).T


def problem_1g(A, i):
    return np.sum(A[i, ::2])


def problem_1h(A, c, d):
    return np.mean(a[np.nonzero((A >= c) * (A <= d))])


def problem_1i(C, k):
    eigenValue, eigenVector = np.linalg.eig(C)
    return eigenVector[:, np.argsort(np.abs(eigenValue))[-k:]]


def problem_1j(x, k, m, s):
    return np.random.multivariate_normal(x+m, np.eye(x.shape[0]) * s, k).T


def problem_1k(A):
    return A[np.random.permutation(A.shape[0]), :]


def problem_1l(x):
    return x - np.mean(x) / np.std(x)


def problem_1m(x, k):
    return np.repeat(np.atleast_2d(x), k, axis=0)


def problem_1n(X):
    Z = np.repeat(np.atleast_3d(X), X.shape[1], axis=2)
    return ((np.swapaxes(Z, 1, 2) - Z)**2).sum(axis=0)**0.5


def linear_regression(X_tr, y_tr):
    return np.linalg.solve(np.matmul(X_tr.T, X_tr), np.matmul(X_tr.T, y_tr))


def fMSE(y_hat, y):
    return np.mean(np.square(y_hat-y))


def train_age_regressor():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    train = fMSE(np.matmul(X_tr, w), ytr)
    test = fMSE(np.matmul(X_te, w), yte)

    return train, test


print(train_age_regressor())


# dataload = np.load("PoissonX.npy")
# plt.hist(dataload,density="true")
# plt.title("Histogram of the given PoissonX Data")
# plt.show()
# mu = 3.7  #Change rate paramters here to 2.5, 3.1, 3.7, and 4.3
# res = poisson(mu)
# plt.plot(dataload, poisson.pmf(dataload, mu), 'bo', ms=8, label='poisson pmf')
# plt.vlines(dataload, 0, poisson.pmf(dataload, mu), colors='b', lw=5, alpha=0.5)
# plt.vlines(dataload, 0, res.pmf(dataload), colors='k', linestyles='-', lw=1, label='frozen pmf')
# plt.legend(loc='best', frameon=False)
# plt.title("Probability Density Distribution for Mu=3.7")
# plt.show()
