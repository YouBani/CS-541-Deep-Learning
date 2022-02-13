import numpy as np
import math
import matplotlib.pyplot as plt


def labels_hot_vector(y):
    hot_vectors = np.zeros((y.size,y.max()+1))
    hot_vectors[np.arange(y.size),y] = 1
    return hot_vectors


def fCE(y, yhat):
    return -np.sum(y*np.log(yhat))/y.shape[0]


def accuracy(y, yhat):
    Y = np.argmax(y, axis=1)
    Yhat = np.argmax(yhat, axis=1)
    return np.mean(Y == Yhat)


def gradfCE(w, XT, y):
    return XT.T.dot(softmax_pre_activation(XT.dot(w)) - y) / len(y)


def softmax_pre_activation(z):
    A = np.exp(z)
    B = np.sum(A, axis=1).reshape(len(z), 1)
    return A/B


def reshape_array(X):
    ones = np.ones(X.shape[0])
    reshapedX = X.T
    X_tilde = np.vstack((reshapedX, ones))
    return X_tilde


def randomize(X, y):
    new_index = np.random.permutation(len(y))
    return X.T[new_index].T, y[new_index]


def SGD(X, y, learning_rate=None, batch_size=None, epochs=10):
    w = 0.01*np.random.randn(X.shape[0]-1, epochs)
    w = np.vstack((w, np.ones((1, epochs))))
    batches = math.ceil(len(y)/batch_size)
    # print(batches)

    mini_batches = []
    losses = []

    # print("w:",w)
    for i in range(epochs):
        X, y = randomize(X, y)
        for j in range(batches):
            start_index = j*batch_size
            end_index = (j*batch_size)+batch_size
            w = w - (learning_rate*gradfCE(w, X.T[start_index:end_index], y[start_index:end_index]))
            yhat = softmax_pre_activation(X.T[start_index:end_index].dot(w))
            loss = fCE(y[start_index:end_index], yhat)
            acc = accuracy(y[start_index:end_index], yhat)
            print("Batch number:", j+1, "| Training Loss (fCE):", loss, "| Accuracy:", acc)
            plt.plot(acc)
            mini_batches.append(j+1)
            losses.append(fCE(y[start_index:end_index], yhat))
    return w, losses


if __name__ == "__main__":
    X_tr = np.load("fashion_mnist_train_images.npy")
    ytr = np.load("fashion_mnist_train_labels.npy")
    X_te = np.load("fashion_mnist_test_images.npy")
    yte = np.load("fashion_mnist_test_labels.npy")

    y_training = labels_hot_vector(ytr)
    y_testing = labels_hot_vector(yte)

    X_tilde_tr = reshape_array(X_tr/255)
    X_tilde_te = reshape_array(X_te/255)

    # LR = [0.01, 0.05, 0.09, 0.1]
    # EPOCHS = [10, 10, 10, 10]
    # BATCHSIZE = [64, 128, 256, 512]
    # ALPHA = [0.000001, 0.000002, 0.000005, 0.00001]
    # lr = epochs = bs = alpha = 0
    # loss = 1000
    # cost = 10000
    # for lr_ in LR:
    #     for epochs_ in EPOCHS:
    #         for bs_ in BATCHSIZE:
    #                 W, COST = SGD(X_tilde_tr, y_training, learning_rate=lr_, batch_size=bs_, epochs=epochs_)
    #
    #                 yhat_training = softmax_pre_activation(X_tilde_tr.T.dot(W))
    #                 # yhat_testing = softmax_pre_activation(X_tilde_te.T.dot(W))
    #
    #                 loss = COST[-1]
    #
    #                 # print("iter: ", iter, ", accuracy: ", accu, ", loss: ", cost[-1])
    #                 # # print("accuracy: ", accu, ", loss: ", COST[-1])
    #                 # iter += 1
    #
    #                 if loss < cost:
    #                     lr = lr_
    #                     epochs = epochs_
    #                     bs = bs_
    #                     cost = loss
    #
    # print("Learning rate: ", lr)
    # print("Number of epochs: ", epochs)
    # print("batche size: ", bs)
    # print("Alpha: ", alpha)

    W, loss = SGD(X_tilde_tr, y_training, learning_rate=0.09, batch_size=64, epochs=10)
    # print(W)
    plt.plot(loss)
    plt.show()	

    yhat_training = softmax_pre_activation(X_tilde_tr.T.dot(W))
    print("Training Accuracy:", accuracy(y_training, yhat_training))

    yhat_testing = softmax_pre_activation(X_tilde_te.T.dot(W))
    print("Testing Accuracy:", accuracy(y_testing,yhat_testing))
    # print("Testing Loss (fCE):",fCE(y_testing, yhat_testing))

