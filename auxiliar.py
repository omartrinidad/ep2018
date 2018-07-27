#!/usr/bin/python
# encoding: utf8


import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import tensorflow as tf


def myplot(dataset):
    """
    """
    plt.figure()

    ax = plt.gca()

    plt.scatter(dataset[:,0], dataset[:,1], s=50)
    plt.show()


def draw_vectors2(vectors):
    """
    """
    _, _, U, V = zip(*vectors)
    plt.figure()

    ax = plt.gca()
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])

    plt.scatter(U, V, s=50)
    plt.show()


def draw_vectors(vectors):
    """
    """
    X, Y, U, V = zip(*vectors)
    plt.figure()

    ax = plt.gca()
    ax.quiver(X, Y, U, V,
            angles='xy', scale_units='xy',
            scale=1, color = {'b', 'g', 'r', 'c', 'm', 'y', 'k'}
            )
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])

    plt.draw()
    plt.show()


def random_points(x, y, size, category):
    """
    Generates x random points around another point
    """
    dataset = np.array((x, y)) - np.random.normal(-0.25, 0.25, (size, 2))
    dataset = np.append(dataset, np.full((size, 1), category), axis=1)
    return dataset


def unison_shuffled(a, b):
    """
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return zip(a[p], b[p])


def non_monotonic_function(x, w, theta):
    """
    """
    wx = np.dot(w.T, x) - theta
    return np.cos(wx)


def visualization(X, Y, w, theta, Yhat, epoch, predictions=None):
    """
    """

    fig = plt.figure()
    title = "Epoch {}, Correct predictions: {}".format(epoch, predictions)
    plt.title(title)

    cols = np.where(Y==1, "red", "blue")
    plt.scatter(X[:,0], X[:,1], c = cols, s = 60);

    extra = 0.666
    x = np.linspace(np.amin(X[:,0]) - extra, np.amax(X[:,0]) + extra, 1000)
    y = np.linspace(np.amin(X[:,1]) - extra, np.amax(X[:,1]) + extra, 1000)

    CX, CY = np.meshgrid(x, y)
    zi = non_monotonic_function(
            np.vstack((CX.ravel(), CY.ravel())), w, theta
            ).reshape((1000, 1000))

    cmap = colors.LinearSegmentedColormap.from_list(
            "", ["purple", "white", "green"]
            )

    plt.contourf(
            x, y, zi, alpha=0.444,
            levels=np.linspace(np.amin(zi.ravel()), np.amax(zi.ravel()), 101),
            cmap=cmap, antialiased = True)

    plt.show()


def perceptron(X, Y):
    """
    """
    n_examples = X.shape[0]

    # initialize weights, and theta, and learning rate
    theta = np.random.uniform(low=-0.99, high=0.99)
    w = np.random.uniform(low=-0.99, high=0.99, size=(2))

    eta_w = 0.005
    eta_theta = 0.001

    unit_step = lambda x: 0 if x < 0 else 1

    for epoch in range(30):

        upd_theta = 0
        upd_weight = 0

        # random batch
        for x, y in unison_shuffled(X, Y):

            yhat = non_monotonic_function(x, w, theta)
            dis = yhat - y

            wx = np.dot(w.T, x) - theta
            exp = np.exp(-0.5 * np.square(wx))
            upd_weight += dis * 2 * exp * wx  * x * -1
            upd_theta += dis * 2 * exp * wx

        w = w - eta_w * upd_weight
        theta = theta - eta_theta * upd_theta

        if epoch % 3 == 0:

            Yhat = [non_monotonic_function(X[i], w, theta) for i in range(len(Y))]
            Yhat = np.where(np.array(Yhat) > 0, 1, -1)
            correct_predictions = np.sum(Yhat == Y)

            if correct_predictions == 200:
                visualization(X, Y, w, theta, Yhat, epoch=epoch, predictions=correct_predictions)
                break
            visualization(X, Y, w, theta, Yhat, epoch, predictions=correct_predictions)
