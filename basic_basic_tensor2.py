# encoding: utf8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def data_matrix(x):
    """
    Add the bias term
    """
    n = len(x)
    b = np.ones((n, 1))
    return np.hstack((x, b))


def plot_anscombe(dataset, slope, offset):
    """
    Plot Anscombe
    """
    x = dataset[:,0:1]
    y = dataset[:,1:2]
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    abline_values = [slope * i + offset for i in range(0, 22)]
    ax.plot(range(0, 22), abline_values)
    ax.plot( x, y, 'ro', lw=2)
    plt.show()


X = np.genfromtxt("data/anscombes.csv", dtype=float, delimiter=',')
X = X[1:,2:]
AX, BX, CX, DX = X[0:11,:], X[11:22,:], X[22:33,:], X[33:44,:] 

x = AX[:,0:1]
XX = data_matrix(x)
yy = AX[:,1:2]

# multiply two matrices
X = tf.placeholder(
        tf.float32,
        shape = AX.shape,
        name = None
        )

y = tf.placeholder(
        tf.float32,
        shape = yy.shape,
        name = None
        )

# Here, the lsq solution
result = tf.matmul(
            tf.matmul(
                tf.matrix_inverse(
                    tf.matmul(
                        tf.transpose(XX),
                        XX
                    )
                ),
                tf.transpose(XX),
            ), 
            yy
        )


with tf.Session() as session:
    feed_dict = {X: XX, y: yy}
    ww = session.run(result, feed_dict = feed_dict)

plot_anscombe(AX, ww[0, 0], ww[1, 0])
