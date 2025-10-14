# -*- coding: utf-8 -*-
"""Some plot functions."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def build_distance_matrix(data, mu):
    """builds a distance matrix.

    Args:
        data: numpy array of shape = (N, d). original data.
        mu:   numpy array of shape = (k, d). Each row corresponds to a cluster center.
    Returns:
        squared distances matrix,  numpy array of shape (N, k):
            row number i column j corresponds to the squared distance of datapoint i with cluster center j.
    """
    N, d = data.shape
    k, _ = mu.shape
    distance_matrix = np.zeros((N, k))
    for j in range(k):
        distance_matrix[:, j] = np.sum(np.square(data - mu[j, :]), axis=1)
    return distance_matrix


def plot_cluster(data, mu, colors, ax):
    """
    plot the cluster.

    Note that the dimension of the column vector `colors`
    should be the same as the number of clusters.
    """
    # build distance matrix.
    distance_matrix = build_distance_matrix(data, mu)
    # get the assignments for each point.
    assignments = np.argmin(distance_matrix, axis=1)
    for k_th in range(mu.shape[0]):
        rows = np.where(assignments == k_th)[0]
        data_of_kth_cluster = data[rows, :]
        ax.scatter(
            data_of_kth_cluster[:, 0],
            data_of_kth_cluster[:, 1],
            # works for clusters more than 3
            s=40,
            c=colors[k_th % len(colors)],
        )
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot(data, mu, mu_old):
    """plot."""
    colors = ["red", "blue", "black", "green", "yellow", "purple"]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    plot_cluster(data, mu_old, colors, ax1)
    ax1.scatter(
        mu_old[:, 0], mu_old[:, 1], facecolors="y", edgecolors="y", s=120, marker="^"
    )

    ax2 = fig.add_subplot(1, 2, 2)
    plot_cluster(data, mu, colors, ax2)
    ax2.scatter(mu[:, 0], mu[:, 1], facecolors="y", edgecolors="y", s=120, marker="^")

    plt.tight_layout()
    plt.show() 
    plt.close()


def plot_image_compression(original_image, image, assignments, mu, k):
    """plot histgram."""
    # init the plot
    fig = plt.figure()

    # visualization
    image_reconstruct = mu[assignments].reshape(original_image.shape)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original_image, cmap="Greys_r")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image_reconstruct, cmap="Greys_r")
    plt.tight_layout()
    plt.show()
