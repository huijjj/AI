import os
import math
from utils import converged, plot_2d, plot_centroids, read_data, \
    load_centroids, write_centroids_tofile
from kmeans import euclidean_distance, assign_data, \
    update_assignment, mean_of_points, update_centroids
import matplotlib.pyplot as plt


def main(data, init_centroids):
    #######################################################
    # You do not need to change anything in this function #
    #######################################################
    centroids = init_centroids
    # plot initial centroids
    plot_centroids(centroids, "init")
    old_centroids = None
    step = 0
    while not converged(centroids, old_centroids):
        # save old centroid
        old_centroids = centroids
        # new assignment
        assignment_dict = update_assignment(data, old_centroids)
        # update centroids
        centroids = update_centroids(assignment_dict)
        step += 1
    print(f"K-means converged after {step} steps.")
    # plot final centroids
    plot_centroids(centroids, "final")
    return centroids


if __name__ == '__main__':
    data, label = read_data("data/mnist.csv")
    init_c = load_centroids("data/mnist_init_centroids.csv")
    final_c = main(data, init_c)
    write_centroids_tofile("mnist_final_centroids.csv", final_c)
