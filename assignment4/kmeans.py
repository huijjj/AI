import os
import math
from utils import converged, plot_2d, plot_centroids, read_data, \
    load_centroids, write_centroids_tofile
import matplotlib.pyplot as plt


# problem for students
def euclidean_distance(dp1, dp2):
    """Calculate the Euclidean distance between two data points.

    Arguments:
        dp1: a list of floats representing a data point
        dp2: a list of floats representing a data point

    Returns: the Euclidean distance between two data points
    """
    ret = 0
    for i in range(len(dp1)):
        ret += (dp1[i] - dp2[i]) ** 2

    return ret ** (1 / 2)        


# problem for students
def assign_data(data_point, centroids):
    """Assign a single data point to the closest centroid. You should use
    the euclidean_distance function (that you previously implemented).

    Arguments:
        data_point: a list of floats representing a data point
        centroids: a dictionary representing the centroids where the keys are
                   strings (centroid names) and the values are lists of
                   centroid locations

    Returns: a string as the key name of the closest centroid to the data point
    """
    ret = ""
    minimum = float("inf")
    for key, val in centroids.items():
        if(euclidean_distance(data_point, val) < minimum):
            ret = key
            minimum = euclidean_distance(data_point, val)
    
    return ret


# problem for students
def update_assignment(data, centroids):
    """Assign all data points to the closest centroids. You should use
    the assign_data function (that you previously implemented).

    Arguments:
        data: a list of lists representing all data points
        centroids: a dictionary representing the centroids where the keys are
                   strings (centroid names) and the values are lists of
                   centroid locations

    Returns: a new dictionary whose keys are the centroids' key names and
             values are lists of points that belong to the centroid. If a
             given centroid does not have any data points closest to it,
             do not include the centroid in the returned dictionary.
    """
    ret = {}
    
    for data_point in data:
        centroid = assign_data(data_point, centroids)
        if centroid in ret:
            ret[centroid].append(data_point)
        else:
            ret[centroid] = [data_point]
    
    return ret
            

# problem for students
def mean_of_points(data):
    """Calculate the mean of a given group of data points. You should NOT
    hard-code the dimensionality of the data points).

    Arguments:
        data: a list of lists representing a group of data points

    Returns: a list of floats as the mean of the given data points
    """
    dim = len(data[0])
    ret = [0.0 for _ in range(dim)]
    
    for data_point in data:
        for i in range(dim):
            ret[i] += data_point[i]
    
    for i in range(dim):
        ret[i] /= len(data)

    return ret

# problem for students
def update_centroids(assignment_dict):
    """Update centroid locations as the mean of all data points that belong
    to the cluster. You should use the mean_of_points function (that you
    previously implemented).

    Arguments:
        assignment_dict: the dictionary returned by update_assignment function

    Returns: A new dictionary representing the updated centroids
    """
    ret = {}
    for key, val in assignment_dict.items():
        ret[key] = mean_of_points(val)

    return ret

def main(data, init_centroids):
    #######################################################
    # You do not need to change anything in this function #
    #######################################################
    centroids = init_centroids
    old_centroids = None
    step = 0
    while not converged(centroids, old_centroids):
        # save old centroid
        old_centroids = centroids
        # new assignment
        assignment_dict = update_assignment(data, old_centroids)
        # update centroids
        centroids = update_centroids(assignment_dict)
        # plot centroid
        fig = plot_2d(assignment_dict, centroids)
        plt.title(f"step{step}")
        fig.savefig(os.path.join("results", "2D", f"step{step}.png"))
        plt.clf()
        step += 1
    print(f"K-means converged after {step} steps.")
    return centroids


if __name__ == '__main__':
    data, label = read_data("data/data_2d.csv")
    init_c = load_centroids("data/2d_init_centroids.csv")
    final_c = main(data, init_c)
    write_centroids_tofile("2d_final_centroids.csv", final_c)
