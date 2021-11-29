import os
import csv
import matplotlib.pyplot as plt
import numpy as np


###################################################
# You do not need to change anything in this file #
###################################################


def read_data(fname):
    data = []
    label = []
    with open(fname, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for r in reader:
            label.append(int(float(r[0])))
            data.append(list(map(float, r[1:])))
    return data, label


def load_centroids(fname):
    centroids = dict()
    with open(fname, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for i, r in enumerate(reader):
            centroids[f"centroid{i}"] = list(map(float, r))
    return centroids


def write_centroids_tofile(fname, centroids):
    centroid_values = np.asarray(list(centroids.values()))
    np.savetxt(fname, centroid_values, delimiter=',')


def plot_2d(assignment_dict, centroids):
    fig = plt.figure()
    colors = {"centroid0": "blue", "centroid1": "red"}
    for k in assignment_dict.keys():
        v = np.array(assignment_dict[k])
        plt.scatter(v[:, 0], v[:, 1], marker='o', s=15, c=colors[k])
        plt.scatter(centroids[k][0], centroids[k][1], marker='x', s=100,
                    c=colors[k], label=k)
    plt.xlim(-2, 5)
    plt.ylim(-2, 6)
    plt.legend()
    return fig

def plot_2d_soft(soft_assignment_dict, centroids):
    fig = plt.figure()
    colors = {"centroid0": "firebrick", "centroid1": "blue"}
    x_axis_values = []
    y_axis_values = []
    centroid0_responsibilities = []
    for data_point, r_dict in soft_assignment_dict.items():
        x_axis_values.append(data_point[0])
        y_axis_values.append(data_point[1])
        centroid0_responsibilities.append(r_dict["centroid0"])
    plt.scatter(x_axis_values, y_axis_values, marker='o', s=15, c=centroid0_responsibilities, cmap='coolwarm')
    for centroid, point in centroids.items():
        plt.scatter(point[0], point[1], marker='x', s=100, c=colors[centroid], label=centroid)
        
    plt.xlim(-2, 5)
    plt.ylim(-2, 6)
    plt.legend()
    return fig    

def plot_digit(digit):
    assert len(digit) == 784
    # mnist digits are size 28 x 28
    im = np.array(digit).reshape(28, 28)
    fig = plt.figure()
    plt.imshow(im, cmap='gray')
    return fig


def plot_centroids(centroids, name):
    for k, v in centroids.items():
        fig = plot_digit(v)
        fig.savefig(os.path.join("results", "MNIST", name, f"{k}.png"))


def converged(c1, c2):
    if c1 is None or c2 is None:
        return False
    conv = True
    for key in c1.keys():
        conv = conv and np.allclose(np.array(c1[key]), np.array(c2[key]))
    return conv
