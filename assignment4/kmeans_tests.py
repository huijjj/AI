from kmeans import euclidean_distance, assign_data, \
    update_assignment, mean_of_points, update_centroids
import numpy as np


# help functions
def assert_dict_eq(dict1, dict2):
    assert type(dict1) is dict
    assert type(dict2) is dict
    # keys
    assert dict1.keys() == dict2.keys()
    # values
    for k, v in dict2.items():
        matrix2 = np.array(v)
        matrix1 = np.array(dict1[k])
        assert np.allclose(np.sort(matrix1, axis=0), np.sort(matrix2, axis=0))


def setup_data_centroids():
    data = [
            [-1.01714716,  0.95954521,  1.20493919,  0.34804443],
            [-1.36639346, -0.38664658, -1.02232584, -1.05902604],
            [1.13659605, -2.47109085, -0.83996912, -0.24579457],
            [-1.48090019, -1.47491857, -0.6221167,  1.79055006],
            [-0.31237952,  0.73762417,  0.39042814, -1.1308523],
            [-0.83095884, -1.73002213, -0.01361636, -0.32652741],
            [-0.78645408,  1.98342914,  0.31944446, -0.41656898],
            [-1.06190687,  0.34481172, -0.70359847, -0.27828666],
            [-2.01157677,  2.93965872,  0.32334723, -0.1659333],
            [-0.56669023, -0.06943413,  1.46053764,  0.01723844]
        ]
    random_centroids = {
            "centroid1": [0.1839742, -0.45809263, -1.91311585, -1.48341843],
            "centroid2": [-0.71767545, 1.2309971, -1.00348728, -0.38204247],
        }
    bad_centroids = {
            "centroid1": [0.1839742, -0.45809263, -1.91311585, -1.48341843],
            "centroid2": [10, 10, 10, 10],
        }
    return data, random_centroids, bad_centroids


# tests begin
def test_eucliean_distance():
    # int
    data1 = [0, 0, 0, 0]
    data2 = [1, 1, 1, 1]
    assert euclidean_distance(data1, data2) == 2

    # negative
    data1 = [-1, -1, -1, -1]
    data2 = [-5, -3, -1, -1]
    assert np.allclose(np.array(euclidean_distance(data1,
                       data2)),
                       np.linalg.norm(np.array(data1) -
                       np.array(data2)).tolist())

    # floats
    data1 = [1.1, 1, 1, 0.5]
    data2 = [4, 3.14, 2, 1]
    assert np.allclose(np.array(euclidean_distance(data1,
                       data2)),
                       np.linalg.norm(np.array(data1) -
                       np.array(data2)).tolist())

    # random
    data1 = np.random.randn(100)
    data2 = np.random.randn(100)
    assert np.allclose(np.array(euclidean_distance(data1.tolist(),
                       data2.tolist())),
                       np.linalg.norm(data1 - data2).tolist())
    print("test_eucliean_distance passed.")


def test_assign_data():
    # set up
    data_empty = [0, 0, 0, 0]
    data_random = [1.1, 5.3, 55, -12.1]
    centroids = {"centroid1": [1, 1, 1, 1],
                 "centroid2": [-10.1, 1, 23.2, 5.099]}
    assert assign_data(data_empty, centroids) == "centroid1"
    assert assign_data(data_random, centroids) == "centroid2"

    data = [10.1, 1, 23.2, 5.099]
    centroids = {"centroid1": [1, 1, 1, 1],
                 "centroid2": [10, 1, 23, 5],
                 "centroid3": [-100, 20.2, 52.9, -37.088]}
    assert assign_data(data, centroids) == "centroid2"
    print("test_assign_data passed.")


def test_update_assignment():
    # set up
    data, random_centroids, bad_centroids = setup_data_centroids()

    # random
    rtn = update_assignment(data, random_centroids)
    answer = {
        "centroid1": [[-1.36639346, -0.38664658, -1.02232584, -1.05902604],
                      [1.13659605, -2.47109085, -0.83996912, -0.24579457],
                      [-0.83095884, -1.73002213, -0.01361636, -0.3265274]],
        "centroid2": [[-1.01714716, 0.95954521, 1.20493919, 0.34804443],
                      [-1.48090019, -1.47491857, -0.6221167, 1.79055006],
                      [-0.31237952, 0.73762417, 0.39042814, -1.1308523],
                      [-0.78645408, 1.98342914, 0.31944446, -0.41656898],
                      [-1.06190687, 0.34481172, -0.70359847, -0.27828666],
                      [-2.01157677, 2.93965872, 0.32334723, -0.1659333],
                      [-0.56669023, -0.06943413, 1.46053764, 0.01723844]]
    }
    assert_dict_eq(rtn, answer)

    # bad
    rtn = update_assignment(data, bad_centroids)
    answer = {
        "centroid1": [[-1.36639346, -0.38664658, -1.02232584, -1.05902604],
                      [1.13659605, -2.47109085, -0.83996912, -0.24579457],
                      [-0.83095884, -1.73002213, -0.01361636, -0.3265274],
                      [-1.01714716, 0.95954521, 1.20493919, 0.34804443],
                      [-1.48090019, -1.47491857, -0.6221167, 1.79055006],
                      [-0.31237952, 0.73762417, 0.39042814, -1.1308523],
                      [-0.78645408, 1.98342914, 0.31944446, -0.41656898],
                      [-1.06190687, 0.34481172, -0.70359847, -0.27828666],
                      [-2.01157677, 2.93965872, 0.32334723, -0.1659333],
                      [-0.56669023, -0.06943413, 1.46053764, 0.01723844]]
    }
    assert_dict_eq(rtn, answer)
    print("test_update_assignment passed.")


def test_mean_of_points():
    # empty
    data = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    assert mean_of_points(data) == [0, 0, 0, 0]

    # random
    data = np.random.randn(10, 4)
    assert np.allclose(np.array(mean_of_points(data.tolist())),
                       data.mean(axis=0))

    # negative
    data = [
            [-1, -10, -70, -89],
            [2, 3, 55, 7],
        ]
    assert np.allclose(np.array(mean_of_points(data)),
                       np.array(data).mean(axis=0))
    print("test_mean_of_points passed.")


def test_update_centroids():
    # set up
    data, random_centroids, bad_centroids = setup_data_centroids()

    # random
    assignment_dict = update_assignment(data, random_centroids)
    answer = {
        'centroid2': [-1.03386497, 0.774388037, 0.33899735, 0.023455955],
        'centroid1': [-0.35358541, -1.529253186, -0.62530377, -0.543782673]
    }
    rtn = update_centroids(assignment_dict)
    assert_dict_eq(rtn, answer)

    # bad
    assignment_dict = update_assignment(data, bad_centroids)
    answer = {
        'centroid1': [-0.82978110, 0.08329567, 0.04970701, -0.146715632]
    }
    rtn = update_centroids(assignment_dict)
    assert_dict_eq(rtn, answer)
    print("test_update_centroids passed.")


if __name__ == '__main__':
    test_eucliean_distance()
    test_assign_data()
    test_update_assignment()
    test_mean_of_points()
    test_update_centroids()
    print("all tests passed.")
