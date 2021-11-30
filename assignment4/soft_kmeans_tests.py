from soft_kmeans import get_responsibility, \
    update_soft_assignment, update_centroids
import numpy as np


# help functions
def assert_cpl_dict_eq(dict1, dict2):
    assert type(dict1) is dict
    assert type(dict2) is dict
    # keys
    assert dict1.keys() == dict2.keys()
    # values
    for k, v in dict2.items():
        value_dict2 = v
        value_dict1 = dict1[k]
        assert_dict_eq(value_dict1, value_dict2)
        
def assert_dict_eq(dict1, dict2):
    # keys
    assert dict1.keys() == dict2.keys()
    # values
    for k, v in dict2.items():
        value2 = v
        value1 = dict2[k]
        assert round(value1, 2)==round(value2, 2)
        
def assert_list_dict_eq(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    for k, v in dict2.items():
        value2 = v
        value1 = dict2[k]
        assert [round(x, 2) for x in value1] == [round(x, 2) for x in value2]



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


def test_get_responsibility():
    # set up
    beta = 0.1
    data_empty = [0, 0, 0, 0]
    data_random = [1.1, 5.3, 55, -12.1]
    centroids = {"centroid1": [1, 1, 1, 1],
                 "centroid2": [-10.1, 1, 23.2, 5.099]}
    
    r_dict = get_responsibility(data_empty, centroids, beta)
    answer = {'centroid1': 0.9155308657289222, 
              'centroid2': 0.0844691342710779}
    assert_dict_eq(r_dict, answer)
    
    r_dict = get_responsibility(data_random, centroids, beta)
    answer = {'centroid1': 0.14627992987508548, 
              'centroid2': 0.8537200701249145}
    assert_dict_eq(r_dict, answer)

    data = [10.1, 1, 23.2, 5.099]
    centroids = {"centroid1": [1, 1, 1, 1],
                 "centroid2": [10, 1, 23, 5],
                 "centroid3": [-100, 20.2, 52.9, -37.088]}
    
    r_dict = get_responsibility(data, centroids, beta)
    answer = {'centroid1': 0.08244480798864791, 
              'centroid2': 0.9175509528077845, 
              'centroid3': 4.239203567615249e-06}
    assert_dict_eq(r_dict, answer)
    
    print("test_get_responsibility passed.")


def test_update_soft_assignment():
    # set up
    beta = 0.1
    data, random_centroids, bad_centroids = setup_data_centroids()

    # random
    rtn = update_soft_assignment(data, random_centroids, beta)
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
    answer = {(-1.01714716, 0.95954521, 1.20493919, 0.34804443): {'centroid1': 0.4574843841340101, 'centroid2': 0.5425156158659898},
              (-1.36639346, -0.38664658, -1.02232584, -1.05902604): {'centroid1': 0.5007677881563272, 'centroid2': 0.4992322118436727},
              (1.13659605, -2.47109085, -0.83996912, -0.24579457): {'centroid1': 0.5344795051006204, 'centroid2': 0.46552049489937974},
              (-1.48090019, -1.47491857, -0.6221167, 1.79055006): {'centroid1': 0.4887440650103685, 'centroid2': 0.5112559349896315},
              (-0.31237952, 0.73762417, 0.39042814, -1.1308523): {'centroid1': 0.4760293414749974, 'centroid2': 0.5239706585250027},
              (-0.83095884, -1.73002213, -0.01361636, -0.32652741): {'centroid1': 0.5092202346857921, 'centroid2': 0.49077976531420786},
              (-0.78645408, 1.98342914, 0.31944446, -0.41656898): {'centroid1': 0.4480588926995608, 'centroid2': 0.5519411073004391},
              (-1.06190687, 0.34481172, -0.70359847, -0.27828666): {'centroid1': 0.46857274090476325, 'centroid2': 0.5314272590952367},
              (-2.01157677, 2.93965872, 0.32334723, -0.1659333): {'centroid1': 0.44333137003488515, 'centroid2': 0.5566686299651148},
              (-0.56669023, -0.06943413, 1.46053764, 0.01723844): {'centroid1': 0.47578799924615917, 'centroid2': 0.5242120007538408}}
    assert_cpl_dict_eq(rtn, answer)

    # bad
    rtn = update_soft_assignment(data, bad_centroids, beta)
    answer = {(-1.01714716, 0.95954521, 1.20493919, 0.34804443): {'centroid1': 0.8214744906007405, 'centroid2': 0.1785255093992595},
              (-1.36639346, -0.38664658, -1.02232584, -1.05902604): {'centroid1': 0.8817351662855751, 'centroid2': 0.1182648337144249},
              (1.13659605, -2.47109085, -0.83996912, -0.24579457): {'centroid1': 0.865330545967665, 'centroid2': 0.134669454032335},
              (-1.48090019, -1.47491857, -0.6221167, 1.79055006): {'centroid1': 0.8460638223935508, 'centroid2': 0.15393617760644912},
              (-0.31237952, 0.73762417, 0.39042814, -1.1308523): {'centroid1': 0.8524878506750114, 'centroid2': 0.1475121493249886},
              (-0.83095884, -1.73002213, -0.01361636, -0.32652741): {'centroid1': 0.8668516294399219, 'centroid2': 0.13314837056007817},
              (-0.78645408, 1.98342914, 0.31944446, -0.41656898): {'centroid1': 0.8314167906280353, 'centroid2': 0.16858320937196472},
              (-1.06190687, 0.34481172, -0.70359847, -0.27828666): {'centroid1': 0.8654691791995124, 'centroid2': 0.13453082080048773},
              (-2.01157677, 2.93965872, 0.32334723, -0.1659333): {'centroid1': 0.8171263166970885, 'centroid2': 0.1828736833029115},
              (-0.56669023, -0.06943413, 1.46053764, 0.01723844): {'centroid1': 0.8299042589594812, 'centroid2': 0.17009574104051883}}
    assert_cpl_dict_eq(rtn, answer)
    print("test_update_soft_assignment passed.")


def test_update_centroids():
    # set up
    beta = 0.1
    data, random_centroids, bad_centroids = setup_data_centroids()

    # random
    assignment_dict = update_soft_assignment(data, random_centroids, beta)
    answer = {'centroid1': [-0.8014801375995664, -0.0011682313637987026, 0.02434401028759097, -0.14874669787931735],
              'centroid2': [-0.8559310091121232, 0.16133973702187182, 0.07314226168771251, -0.14483894294652405]}
    rtn = update_centroids(assignment_dict)
    assert_list_dict_eq(rtn, answer)

    # bad
    assignment_dict = update_soft_assignment(data, bad_centroids, beta)
    answer = {'centroid1': [-0.823711394072307, 0.05687857018429321, 0.03411324426119401, -0.15348291214100224],
              'centroid2': [-0.8635875758325249, 0.23043094183316448, 0.1365596240929493, -0.10902393092336808]}
    rtn = update_centroids(assignment_dict)
    assert_list_dict_eq(rtn, answer)
    print("test_update_centroids passed.")


if __name__ == '__main__':
    test_get_responsibility()
    test_update_soft_assignment()
    test_update_centroids()
    print("all tests passed.")
