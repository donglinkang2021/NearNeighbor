import numpy as np
from scipy.spatial.distance import cdist
from method import TSPMethod, solve, calculate_distance_matrix
from plot import plot_route


def main():
    # generate random data
    np.random.seed(1234)
    dataset_size = 10000
    tsp_size = 300
    data = np.random.uniform(size=(dataset_size, tsp_size, 2)).astype(np.float32)

    sample_idx = 0
    sample_data = data[sample_idx][:30]
    dist_matrix = calculate_distance_matrix(sample_data)

    for method in TSPMethod:
        route = solve(dist_matrix, method)
        plot_route(sample_data, dist_matrix, route, method.name)

if __name__ == '__main__':
    main()



    

