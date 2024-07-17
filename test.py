import numpy as np
from plot import plot_cities, plot_transform, plot_route, plot_transform_route
from scipy.spatial.distance import cdist
from gif import create_gif 
from method import solve

def softmax(x:np.ndarray, axis:int=-1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def normalize(x:np.ndarray, axis:int=-1, epsilon:float=1e-8) -> np.ndarray:
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return (x - mean) / (std + epsilon)

def attention_dist(x:np.ndarray) -> np.ndarray:
    return x @ x.T

def cosine_dist(cities:np.ndarray) -> np.ndarray:
    return cdist(cities, cities, metric='cosine')

def euclidean_dist(cities:np.ndarray) -> np.ndarray:
    return cdist(cities, cities, metric='euclidean')

def neighbor_aggregation(cities:np.ndarray) -> np.ndarray:
    # note that the input cities should be normalized
    dist_matrix = attention_dist(cities)
    weight = softmax(dist_matrix)
    cities = weight @ cities
    cities = normalize(cities, 0)
    return cities

def main():
    # np.random.seed(1234)
    num_cities = 500
    cities_origin = np.random.rand(num_cities, 2)
    distance_matrix_origin = euclidean_dist(cities_origin)

    cities_list = []
    cities = cities_origin
    cities = normalize(cities, 0)
    cities_list.append(cities)
    for i in range(3):
        # plot_cities(cities, title=f'iter={i}')
        transformed_cities = neighbor_aggregation(cities)
        # plot_transform(cities, transformed_cities, title=f'iter={i}')
        cities = transformed_cities
        cities_list.append(cities)
        # distance_matrix = euclidean_dist(cities)
        # best_tour = solve(distance_matrix)
        # plot_route(cities_origin, distance_matrix_origin, best_tour, title=f'iter={i}')

    cities_list = np.array(cities_list)
    print(cities_list.shape)
    plot_transform_route(cities_list, title='transform_route')
    

    # create_gif('images/cities')
    # create_gif('images/transform')
    # create_gif('images/route')

if __name__ == '__main__':
    main()