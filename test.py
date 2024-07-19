import numpy as np
from plot import (
    plot_cities,
    plot_transform,
    plot_3d_cities, 
    plot_3d_transform
)
from scipy.spatial.distance import cdist
from gif import create_gif 
from method import kmeans

def softmax(x:np.ndarray, axis:int=-1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def normalize(x:np.ndarray, axis:int=-1, epsilon:float=1e-8) -> np.ndarray:
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return (x - mean) / (std + epsilon)

def neighbor_aggregation(cities:np.ndarray) -> np.ndarray:
    # note that the input cities should be normalized
    dist_matrix = cities @ cities.T
    weight = softmax(dist_matrix)
    cities = weight @ cities
    cities = normalize(cities, 0)
    return cities

def k_neighbor_aggregation(cities:np.ndarray, k:int) -> np.ndarray:
    assert cities.shape[0] % k == 0, 'The number of cities should be divisible by k'
    cities = cities.reshape(k, -1, 2) # k, num_cities // k, 2
    cities = normalize(cities, 0)
    dist_matrix = cities @ cities.transpose(0, 2, 1) # k, num_cities // k, num_cities // k
    weight = softmax(dist_matrix) # k, num_cities // k, num_cities // k
    cities = weight @ cities
    cities = cities.reshape(-1, 2) # num_cities, 2
    return cities

def main():
    # np.random.seed(1234)
    num_cities = 400
    cities_origin = np.random.rand(num_cities, 2)
    k = 16

    cities = cities_origin
    cities = normalize(cities, 0)
    for i in range(20):
        plot_cities(cities, title=f'iter={i:02d}')
        # transformed_cities = neighbor_aggregation(cities)
        transformed_cities = k_neighbor_aggregation(cities, k)
        plot_transform(cities, transformed_cities, title=f'iter={i:02d}')
        cities = transformed_cities

    create_gif('images/transform')
    create_gif('images/cities')

    
if __name__ == '__main__':
    main()