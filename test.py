import numpy as np
from plot import plot_3d_cities, plot_3d_transform
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
    cities_origin = np.random.rand(num_cities, 3)

    cities = cities_origin
    cities = normalize(cities, 0)
    for i in range(20):
        plot_3d_cities(cities, title=f'iter={i:02d}')
        transformed_cities = neighbor_aggregation(cities)
        plot_3d_transform(cities, transformed_cities, title=f'iter={i:02d}')
        cities = transformed_cities

    create_gif('images/3d_transform')
    create_gif('images/3d_cities')

    
if __name__ == '__main__':
    main()