import matplotlib.pyplot as plt
import numpy as np
from plot import plot_cities, plot_transform
from scipy.spatial.distance import cdist
from method import solve
from gif import create_gif 

def softmax(x:np.ndarray, axis:int=-1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def normalize(x:np.ndarray, axis:int=-1, epsilon:float=1e-8) -> np.ndarray:
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return (x - mean) / (std + epsilon)

def cosine_dist(x:np.ndarray) -> np.ndarray:
    return x @ x.T

def neighbor_aggregation(cities:np.ndarray) -> np.ndarray:
    # note that the input cities should be normalized
    dist_matrix = cosine_dist(cities)
    weight = softmax(dist_matrix)
    cities = weight @ cities
    cities = normalize(cities, 0)
    return cities

def main():
    np.random.seed(1234)
    num_cities = 30
    cities = np.random.rand(num_cities, 2)
    for i in range(10):
        cities = normalize(cities, 0)
        plot_cities(cities, title=f'iter={i}')
        transformed_cities = neighbor_aggregation(cities)
        plot_transform(cities, transformed_cities, title=f'iter={i}')
        cities = transformed_cities
    
    create_gif('images')
    # create_gif('images/transform')

if __name__ == '__main__':
    main()