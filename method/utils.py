import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple
import math

__all__ = [
    'euclidean_distance',
    'calculate_total_distance',
    'calculate_distance_matrix',
    'kmeans',
    'neighbor_aggregation',
    'softmax',
    'normalize',
    'calc_avg_time',
]

def neighbor_aggregation(x:np.ndarray) -> np.ndarray:
    dist_matrix = cdist(x, x, metric='euclidean')
    dist_matrix = softmax(dist_matrix, axis=-1)
    return dist_matrix @ x

def softmax(x:np.ndarray, axis:int=-1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def normalize(x:np.ndarray, axis:int=-1, epsilon:float=1e-8) -> np.ndarray:
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return (x - mean) / (std + epsilon)

def euclidean_distance(X:np.ndarray, Y:np.ndarray) -> np.ndarray:
    return cdist(X, Y, metric='euclidean')

def calculate_total_distance(
        tour:list, 
        distance_matrix: np.ndarray
    ) -> float:
    return sum(distance_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))

def calc_avg_time(
        tour:List[int], 
        distance_matrix:np.ndarray
    ) -> Tuple[float, float]:
    arrival_time = {i: 0 for i in range(distance_matrix.shape[0])}
    total_time = 0
    for i in range(len(tour) - 1):
        start, end = tour[i], tour[i + 1]
        total_time = arrival_time[start] + distance_matrix[start, end]
        arrival_time[end] = total_time
    avg_time = np.mean(list(arrival_time.values()))
    return avg_time, total_time

def calculate_distance_matrix(cities: np.ndarray) -> np.ndarray:
    return euclidean_distance(cities, cities)

def kmeans(
        X:np.ndarray, k:int, max_iters:int=100
    ) -> Tuple[np.ndarray, np.ndarray]:
    """codeboos [n_samples, d_features], code [n_samples,]"""
    # Randomly initialize k cluster codebooks
    n_samples, d_features = X.shape
    codebooks = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Compute the distance between each data point and the codebooks
        distances = euclidean_distance(X, codebooks)
        
        # Assign each data point to the closest centroid
        code = np.argmin(distances, axis=1)
        
        # Compute the new codebooks
        new_codebooks = np.array([X[code == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.all(codebooks == new_codebooks):
            break
        codebooks = new_codebooks
        
    return codebooks, code