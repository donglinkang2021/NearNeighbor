import numpy as np

def calculate_total_distance(
        tour:list, 
        distance_matrix: np.ndarray
    ):
    return sum(distance_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
