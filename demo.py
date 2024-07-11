import numpy as np
from scipy.spatial.distance import cdist

# 生成随机城市坐标
np.random.seed(42)
num_cities = 30
cities = np.random.rand(num_cities, 2)

# 计算城市之间的距离矩阵
distance_matrix = cdist(cities, cities, metric='euclidean')

# 求解
from method import (
    calculate_total_distance,
    greedy_tsp, 
    simulated_annealing
)
# method = 'Greedy Algorithm'
# tour = greedy_tsp(distance_matrix)

method = 'Simulated Annealing'
tour = simulated_annealing(
    distance_matrix, 
    initial_temp=3000, 
    cooling_rate=0.995, 
    max_iter=10000
)

# 可视化
total_distance = calculate_total_distance(tour, distance_matrix)
from plot import plot_tsp_solution
plot_tsp_solution(cities, tour, total_distance, method)

