import numpy as np
from scipy.spatial.distance import cdist
from method import TSPMethod, solve, calculate_total_distance, neighbor_aggregation
from plot import plot_tsp_solution

# 生成随机城市坐标
np.random.seed(42)
num_cities = 30
cities = np.random.rand(num_cities, 2)
agg_cities = neighbor_aggregation(cities)

# 计算城市之间的距离矩阵
distance_matrix = cdist(cities, cities, metric='euclidean')
agg_distance_matrix = cdist(agg_cities, agg_cities, metric='euclidean')

# 求解
for method in TSPMethod:
    best_tour = solve(agg_distance_matrix, method)
    total_distance = calculate_total_distance(best_tour, distance_matrix)
    print(f"Best Distance using {method.phrase}: {best_tour}\nTotal Distance: {total_distance:.2f}")
    plot_tsp_solution(cities, best_tour, total_distance, method.phrase)

