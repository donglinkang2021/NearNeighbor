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
    simulated_annealing,
    greedy_algorithm, 
    genetic_algorithm
)
# method = 'Greedy Algorithm'
# best_tour = greedy_algorithm(distance_matrix)
# best_distance = calculate_total_distance(best_tour, distance_matrix)

# method = 'Simulated Annealing'
# best_tour, best_distance = simulated_annealing(
#     distance_matrix, 
#     initial_temp=5, 
#     cooling_rate=0.995, 
#     max_iter=10000
# )

method = 'Genetic Algorithm'
best_tour, best_distance = genetic_algorithm(
    distance_matrix,
    pop_size=200,
    elite_size=40,
    mutation_rate=0.01,
    generations=500
)

from plot import plot_tsp_solution
plot_tsp_solution(cities, best_tour, best_distance, method)
print(f"Best Tour: {best_tour}")
print(f"Best Distance: {best_distance}")

