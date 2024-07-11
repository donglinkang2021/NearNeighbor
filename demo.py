import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 生成随机城市坐标
np.random.seed(42)
num_cities = 30
cities = np.random.rand(num_cities, 2)

# 计算城市之间的距离矩阵
distance_matrix = cdist(cities, cities, metric='euclidean')

from method import greedy_tsp
tour = greedy_tsp(distance_matrix)

# 可视化TSP路径
plt.figure(figsize=(8, 6))
plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='o')

# 绘制路径并添加箭头
# 同时计算路径长度
total_distance = 0
for i in range(len(tour) - 1):
    start, end = tour[i], tour[i + 1]
    plt.arrow(
        cities[start, 0], cities[start, 1], 
        cities[end, 0] - cities[start, 0], 
        cities[end, 1] - cities[start, 1],
        head_width=0.02, length_includes_head=True, color='blue'
    )
    total_distance += distance_matrix[start, end]

# 显示城市编号
for i, (x, y) in enumerate(cities):
    plt.text(x, y, str(i), fontsize=12, ha='right')

title = f"Traveling Salesman Problem Solution using Greedy Algorithm\nTotal Distance: {total_distance:.2f}"

plt.title(title)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.savefig('tsp_greedy.png')
