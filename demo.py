import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 生成随机城市坐标
np.random.seed(42)
num_cities = 20
cities = np.random.rand(num_cities, 2)

# 计算城市之间的距离矩阵
distance_matrix = cdist(cities, cities, metric='euclidean')

# 贪心算法解决TSP问题
def greedy_tsp(distance_matrix):
    num_cities = distance_matrix.shape[0]
    visited = [False] * num_cities
    tour = [0]
    visited[0] = True
    for _ in range(num_cities - 1):
        last = tour[-1]
        next_city = np.argmin([distance_matrix[last, j] if not visited[j] else np.inf for j in range(num_cities)])
        tour.append(next_city)
        visited[next_city] = True
    tour.append(0)  # 回到起点
    return tour

tour = greedy_tsp(distance_matrix)

# 可视化TSP路径
plt.figure(figsize=(8, 6))
plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='o')

# 绘制路径并添加箭头
for i in range(len(tour) - 1):
    start, end = tour[i], tour[i + 1]
    plt.arrow(
        cities[start, 0], cities[start, 1], 
        cities[end, 0] - cities[start, 0], cities[end, 1] - cities[start, 1],
        head_width=0.02, length_includes_head=True, color='blue'
    )

# 显示城市编号
for i, (x, y) in enumerate(cities):
    plt.text(x, y, str(i), fontsize=12, ha='right')

plt.title('Traveling Salesman Problem Solution using Greedy Algorithm with Arrows')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()
