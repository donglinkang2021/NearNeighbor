import numpy as np
import matplotlib.pyplot as plt
def plot_tsp_solution(
        cities: np.ndarray, 
        tour: list, 
        total_distance: float,
        method: str,
    ):
    # 可视化TSP路径
    plt.figure(figsize=(8, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='o')

    # 绘制路径并添加箭头
    # 同时计算路径长度
    for i in range(len(tour) - 1):
        start, end = tour[i], tour[i + 1]
        plt.arrow(
            cities[start, 0], cities[start, 1], 
            cities[end, 0] - cities[start, 0], 
            cities[end, 1] - cities[start, 1],
            head_width=0.02, length_includes_head=True, color='blue'
        )

    # 显示城市编号
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i), fontsize=12, ha='right')

    title = f"Traveling Salesman Problem Solution using {method}"
    metric = f"Total Distance: {total_distance:.2f}"

    plt.title(f"{title}\n{metric}")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.savefig(f"images/{title.replace(' ', '_')}.png")