import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from method import (
    neighbor_aggregation, 
    calculate_distance_matrix,
    calc_avg_time,
    solve,
    normalize,
    TSPMethod,
)
from typing import List

# plot cities
def plot_cities(cities:np.ndarray, title:str):
    plt.figure(figsize=(10, 10))
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', label='cities', marker='x')
    plt.plot(cities[0][0], cities[0][1], 'rp', markersize=10, label='Start Point')
    for i, city in enumerate(cities):
        plt.annotate(f'{i}', city, fontsize=12)
    plt.legend()
    plt.title(title)
    plt.axis('equal')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    img_name = f'images/cities_{title}.png'
    plt.savefig(img_name)
    print(f'{img_name} saved')

def plot_transform(cities:np.ndarray, cities_changed:np.ndarray, title:str):
    colors = matplotlib.colormaps['tab20'] 
    plt.figure(figsize=(10, 10))
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', label='cities1', marker='x')
    plt.scatter(cities_changed[:, 0], cities_changed[:, 1], c='red', label='cities2', marker='o')
    for i, city in enumerate(cities):
        plt.annotate(f'{i}', city, fontsize=12)
    for i, city in enumerate(cities_changed):
        plt.annotate(f'{i}', city, fontsize=12)

    for idx, (city, city_changed) in enumerate(zip(cities, cities_changed)):
        plt.arrow(
            city[0], city[1], 
            city_changed[0] - city[0], 
            city_changed[1] - city[1], 
            head_width=0.05, 
            head_length=0.05, 
            length_includes_head=True, 
            color=colors(idx)
        )
    plt.legend()
    plt.title(title)
    plt.axis('equal')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    img_name = f'images/transform_{title}.png'
    plt.savefig(img_name)
    print(f'{img_name} saved')

def plot_route(cities:np.ndarray, distance_matrix_city:np.ndarray, route:List[int], title:str):
    avg_time, total_time = calc_avg_time(route, distance_matrix_city)
    plt.figure(figsize=(10, 10))
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', label='cities', marker='x')
    plt.plot(cities[0][0], cities[0][1], 'rp', markersize=10, label='Start Point')
    for i, city in enumerate(cities):
        plt.annotate(f'{i}', city, fontsize=12)
    for i in range(len(route) - 1):
        start, end = route[i], route[i + 1]
        plt.arrow(
            cities[start, 0], cities[start, 1], 
            cities[end, 0] - cities[start, 0], 
            cities[end, 1] - cities[start, 1],
            head_width=0.02, length_includes_head=True, linewidth=2
        )
    plt.legend()
    plt.title(f'{title} Routes\nAvg time: {avg_time:.2f}, Total time: {total_time:.2f}')
    plt.axis('equal')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    img_name = f'images/route_{title}.png'
    plt.savefig(img_name)
    print(f'{img_name} saved')

def main():
    Path('images').mkdir(exist_ok=True)

    # 生成随机城市坐标
    np.random.seed(1234)
    num_cities = 20
    cities = np.random.rand(num_cities, 2)
    distance_matrix_city = calculate_distance_matrix(cities)

    norm_cities = normalize(cities, axis=0)
    distance_matrix_norm_city = calculate_distance_matrix(norm_cities)

    agg_cities = neighbor_aggregation(cities)
    distance_matrix_agg_city = calculate_distance_matrix(agg_cities)
    
    agg_norm_cities = neighbor_aggregation(norm_cities)
    distance_matrix_agg_norm_city = calculate_distance_matrix(agg_norm_cities)

    plot_cities(cities, 'Original')
    plot_cities(norm_cities, 'Normalized')
    plot_cities(agg_cities, 'Aggregated')
    plot_cities(agg_norm_cities, 'Normalized_Aggregated')

    plot_transform(cities, norm_cities, 'Original2Normalized')
    plot_transform(cities, agg_cities, 'Original2Aggregated')
    plot_transform(norm_cities, agg_norm_cities, 'Normalized2Aggregated')

    route0 = solve(distance_matrix_city, TSPMethod.NEAREST_NEIGHBOR)
    route1 = solve(distance_matrix_norm_city, TSPMethod.NEAREST_NEIGHBOR)
    route2 = solve(distance_matrix_agg_city, TSPMethod.NEAREST_NEIGHBOR)
    route3 = solve(distance_matrix_agg_norm_city, TSPMethod.NEAREST_NEIGHBOR)

    plot_route(cities, distance_matrix_city, route0, 'Original')
    plot_route(cities, distance_matrix_city, route1, 'Normalized')
    plot_route(cities, distance_matrix_city, route2, 'Aggregated')
    plot_route(cities, distance_matrix_city, route3, 'Normalized_Aggregated')

if __name__ == '__main__':
    main()