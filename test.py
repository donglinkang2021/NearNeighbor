import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from method import (
    neighbor_aggregation, 
    calculate_distance_matrix,
    solve,
    normalize,
    TSPMethod,
)
from plot import plot_cities, plot_route, plot_transform


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
    plot_transform(norm_cities, agg_norm_cities, 'Normalized2Normalized_Aggregated')
    plot_transform(cities, agg_norm_cities, 'Original2Normalized_Aggregated')

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