from method import (
    neighbor_aggregation, 
    calculate_distance_matrix,
    calc_avg_time,
    solve,
    normalize,
    TSPMethod,
)
import numpy as np

# 生成随机城市坐标
np.random.seed(1234)
num_cities = 30
cities = np.random.rand(num_cities, 2)
distance_matrix_city = calculate_distance_matrix(cities)

norm_cities = normalize(cities, axis=0)
distance_matrix_norm_city = calculate_distance_matrix(norm_cities)

agg_cities = neighbor_aggregation(norm_cities)
distance_matrix_agg_city = calculate_distance_matrix(agg_cities)

route0 = solve(distance_matrix_city, TSPMethod.NEAREST_NEIGHBOR)
route1 = solve(distance_matrix_norm_city, TSPMethod.NEAREST_NEIGHBOR)
route2 = solve(distance_matrix_agg_city, TSPMethod.NEAREST_NEIGHBOR)
avg_time0, total_time0 = calc_avg_time(route0, distance_matrix_city)
avg_time1, total_time1 = calc_avg_time(route1, distance_matrix_city)
avg_time2, total_time2 = calc_avg_time(route2, distance_matrix_city)

# plot cities
import matplotlib
import matplotlib.pyplot as plt
colors = matplotlib.colormaps['tab20'] 
plt.figure(figsize=(10, 10))
plt.scatter(cities[:, 0], cities[:, 1], c='blue', label='cities', marker='x')
plt.plot(cities[0][0], cities[0][1], 'rp', markersize=10, label='Start Point')
for i, city in enumerate(cities):
    plt.annotate(f'{i}', city, fontsize=12)

plt.scatter(norm_cities[:, 0], norm_cities[:, 1], c='green', label='norm_cities', marker='o')
for i, city in enumerate(norm_cities):
    plt.annotate(f'{i}', city, fontsize=12)

plt.scatter(agg_cities[:, 0], agg_cities[:, 1], c='red', label='agg_cities', marker='o')
for i, city in enumerate(agg_cities):
    plt.annotate(f'{i}', city, fontsize=12)


# plot arrows from cities to agg_cities
for idx, (city, norm_city, agg_city) in enumerate(zip(cities, norm_cities, agg_cities)):
    plt.arrow(
        city[0], city[1], 
        norm_city[0] - city[0], 
        norm_city[1] - city[1], 
        head_width=0.05, 
        head_length=0.05, 
        length_includes_head=True, 
        color=colors(idx)
    )
    plt.arrow(
        norm_city[0], norm_city[1], 
        agg_city[0] - norm_city[0], 
        agg_city[1] - norm_city[1], 
        head_width=0.05, 
        head_length=0.05, 
        length_includes_head=True, 
        color=colors(idx)
    )
    plt.arrow(
        city[0], city[1], 
        agg_city[0] - city[0], 
        agg_city[1] - city[1], 
        head_width=0.03, 
        head_length=0.03, 
        length_includes_head=True, 
        color=colors(idx)
    )

# plot route
for i in range(len(route0) - 1):
    start, end = route0[i], route0[i + 1]
    plt.arrow(
        cities[start, 0], cities[start, 1], 
        cities[end, 0] - cities[start, 0], 
        cities[end, 1] - cities[start, 1],
        head_width=0.02, length_includes_head=True, color='blue', 
        linestyle='--', linewidth=2, alpha=0.5
    )
    start, end = route1[i], route1[i + 1]
    plt.arrow(
        cities[start, 0], cities[start, 1], 
        cities[end, 0] - cities[start, 0], 
        cities[end, 1] - cities[start, 1],
        head_width=0.02, length_includes_head=True, color='green', 
        linestyle='-.', linewidth=1, alpha=0.5
    )  
    start, end = route2[i], route2[i + 1]
    plt.arrow(
        cities[start, 0], cities[start, 1], 
        cities[end, 0] - cities[start, 0], 
        cities[end, 1] - cities[start, 1],
        head_width=0.02, length_includes_head=True, color='red',
        linestyle='-', linewidth=1, alpha=0.5
    )    

metric1 = f'Avg time: {avg_time0:.2f} vs {avg_time1:.2f} vs {avg_time2:.2f}'
metric2 = f'Total time: {total_time0:.2f} vs {total_time1:.2f} vs {total_time2:.2f}'
plt.title(f'Cities, Norm Cities, Aggregated Cities\n{metric1}\n{metric2}')
# plt.title(f'Cities, Norm Cities, Aggregated Cities')
plt.legend()
plt.axis('equal')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid()
img_name = 'cities.png'
plt.savefig(img_name)
print(f'{img_name} saved')

