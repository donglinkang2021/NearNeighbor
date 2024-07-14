from method import softmax, normalize, neighbor_aggregation, calculate_distance_matrix
import numpy as np

# 生成随机城市坐标
np.random.seed(42)
num_cities = 5
cities = np.random.rand(num_cities, 2)
distance_matrix_city = calculate_distance_matrix(cities)
agg_cities = neighbor_aggregation(cities)
distance_matrix_agg_city = calculate_distance_matrix(agg_cities)
print(distance_matrix_city)
print(distance_matrix_agg_city)
print(distance_matrix_agg_city - distance_matrix_city)

# plot cities
import matplotlib
import matplotlib.pyplot as plt
colors = matplotlib.colormaps['tab20'] 
plt.figure(figsize=(10, 10))
plt.scatter(cities[:, 0], cities[:, 1], c='blue', label='cities')
for i, city in enumerate(cities):
    plt.annotate(f'{i}', city)

plt.scatter(agg_cities[:, 0], agg_cities[:, 1], c='red', label='agg_cities')
for i, city in enumerate(agg_cities):
    plt.annotate(f'{i}', city)

# plot arrows from cities to agg_cities
for idx, (city, agg_city) in enumerate(zip(cities, agg_cities)):
    plt.arrow(
        city[0], city[1], 
        agg_city[0] - city[0], 
        agg_city[1] - city[1], 
        head_width=0.01, 
        head_length=0.01, 
        length_includes_head=True, 
        color=colors(idx)
    )

plt.title('Cities')
plt.legend()
plt.axis('equal')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid()
img_name = 'cities.png'
plt.savefig(img_name)
print(f'{img_name} saved')

