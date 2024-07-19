import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List
from method import calc_avg_time
from pathlib import Path
Path("images").mkdir(exist_ok=True)

def plot_tsp_solution(
        cities: np.ndarray, 
        tour: list, 
        total_distance: float,
        method: str,
    ):
    plt.figure(figsize=(10, 10))
    plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='o')

    for i in range(len(tour) - 1):
        start, end = tour[i], tour[i + 1]
        plt.arrow(
            cities[start, 0], cities[start, 1], 
            cities[end, 0] - cities[start, 0], 
            cities[end, 1] - cities[start, 1],
            head_width=0.02, length_includes_head=True, color='blue'
        )

    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i), fontsize=12, ha='right')

    title = f"Traveling Salesman Problem Solution using {method}"
    metric = f"Total Distance: {total_distance:.2f}"

    plt.title(f"{title}\n{metric}")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.savefig(f"images/{title.replace(' ', '_')}.png")
    plt.close()
    print(f"Plot saved as images/{title.replace(' ', '_')}.png")

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
    Path("images/cities").mkdir(exist_ok=True)
    img_name = f'images/cities/{title}.png'
    plt.savefig(img_name)
    plt.close()
    print(f'{img_name} saved')

def plot_transform(cities:np.ndarray, cities_changed:np.ndarray, title:str):
    n_city, dim = cities.shape
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_city))
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
            head_width=0.03, 
            head_length=0.03, 
            length_includes_head=True, 
            color=colors[idx]
        )
    plt.legend()
    plt.title(title)
    plt.axis('equal')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    Path("images/transform").mkdir(exist_ok=True)
    img_name = f'images/transform/{title}.png'
    plt.savefig(img_name)
    plt.close()
    print(f'{img_name} saved')

def plot_transform_route(cities_list:np.ndarray, title:str):
    n_frame, n_city, dim = cities_list.shape
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_city))
    plt.figure(figsize=(20, 20))
    for idx in range(n_city):
        plt.plot(
            cities_list[:, idx, 0], cities_list[:, idx, 1], 
            marker='o', color=colors[idx], 
            label=f'{idx}'
        )

        for j in range(n_frame - 1):
            plt.arrow(
                cities_list[j, idx, 0], cities_list[j, idx, 1],
                cities_list[j + 1, idx, 0] - cities_list[j, idx, 0],
                cities_list[j + 1, idx, 1] - cities_list[j, idx, 1],
                head_width=0.05, 
                head_length=0.05, 
                length_includes_head=True, 
                color=colors[idx]
            )
    if n_city < 50:
        plt.legend(
            loc='upper left', 
            bbox_to_anchor=(1, 1), 
            fontsize='small', 
            framealpha=0.5, 
            ncol=2
        )
    plt.title(title)
    # plt.axis('equal')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    Path("images/transform_route").mkdir(exist_ok=True)
    img_name = f'images/transform_route/{title}.png'
    plt.savefig(img_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f'{img_name} saved')

def plot_route(cities:np.ndarray, distance_matrix_city:np.ndarray, route:List[int], title:str):
    avg_time, total_time, arrival_time = calc_avg_time(route, distance_matrix_city)
    plt.figure(figsize=(10, 10))
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', label='cities', marker='x')
    plt.plot(cities[0][0], cities[0][1], 'rp', markersize=10, label='Start Point')
    for i, city in enumerate(cities):
        plt.annotate(f'{i} {arrival_time[i]:.2f}', city, fontsize=12, color='blue', ha='right', va='bottom')
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
    Path("images/route").mkdir(exist_ok=True)
    img_name = f'images/route/{title}.png'
    plt.savefig(img_name)
    plt.close()
    print(f'{img_name} saved')

# plot 3d cities
def plot_3d_cities(cities:np.ndarray, title:str):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cities[:, 0], cities[:, 1], cities[:, 2])
    for i, city in enumerate(cities):
        ax.text(city[0], city[1], city[2], f'{i}', fontsize=12)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(title)
    Path("images/3d_cities").mkdir(exist_ok=True)
    img_name = f'images/3d_cities/{title}.png'
    plt.savefig(img_name)
    plt.close()
    print(f'{img_name} saved')

# plot 3d transform
def plot_3d_transform(cities:np.ndarray, cities_changed:np.ndarray, title:str):
    n_city, dim = cities.shape
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cities[:, 0], cities[:, 1], cities[:, 2])
    ax.scatter(cities_changed[:, 0], cities_changed[:, 1], cities_changed[:, 2])
    for i, city in enumerate(cities):
        ax.text(city[0], city[1], city[2], f'{i}', fontsize=12)
    for i, city in enumerate(cities_changed):
        ax.text(city[0], city[1], city[2], f'{i}', fontsize=12)
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_city))
    for city, city_changed in zip(cities, cities_changed):
        ax.quiver(
            city[0], city[1], city[2], 
            city_changed[0] - city[0], 
            city_changed[1] - city[1], 
            city_changed[2] - city[2],
            color=colors[i]
        )
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(title)
    Path("images/3d_transform").mkdir(exist_ok=True)
    img_name = f'images/3d_transform/{title}.png'
    plt.savefig(img_name)
    plt.close()
    print(f'{img_name} saved')