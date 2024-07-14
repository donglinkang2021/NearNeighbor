import numpy as np
from scipy.spatial.distance import cdist
from method import TSPMethod, solve, calculate_total_distance, neighbor_aggregation
from typing import List, Tuple
from tqdm import tqdm

def solve_tsp(cities: np.ndarray, is_agg:bool = True) -> Tuple[List[int], float]:
    if is_agg:
        agg_cities = neighbor_aggregation(cities)
    else:
        agg_cities = cities
    distance_matrix = cdist(agg_cities, agg_cities, metric='euclidean')
    method = TSPMethod.NEAREST_NEIGHBOR
    best_tour = solve(distance_matrix, method)

    distance_matrix = cdist(cities, cities, metric='euclidean')
    total_distance = calculate_total_distance(best_tour, distance_matrix)
    return best_tour, total_distance

def main():
    # generate random data
    np.random.seed(1234)
    dataset_size = 10000
    tsp_size = 300
    data = np.random.uniform(size=(dataset_size, tsp_size, 2)).astype(np.float32)

    pbar = tqdm(total=dataset_size, desc='Processing', dynamic_ncols=True)
    scale_num = 30
    result_list = []
    agg_result_list = []
    for i in range(dataset_size):
        cities = data[i][:scale_num]

        best_tour, total_distance = solve_tsp(cities, is_agg=False)
        result_list.append({'tour': best_tour, 'total_distance': total_distance})

        best_tour, total_distance = solve_tsp(cities, is_agg=True)
        agg_result_list.append({'tour': best_tour, 'total_distance': total_distance})
        pbar.update(1)
    pbar.close()
    avg_distance = np.mean([result['total_distance'] for result in result_list])
    print(f'Average distance: {avg_distance}')
    avg_distance = np.mean([result['total_distance'] for result in agg_result_list])
    print(f'Average distance with aggregation: {avg_distance}')

if __name__ == '__main__':
    main()



    

