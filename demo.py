import numpy as np
from scipy.spatial.distance import cdist
from method import TSPMethod, solve, calc_avg_time, neighbor_aggregation, normalize
from typing import List, Tuple
from tqdm import tqdm

def solve_tsp(cities: np.ndarray, is_agg:bool = True) -> Tuple[List[int], float]:
    distance_matrix_origin = cdist(cities, cities, metric='euclidean')
    if is_agg:
        cities = neighbor_aggregation(normalize(cities, axis=0))
    distance_matrix = cdist(cities, cities, metric='euclidean')
    best_tour = solve(distance_matrix)
    avg_time, total_time, _ = calc_avg_time(best_tour, distance_matrix_origin)
    return best_tour, avg_time, total_time

def main():
    # generate random data
    np.random.seed(1234)
    dataset_size = 10000
    tsp_size = 300
    data = np.random.uniform(size=(dataset_size, tsp_size, 2)).astype(np.float32)

    pbar = tqdm(total=dataset_size, desc='Processing', dynamic_ncols=True)
    scale_num = 50
    result = {
        'original': [],
        'agg': [],
    }
    for i in range(dataset_size):
        cities = data[i][:scale_num]

        best_tour, avg_time, total_time = solve_tsp(cities, is_agg=False)
        result['original'].append({
            'tour': best_tour, 
            'avg_time': avg_time, 
            'total_time': total_time
        })

        best_tour, avg_time, total_time = solve_tsp(cities, is_agg=True)
        result['agg'].append({
            'tour': best_tour, 
            'avg_time': avg_time, 
            'total_time': total_time
        })

        pbar.update(1)
    pbar.close()

    for key in result.keys():
        mean_total_time = np.mean([x['total_time'] for x in result[key]])
        mean_avg_time = np.mean([x['avg_time'] for x in result[key]])
        print(f'{key} total_time: {mean_total_time}, avg_time: {mean_avg_time}')

if __name__ == '__main__':
    main()



    

