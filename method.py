import numpy as np

# 贪心算法解决TSP问题
def greedy_tsp(distance_matrix: np.ndarray):
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

def calculate_total_distance(tour:list, distance_matrix: np.ndarray):
    return sum(distance_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))

# 模拟退火算法解决TSP问题
def simulated_annealing(
        distance_matrix: np.ndarray, 
        initial_temp: float = 1000, 
        cooling_rate: float = 0.995, 
        max_iter: int = 10000
    ):
    num_cities = distance_matrix.shape[0]
    current_tour = list(range(num_cities))
    np.random.shuffle(current_tour)
    current_tour.append(current_tour[0])
    current_distance = calculate_total_distance(current_tour, distance_matrix)
    
    best_tour = list(current_tour)
    best_distance = current_distance
    
    temperature = initial_temp

    for i in range(max_iter):
        if temperature <= 0:
            break
        
        # 生成新的邻居解
        new_tour = list(current_tour)
        city1, city2 = np.random.randint(1, num_cities, size=2)
        new_tour[city1], new_tour[city2] = new_tour[city2], new_tour[city1]
        new_distance = calculate_total_distance(new_tour, distance_matrix)
        
        # 接受新的解的概率
        acceptance_probability = np.exp((current_distance - new_distance) / temperature)
        if new_distance < current_distance or np.random.rand() < acceptance_probability:
            current_tour = new_tour
            current_distance = new_distance
        
        # 更新最佳解
        if current_distance < best_distance:
            best_tour = list(current_tour)
            best_distance = current_distance
        
        # 降低温度
        temperature *= cooling_rate
    
    return best_tour