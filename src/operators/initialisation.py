import matplotlib.pyplot as plt
import coloredlogs
import logging
import random
import time
import os
import sys

import numpy as np
import math as m

from src.util import sort_adjacency_list,adjacency_list
from src.operators.objective import constrained_fitness
from src.visualisations.stat import stat

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

new_limit = 5000
sys.setrecursionlimit(new_limit)

def random_valid_initialise(lambdaa: int, amount_cities, sorted_connected_dict) -> np.ndarray:
    closed_paths = []
    while len(closed_paths) < lambdaa:
        start_city = random.randint(0, amount_cities - 1)
        path = [start_city]
        initial_path = backtrack(path, amount_cities, sorted_connected_dict,closed_paths,True)
        if initial_path is not None:
            closed_paths.append(initial_path)
    closed_paths = np.array(closed_paths)
    return closed_paths

def greedy_initialise(lambdaa: int, amount_cities, sorted_connected_dict) -> np.ndarray:
    closed_paths = []
    while len(closed_paths) < lambdaa:
        start_city = random.randint(0, amount_cities - 1)
        path = [start_city]
        initial_path = backtrack(path, amount_cities, sorted_connected_dict, closed_paths, False)
        if initial_path is not None:
            closed_paths.append(initial_path)
    closed_paths = np.array(closed_paths)
    return closed_paths

def random_and_greedy_initialise(lambdaa: int, amount_cities:int, adj_matrix, proportion:float):
    match(proportion):
        case 0:
            count = max(1, lambdaa)
            output2 = greedy_initialise(count, amount_cities, adj_matrix)
            return output2
        
        case 1:
            count = max(1, lambdaa)
            output1 = random_valid_initialise(lambdaa, amount_cities, adj_matrix)
            return output1

        case _: 
            random_count = max(1, m.floor(lambdaa * proportion))
            greedy_count = max(1, lambdaa - random_count)

            output1 = random_valid_initialise(random_count,amount_cities, adj_matrix)
            output2 = greedy_initialise(greedy_count,amount_cities,adj_matrix)
            return np.concatenate((output1,output2),axis=0)

def backtrack(current_path, amount_cities, sorted_connected_dict, closed_paths, rand=False):
    if len(current_path) == amount_cities:
        last_city = current_path[-1]
        start_city = current_path[0]
        if start_city in sorted_connected_dict[last_city]:
            if current_path in closed_paths:
                return None
            else:
                return current_path
        else:
            return None

    current_city = current_path[-1]
    neighbors = sorted_connected_dict[current_city]

    if rand: 
        idx = min(5, len(neighbors))
        permuted_neighbors = np.concatenate((np.random.permutation(neighbors[idx:]), neighbors[:idx]))

    else : 
        permuted_neighbors = list(neighbors)

    for neighbor in permuted_neighbors:
        if neighbor not in current_path:
            new_path = current_path + [neighbor]
            result = backtrack(new_path, amount_cities, sorted_connected_dict, closed_paths,rand)
            if result is not None:
                return result
    return None
    


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "../../assingment_data/tour1000.csv")
    file = open(csv_path)
    DISTANCE = np.loadtxt(csv_path, delimiter=",")
    file.close()

    # FIXME: OPLETTEN: bij adjacency list de originele distance matrix gebruiken !!!
    ADJ_DICT = adjacency_list(DISTANCE)
    SORTED_ADJ_DICT = sort_adjacency_list(ADJ_DICT, DISTANCE)

    for i in range(2):
        t1 = time.time()
        result1 = greedy_initialise(
            100, DISTANCE.shape[0], SORTED_ADJ_DICT)
        t2 = time.time()
        print("time")
        print(t2 - t1)

        result2 = random_valid_initialise(
            100, DISTANCE.shape[0], SORTED_ADJ_DICT)
        t3 = time.time()
        print("time")
        print(t3 - t2)

        xl = list(range(0, 100))
        arr = [] * 100

        for x in range(len(result2)):
            fitness1 = constrained_fitness(result2[x], DISTANCE)
            arr.append(fitness1)

        plt.figure()
        plt.plot(xl, np.sort(arr))
        plt.show()

        print(np.unique(arr).size)
        print(stat.calculate_amount_duplicates(result2))

        arr = [] * 100
        for x in range(len(result1)):
            fitness2 = constrained_fitness(result1[x], DISTANCE)
            arr.append(fitness2)

        plt.figure()
        plt.plot(xl, np.sort(arr))
        plt.show()

        print(np.unique(arr).size)
        print(stat.calculate_amount_duplicates(result2))
