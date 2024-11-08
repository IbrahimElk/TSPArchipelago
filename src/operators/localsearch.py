import numpy as np
import coloredlogs
import logging
import os

from src.operators.initialisation import random_valid_initialise
from src.operators.objective import constrained_fitness
from src.util import adjacency_list,sort_adjacency_list,replace_infs
from numba import njit
import numpy as np
import random 


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

@njit
def two_opts(ind, distance_matrix, max_iterations):
    path = ind
    length = len(ind)
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        best_improv = False
        for i in range(1, length - 2):
            for j in range(i + 2, length):

                point1 = path[i - 1]
                point2 = path[i]
                point3 = path[j-1]
                point4 = path[j]
                
                result = distance_matrix[point1][point3] + distance_matrix[point2][point4] - distance_matrix[point1][point2] - distance_matrix[point3][point4]   

                if result < 0:
                    path[i:j] = path[j - 1:i - 1:-1]
                    best_improv = True

        if not best_improv:
            break
    return path



if __name__ == "__main__":

    from pytsp import k_opt_tsp
    
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "../../assingment_data/tour500.csv")
    file = open(csv_path)
    distance_matrix = np.loadtxt(csv_path, delimiter=",")
    file.close()

# -----------------------------------------------------------------------------------------
    
    a = adjacency_list(distance_matrix)
    b = sort_adjacency_list(a, distance_matrix)
    
    adjusted_distance_matrix = replace_infs(distance_matrix,2)
    print(len(adjusted_distance_matrix))

    tour = k_opt_tsp.tsp_3_opt(adjusted_distance_matrix)
    result = constrained_fitness(tour, adjusted_distance_matrix)
    print(result)
    # num_cities = len(distance_matrix)
    # initial_route = random_valid_initialise(1, num_cities, b)[0]
    # best_route = []
    # best_distance = 0

    # cost = constrained_fitness(initial_route, distance_matrix)
    # print("initial Path before 2-opt: ", initial_route)
    # print("intial Fitness: ", cost)

    # def constrained_fitness(individual: list[int]) -> np.float64:
    #     start = individual[-1]
    #     totalCost = 0

    #     for destination in individual:
    #         cost = distance_matrix[start][destination]
    #         if cost == float("inf"):
    #             return cost
    #         totalCost += cost
    #         start = destination
    #     return totalCost

    # best_path, best_fitness = two_opts(initial_route, distance_matrix, iterations=100)

    # print("Best Path after 2-opt: ", best_path)
    # print("Best Fitness: ", best_fitness)
