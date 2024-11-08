import numpy as np
import os
from src.util import adjacency_list, sort_adjacency_list,replace_infs
from src.operators.initialisation import random_valid_initialise
from src.operators.objective import constrained_fitness
import time

def select_tourn(population: np.ndarray, k: int, fitness: callable):
    """
    Performs tournament selection to choose the best individual from a population.

    Args:
    - population (np.ndarray): Population of individuals.
    - k (int): Number of individuals participating in each tournament.
    - fitness (callable): Function to calculate fitness.

    Returns:
    - np.ndarray: Selected individual based on tournament selection.

    Note:
    This method performs tournament selection by randomly selecting 'k' individuals from
    the population, evaluating their fitness using the provided fitness function, and returning
    the best individual among them.
    """
    selected_indices = np.random.choice(range(len(population)), k)
    selected_population = population[selected_indices]
    selected_fitness = [fitness(individual)
                        for individual in selected_population]
    best_index = np.argmin(selected_fitness)
    return selected_population[best_index]


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "../../assingment_data/tour50.csv")
    file = open(csv_path)
    DISTANCE = np.loadtxt(csv_path, delimiter=",")
    file.close()

    DISTANCE = replace_infs(DISTANCE)
    ADJ_DICT = adjacency_list(DISTANCE)
    SORTED_ADJ_DICT = sort_adjacency_list(ADJ_DICT, DISTANCE)
    init = random_valid_initialise(50, DISTANCE.shape[0], SORTED_ADJ_DICT)

    k = 10

    t1 = time.time()
    first_selected_individual = select_tourn(
        init, k, constrained_fitness, {"DISTANCE_MATRIX": DISTANCE})
    t2 = time.time()
    second_selected_individual = select_tourn(
        init, k, constrained_fitness, {"DISTANCE_MATRIX": DISTANCE})
    t3 = time.time()

    print(t2-t1)
    print(t3-t2)
    print(t3-t1)
    print("\nFirst Selected Individual with the Lowest Sum:")
    print(first_selected_individual)
    print("\Second nSelected Individual with the Lowest Sum:")
    print(second_selected_individual)
