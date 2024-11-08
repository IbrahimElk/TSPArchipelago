import time
import numpy as np
import random
import coloredlogs
import logging
import os
import sys
import math as m

from src.util import are_arrays_equal_permutation_wise, replace_infs, adjacency_list, sort_adjacency_list
from src.operators.initialisation import backtrack,random_valid_initialise
from src.operators.objective import constrained_fitness

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

new_limit = 5000
sys.setrecursionlimit(new_limit)

def random_recombination(p1, p2,sorted_adjacency, fitness:callable, method=None, seed=None):
    if method == None:
        method = random.randint(0, 3)
    match (method):
        case 0:
            return PMX(p1, p2, seed)[0]
        case 1:
            return order_crossover(p1,p2)[0]
        case 2:
            return cycle_crossover(p1, p2)[0]
        case 3: 
            return order_crossover_with_backtracking(p1, p2, sorted_adjacency, fitness)
        case _:
            return random_recombination(p1, p2, sorted_adjacency, fitness, random.randint(0, 3))


def order_crossover_with_backtracking(parent1: np.ndarray, parent2: np.ndarray, sorted_adjacency, fitness: callable, randomness=True):
    """
    Performs order crossover with backtracking to generate a child from two parents.

    Args:
    - parent1 (np.ndarray): First parent.
    - parent2 (np.ndarray): Second parent.
    - sorted_adjacency: Sorted adjacency information.
    - fitness (callable): Function to calculate fitness.
    - randomness (bool, optional): Flag to determine randomness in completion. Defaults to True.

    Returns:
    - list: Child generated from parents using order crossover with backtracking.

    Note:
    This method creates a child by performing order crossover with backtracking between two parents.
    If backtracking fails, completion is done randomly or greedily based on the 'randomness' flag.
    """

    min_distance = m.floor(len(parent1)/3)
    max_idx = len(parent1) - 1

    parents = [parent1, parent2]

    fitness_values = [fitness(parents[0]), fitness(parents[1])]

    min_fitness_parent_index = np.argmin(fitness_values)
    max_fitness_parent_index = np.argmax(fitness_values)

    parent_with_min_fitness = parents[min_fitness_parent_index]
    parent_with_max_fitness = parents[max_fitness_parent_index]

    idx1 = random.randint(0, max_idx - min_distance)
    idx2 = random.randint(idx1 + min_distance, max_idx)

    start, end = min(idx1, idx2), max(idx1, idx2)

    child = list(parent_with_min_fitness[start:end])

    indices = np.concatenate(
        [range(end, parent_with_min_fitness.size), range(0, end)])
    potential_cities = [parent_with_max_fitness[i] for i in indices]

    result = __backtrack_crossover(
        child, potential_cities, parent_with_min_fitness, parent_with_max_fitness, sorted_adjacency)
    if result is None:
        child = __complete_child_randomly(
            parent1, sorted_adjacency, randomness)

    return child

def order_crossover(p1, p2, seed = None):
    rng = np.random.default_rng(seed=seed)
    idx1, idx2 = np.sort(rng.choice(np.arange(len(p1)), size=2, replace=False))

    offspring1 = np.empty(len(p1), dtype=object)
    offspring2 = np.empty(len(p1), dtype=object)

    offspring1[idx1:idx2] = p1[idx1:idx2]
    offspring2[idx1:idx2] = p2[idx1:idx2]

    offspring1_idx = idx2
    offspring2_idx = idx2
    for idx in np.concatenate([range(idx2, p1.size), range(0, idx2)]):
        if p2[idx] not in offspring1:
            offspring1[offspring1_idx] = p2[idx]
            offspring1_idx = (offspring1_idx + 1) % p1.size
        if p1[idx] not in offspring2:
            offspring2[offspring2_idx] = p1[idx]
            offspring2_idx = (offspring2_idx + 1) % p1.size
    return (offspring1, offspring2)


def __complete_child_randomly(parent1, sorted_adjacency, randomness):
    """
    Completes the child randomly or greedily if backtracking fails.

    Args:
    - parent1: Parent array.
    - sorted_adjacency: Sorted adjacency information.
    - randomness (bool): Flag determining randomness in completion.

    Returns:
    - list: Completed child.

    Note:
    This method completes the child randomly or greedily based on the 'randomness' flag
    if backtracking fails during crossover.
    """

    start, end = sorted(np.random.choice(
        np.arange(len(parent1)), 2, replace=False))
    child = []
    child = list(parent1[start:end])
    result = backtrack(child, len(
        parent1), sorted_adjacency,[], randomness)
    return result

def __backtrack_crossover(child: list, potential_cities: list, parent1: np.ndarray, parent2: np.ndarray, sorted_connected_dict):
    """
    Performs backtracking crossover between parents to generate a child.

    Args:
    - child (list): Partially created child.
    - potential_cities (list): List of potential cities for completion.
    - parent1 (np.ndarray): First parent.
    - parent2 (np.ndarray): Second parent.
    - sorted_connected_dict: Sorted connected dictionary.

    Returns:
    - list or None: Completed child or None if backtracking fails.

    Note:
    This method performs backtracking crossover between parents to create a child.
    If backtracking fails, it returns None.
    """

    if len(child) == parent1.shape[0]:
        last_city = child[-1]
        start_city = child[0]
        if start_city in sorted_connected_dict[last_city]:
            if (are_arrays_equal_permutation_wise(parent1, child) or are_arrays_equal_permutation_wise(parent2, child)):
                return None
            else:
                return child
        else:
            return None

    for city_from_parent in potential_cities:
        if city_from_parent not in child:

            new_start_index = (potential_cities.index(
                city_from_parent) + 1) % len(potential_cities)
            reordered_potential_cities = potential_cities[new_start_index:] + \
                potential_cities[:new_start_index]

            child.append(city_from_parent)

            indx = reordered_potential_cities.index(city_from_parent)
            reordered_potential_cities.pop(indx)

            result = __backtrack_crossover(
                child, reordered_potential_cities, parent1, parent2, sorted_connected_dict)
            if result is not None:
                return result

            child.remove(city_from_parent)
    return None    


def PMX_one_offspring(p1, p2, cut1, cut2,end):
    offspring = np.empty(end, dtype=p1.dtype)

    # Copy the mapping section (middle) from parent1
    offspring[cut1:cut2] = p1[cut1:cut2]

    # copy the rest from parent2 (provided it's not already there)
    for i in np.concatenate([np.arange(0, cut1), np.arange(cut2, end)]):
        candidate = p2[i]
        while candidate in p1[cut1:cut2]:
            candidate = p2[np.where(p1 == candidate)[0][0]]
        offspring[i] = candidate
    return offspring

def PMX(parent1: np.ndarray, parent2: np.ndarray, seed=None):
    lengte = len(parent1)
    """
    parent1 and parent2 are 1D np.array
    """
    rng = np.random.default_rng(seed=seed)

    cutoff_1, cutoff_2 = np.sort(
        rng.choice(np.arange(lengte + 1), size=2, replace=False)
    )
    offspring1 = PMX_one_offspring(parent1, parent2, cutoff_1, cutoff_2, lengte)
    offspring2 = PMX_one_offspring(parent2, parent1, cutoff_1, cutoff_2, lengte)

    return offspring1, offspring2

def cycle_crossover(p1: np.ndarray, p2: np.ndarray):
    s = len(p1)
    # Initialise offspring
    offspring1 = np.empty(s, object)
    offspring2 = np.empty(s, object)
    # While there are uninitialised elements in the offspring:
    # empty is a list of indices where offspring1[index] = None
    while (empty := np.where(offspring1 == None)[0]).size > 0:
        idx = empty[0]
        offspring1[idx] = p1[idx]
        offspring2[idx] = p2[idx]
        idx = np.where(p1 == p2[idx])[0][0]
        while idx != empty[0]:
            offspring1[idx] = p1[idx]
            offspring2[idx] = p2[idx]
            # Find where p1 is equal to the item in p2 at the same index
            idx = np.where(p1 == p2[idx])[0][0]
        # Switch the roles of p1 and p2 (Causes the cycles to be selected alternatingly from each parent)
        p1, p2 = p2, p1
    return (offspring1, offspring2)

if __name__ == "__main__":
    # Example usage
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "../../assingment_data/tour50.csv")
    file = open(csv_path)
    DISTANCE = np.loadtxt(csv_path, delimiter=",")
    file.close()

    DISTANCE = replace_infs(DISTANCE)
    ADJ_DICT = adjacency_list(DISTANCE)
    SORTED_ADJ_DICT = sort_adjacency_list(ADJ_DICT, DISTANCE)

    [parent1, parent2] = random_valid_initialise(
        2, DISTANCE.shape[0], SORTED_ADJ_DICT)
    print("parent1", parent1)
    print("parent2", parent2)

    t1 = time.time()
    for i in range(200):
        offspring = order_crossover_with_backtracking(
            parent1, parent2, SORTED_ADJ_DICT, constrained_fitness, {"DISTANCE_MATRIX": DISTANCE})
        print("Offspring:", offspring)
    t2 = time.time()

    print("time")
    print(t2 - t1)

