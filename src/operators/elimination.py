import numpy as np
import coloredlogs
import logging
from src.util import are_arrays_equal_permutation_wise,move_zero_to_front_np
import random

from src.operators.initialisation import random_valid_initialise, greedy_initialise

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

def random_elimination(population: list[np.ndarray],
                offspring: list[np.ndarray],
                lambd: int,
                K: int,
                amount_cities: int,
                sorted_connected_dict: dict,
                fitness: callable,
                method = None):
    """
    Randomly selects a mutation method and applies the mutation to the individual.

    Args:
    - ind: Individual to mutate.
    - fitness (callable): Function to calculate fitness.

    Returns:
    - list: Mutated individual.

    Note:
    Randomly selects between 'random_inversion' and 'random_scramble' mutation methods.
    """
    if method == None:
        method = random.randint(0, 1)
    match (method):
        case 0:
            return ktourn_elimination(population,
                offspring,
                lambd,
                K,
                amount_cities,
                sorted_connected_dict,
                fitness)
        case 1:
            return lmabda_mu_elimination(population,
                offspring,
                lambd,
                K,
                amount_cities,
                sorted_connected_dict,
                fitness)
                
def ktourn_elimination(population: list[np.ndarray],
                offspring: list[np.ndarray],
                lambd: int,
                K: int,
                amount_cities: int,
                sorted_connected_dict: dict,
                fitness: callable):
    """
    Performs elimination from the combined population and offspring based on fitness.

    Args:
    - distance_matrix (np.ndarray): Matrix containing distances between cities.
    - population (list): List of individuals in the population.
    - offspring (list): List of individuals in the offspring.
    - lambd (int): Lambda value for elimination.
    - K (int): Number of individuals to select for comparison.
    - amount_cities (int): Number of cities.
    - sorted_connected_dict (dict): Dictionary containing sorted adjacency matrix.
    - fitness (callable): Function to calculate fitness.

    Returns:
    - np.ndarray: Updated population after elimination.
    """
    combined: list = np.concatenate((population, offspring), axis=0).tolist()

    result_after_elim = []
    counter = 0
    iter = 0
    while counter < lambd and iter < 1.5 * lambd:
        iter += 1

        selected = np.random.choice(range(len(combined)), K, replace=False)
        selected_fitness = [fitness(combined[i]) for i in selected]
        final_selected = combined[selected[selected_fitness.index(
            min(selected_fitness))]]
        
        if final_selected not in result_after_elim:
            result_after_elim.append(final_selected)
            counter += 1
        combined.remove(final_selected)

    if counter < lambd:
        logger.warning("NEW POPULATION CREATED IN ktourn_elimination")
        resterende_aantal = lambd - counter
        new_pop = greedy_initialise(
            resterende_aantal, amount_cities, sorted_connected_dict)
        result_after_elim.extend(new_pop)

    result_after_elim = np.array(result_after_elim)
    return result_after_elim

def lmabda_mu_elimination(population: list[np.ndarray],
                offspring: list[np.ndarray],
                lambd: int,
                K: int,
                amount_cities: int,
                sorted_connected_dict: dict,
                fitness: callable):
    combined: list = np.concatenate((population, offspring), axis=0).tolist()
    combined = sorted(combined, key=lambda ind: fitness(ind))

    stop = len(combined)
    result_after_elim = []
    counter = 0
    iter = 0
    while counter < lambd and iter < stop:
        ind = combined[iter]
        if ind not in result_after_elim:
            result_after_elim.append(ind)
            counter += 1
        iter += 1
    if counter < lambd:
        logger.warning("NEW POPULATION CREATED IN lmabda_mu_elimination")
        resterende_aantal = lambd - counter
        new_pop = greedy_initialise(
            resterende_aantal, amount_cities, sorted_connected_dict)
        result_after_elim.extend(new_pop)

    result_after_elim = np.array(result_after_elim)
    return result_after_elim

# def similar_path_in_pop(final_selected, result_after_elim):
#     """
#     Checks for similar paths in the population after elimination.

#     Args:
#     - final_selected: Individual selected after elimination.
#     - result_after_elim (list): List of individuals after elimination.

#     Returns:
#     - bool: True if a similar path exists, False otherwise.
#     """
#     for ind in result_after_elim:
#         if  are_arrays_equal_permutation_wise(ind, final_selected):
#             return True
#     return False


