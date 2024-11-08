import os
import random
import coloredlogs
import logging

import numpy as np
import math as m

from src.operators.initialisation import random_valid_initialise
from src.operators.objective import constrained_fitness
from src.util import adjacency_list, sort_adjacency_list, replace_infs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def random_mutate(ind):
    """
    Randomly selects a mutation method and applies the mutation to the individual.

    Args:
    - ind: Individual to mutate.

    Returns:
    - list: Mutated individual.

    Note:
    Randomly selects between 'random_inversion' and 'random_scramble' mutation methods.
    """
    method = random.randint(3, 4)
    match (method):
        case 3:
            return random_inversion(ind)
        case 4:
            return random_scramble(ind)

def random_inversion(path: list[int]):
    """
    Applies the random inversion mutation to the given path.

    Args:
    - path (list[int]): Path to mutate.

    Returns:
    - list: Mutated path after applying random inversion.

    Note:
    Randomly selects a segment of the path and reverses it.
    """
    min_distance = m.floor(len(path)/3)
    max_idx = len(path) - 1

    idx1 = random.randint(0, max_idx - min_distance)
    idx2 = random.randint(idx1 + min_distance, max_idx)

    pathcopy = list(path)
    start, end = min(idx1, idx2), max(idx1, idx2)

    pathcopy[start:end + 1] = pathcopy[start:end + 1][::-1]
    return pathcopy

def random_scramble(path: list[int]):
    """
    Applies the random scramble mutation to the given path.

    Args:
    - path (list[int]): Path to mutate.

    Returns:
    - list: Mutated path after applying random scramble.

    Note:
    Randomly selects a segment of the path and shuffles the elements within that segment.
    """
    min_distance = m.floor(len(path)/3)
    max_idx = len(path) - 1

    idx1 = random.randint(0, max_idx - min_distance)
    idx2 = random.randint(idx1 + min_distance, max_idx)
    pathcopy = list(path)

    start, end = min(idx1, idx2), max(idx1, idx2)

    segment = pathcopy[start:end + 1]
    random.shuffle(segment)
    pathcopy[start:end + 1] = segment

    return pathcopy
    

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "../../assingment_data/tour50.csv")
    file = open(csv_path)
    distance_matrix = np.loadtxt(csv_path, delimiter=",")
    file.close()

    DISTNACE = replace_infs(distance_matrix)
    LEUK = adjacency_list(DISTNACE)
    IDD = sort_adjacency_list(LEUK, distance_matrix)

    ind = random_valid_initialise(1, distance_matrix.shape[0], IDD)[0]
    logger.debug("ind")
    logger.debug(ind)

    n1 = random_inversion(ind, constrained_fitness, {
                                   "DISTANCE_MATRIX": distance_matrix})
    n4 = random_mutate(ind, constrained_fitness, {
                                "DISTANCE_MATRIX": distance_matrix})
    n5 = random_scramble(ind, constrained_fitness, {
                                  "DISTANCE_MATRIX": distance_matrix})
    logger.debug("n1")
    logger.debug(n1)
    logger.debug("n4")
    logger.debug(n4)
    logger.debug("n5")
    logger.debug(n5)

    for i in range(1500):

        n1 = random_inversion(ind, constrained_fitness, {
                                       "DISTANCE_MATRIX": distance_matrix})
        n4 = random_mutate(ind, constrained_fitness, {
                                    "DISTANCE_MATRIX": distance_matrix})
        n5 = random_scramble(ind, constrained_fitness, {
                                      "DISTANCE_MATRIX": distance_matrix})

        logger.debug("n1")
        logger.debug(np.all(n1 == ind))
        logger.debug("n4")
        logger.debug(np.all(n4 == ind))
        logger.debug("n5")
        logger.debug(np.all(n5 == ind))
