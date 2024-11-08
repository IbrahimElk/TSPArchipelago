import numpy as np



def constrained_fitness(individual: list[int], DISTANCE_MATRIX: np.ndarray) -> np.float64:
    """
    Calculates the fitness of an individual based on the given distance matrix.

    Args:
    - individual (list[int]): List representing the individual's path or route.
    - DISTANCE_MATRIX (numpy.ndarray): Matrix containing distances between cities.

    Returns:
    - np.float64: Total cost or fitness of the individual's path.

    Notes:
    This function may return "inf" if there exists a connection in the individual's path with a weight of "inf".
    The function assumes that the `individual` has the same length as the first or second dimension of the `distance matrix`.
    """
    start = individual[-1]
    totalCost = 0

    for destination in individual:
        cost = DISTANCE_MATRIX[start][destination]
        if cost == float("inf"):
            return cost
        totalCost += cost
        start = destination
    return totalCost
