import numpy as np
from numba import njit

def adjacency_list(distance_matrix: np.ndarray) -> list:
    """
    Creates an adjacency list representing the connections between nodes based on the distance matrix.
    The distance_matrix contains "inf" for nodes that are not connected.

    Returns:
        list: An adjacency list representing the graph.

    For example:

    graph = [
        [1, 2, 3],  # Connected cities for node 0
        [0, 3],     # Connected cities for node 1
        [0, 3],     # Connected cities for node 2
        [1, 2]      # Connected cities for node 3
    ]
    """
    num_cities: int = distance_matrix.shape[0]
    graph: list[list[int]] = [[] for _ in range(num_cities)]

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j and distance_matrix[i, j] != np.inf:
                graph[i].append(j)
    return graph

def sort_adjacency_list(adjacency_list: np.ndarray, distance_matrix: np.ndarray) -> list:
    """
    Sorts the given adjacency list based on the distance matrix.

    Args:
        adjacency_list (list): An adjacency list representing the graph.
        distance_matrix (np.ndarray): The distance matrix containing distances between nodes.

    Returns:
        list: A sorted adjacency list where inner lists are sorted by distance.
    """
    sorted_adjacency_list = [[] for _ in range(len(adjacency_list))]

    for node, connections in enumerate(adjacency_list):
        distances = [(neighbor, distance_matrix[node, neighbor]) for neighbor in connections]
        distances.sort(key=lambda tuuple: tuuple[1])
        sorted_adjacency_list[node] = [ neighbor for neighbor, _ in distances]

    return sorted_adjacency_list



def preprocessing(DISTANCE_MATRIX: np.matrix):
    """
    returns a normalization of the distances of the given distance matrix.
    """
    DM = DISTANCE_MATRIX.copy()

    min_distance = np.min(DM[DM != np.inf])
    max_distance = np.max(DM[DM != np.inf])

    DM[DM != np.inf] = (DM[DM != np.inf] - min_distance) / \
        (max_distance - min_distance)

    return DM

def replace_infs(DISTANCE_MATRIX: np.matrix,scale:int):
    """
    replaces all 'infs' with maximum value of distance matrix, assuming not all values are infs.
    """
    DM = DISTANCE_MATRIX.copy()

    DM[DM == np.inf] = np.nan
    max_non_inf = np.nanmax(DM)

    # het dubbele van max wordt in de inf's geplaatst.
    DM = np.nan_to_num(DM, nan=scale*max_non_inf)

    return DM

@njit
def are_arrays_equal_permutation_wise(arr1, arr2):
    if len(arr1) != len(arr2):
        return False

    arr1_concat = ''.join(map(str, arr1)) * 2
    arr2_str = ''.join(map(str, arr2))

    return arr2_str in arr1_concat

def move_zero_to_front_np(arr):
    if not np.any(arr == 0):
        return arr

    zero_index = np.where(arr == 0)[0][0]
    rearranged_arr = np.roll(arr, -zero_index)

    return rearranged_arr

if __name__ == "__main__":
    print(are_arrays_equal_permutation_wise([0, 1, 2], [1, 2, 0]))
