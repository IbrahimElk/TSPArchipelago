import tsplib95
import numpy as np
import os
import pandas as pd
import sys
# eenmalig gebruik. 

def load_and_extract_tsplib(instance_file, output_folder):
    instance = tsplib95.load(instance_file)

    num_nodes = instance.dimension

    distance_matrix = np.zeros((num_nodes, num_nodes))

    for edge in instance.get_edges():
        node1, node2 = edge
        distance = instance.get_weight(node1, node2)
        distance_matrix[node1 - 1][node2 - 1] = distance 

    instance_name = instance.name
    csv_filename = os.path.join(output_folder, f"{instance_name}.csv")
    df = pd.DataFrame(distance_matrix)
    df.to_csv(csv_filename, index=False)

    return instance_name, num_nodes, distance_matrix

if __name__ == "__main__":

    hyperparameters_folder = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.join(hyperparameters_folder, "../../data/")
    symmetric_instance_folder = os.path.join(root_dir, "ALL_tsp")
    asymmetric_instance_folder = os.path.join(root_dir, "ALL_atsp")
    output_folder = os.path.join(root_dir, "distance_matrices")
    os.makedirs(output_folder, exist_ok=True)

    asymmetric_instance_files = [
        os.path.join(asymmetric_instance_folder, filename) for filename in os.listdir(asymmetric_instance_folder)
    ]

    for asymmetric_instance_file in asymmetric_instance_files:
        asymmetric_instance_name, asymmetric_num_nodes, asymmetric_distance_matrix = load_and_extract_tsplib(asymmetric_instance_file, output_folder)

    symmetric_instance_files = [
        os.path.join(symmetric_instance_folder, filename) for filename in os.listdir(symmetric_instance_folder)
    ]

    for symmetric_instance_file in symmetric_instance_files: 
        symmetric_instance_name, symmetric_num_nodes, symmetric_distance_matrix = load_and_extract_tsplib(symmetric_instance_file, output_folder)
        