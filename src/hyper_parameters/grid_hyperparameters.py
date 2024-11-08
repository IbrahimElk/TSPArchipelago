# import os
# import sys

# from sklearn.base import BaseEstimator
# import numpy as np
# import os
# import json

# from src.util import Graph
# from src.GA import GA
# from sklearn.model_selection import ParameterGrid
# # from src.hyper_parameters.baysian_hyperparameters import CustomEstimator

# if __name__ == "__main__":
#     script_dir = os.path.dirname(__file__)
#     json_path = os.path.join(script_dir, "../../data/optimal_solutions.json")
#     with open(json_path, 'r') as file:
#         data = json.load(file)

#     distance_matrices_dataframes = []
#     csv_folder = os.path.dirname(os.path.abspath(__file__)) + "/../../data/distance_matrices"
#     for csv_file in os.listdir(csv_folder):
#         if csv_file.endswith(".csv"):
#             f = os.path.join(csv_folder, csv_file)
#             file = open(f)
#             distance_matrix = np.loadtxt(f, delimiter=",")
#             file.close()
#             print(csv_file)
#             index = csv_file.find(".")
#             result = csv_file[:index]
#             distance_matrices_dataframes.append((distance_matrix, data[result]))

#     # print(len(distance_matrices_dataframes))
#     # for matr in distance_matrices_dataframes:
#     #     print(np.array(matr).shape)
    
#     filtered_distance_matrices = []
#     dimensions = []
#     for df in distance_matrices_dataframes:
#         if df[0].shape[0] <= 200 and df[0].shape[0] >= 100:
#             filtered_distance_matrices.append(df)
#             dimensions.append(df[0].shape[0])

#     objective_function = CustomEstimator(filtered_distance_matrices)
#     results = []

#     hyperparameter_grid = {
#         "lambdaa": [50, 100, 200, 350, 500],
#         "alpha": [0.1, 0.25, 0.5, 0.75, 1.0],
#         "mu": [100, 200, 350, 500],
#         "K": [3, 5, 7],
#         "init_arg": [0],
#         "select_arg": [0],
#         "recomb_arg": [0],
#         "mutate_arg": [0],
#         "repair_arg": [0],
#         "elim_arg": [0],
#         # "obj_arg": [0, 1],
#         "obj_arg": [0],
#         "plot": [False],
#         "printing": [False],
#         "debug": [False],
#     }

#     grid = list(ParameterGrid(hyperparameter_grid))
    
#     with open('hyperparameter_grid.json', 'w') as json_file:
#         json.dump(grid, json_file)

#     best_hyperparameters = None
#     best_score = float('inf')

#     for hyperparameters in grid:
#         score = objective_function.fit(hyperparameters)
#         if score < best_score:
#             best_score = score
#             best_hyperparameters = hyperparameters

#     print("Best Hyperparameters:", best_hyperparameters)
#     print("Optimal Score:", -best_score)

#     result = {
#         'hyperparameters': best_hyperparameters,
#         'optimal_score': -best_score
#     }
    
# # reporter 10 sec 
# # Best Hyperparameters: {'K': 3, 'alpha': 0.25, 'debug': False, 'elim_arg': 0, 'init_arg': 0, 'lambdaa': 200, 'mu': 100, 'mutate_arg': 0, 'obj_arg': 0, 'plot': False, 'printing': False, 'recomb_arg': 0, 'repair_arg': 0, 'select_arg': 0}
# # Optimal Score: -0.038314786936660294