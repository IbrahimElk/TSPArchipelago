import numpy as np
import os
import json
import coloredlogs
import logging
import matplotlib.pyplot as plt
from src.util import sort_adjacency_list, adjacency_list, replace_infs,move_zero_to_front_np
from src.IslandModelGA import IslandModelGA
import time

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG', logger=logger)

    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, "../../data/optimal_solutions.json")
    with open(json_path, 'r') as file:
        data = json.load(file)

    distance_matrices_dataframes = []
    csv_folder = os.path.dirname(os.path.abspath(
        __file__)) + "/../../data/distance_matrices"

    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith(".csv"):
            f = os.path.join(csv_folder, csv_file)
            file = open(f)
            distance_matrix = np.loadtxt(f, delimiter=",")
            file.close()
            index = csv_file.find(".")
            result = csv_file[:index]

            sorted_adj_matrix = sort_adjacency_list(
                adjacency_list(distance_matrix), distance_matrix)
            DISTANCES = replace_infs(distance_matrix, 3)

            distance_matrices_dataframes.append(
                (DISTANCES, sorted_adj_matrix, data[result]))

    filtered_distance_matrices = []
    dimensions = []
    for df in distance_matrices_dataframes:
        if df[0].shape[0] <= 1100 and df[0].shape[0] >= 190:
            filtered_distance_matrices.append(df)
            dimensions.append(df[0].shape[0])

    print(dimensions)

    error = []
    for fd in filtered_distance_matrices:
        config = [{
                "distance_matrix": fd[0],
                "sorted_adj_matrix": fd[1],
                "lambdaa":  50,
                "alpha": 0.5,
                "beta": 0.5,
                "mu": 50,
                "Kselect": 3,
                "Kelim": 5,
                "init_arg": 0,  # 2 : random & greedy, 1 : random , 0 : greedy
                "recomb_method": 4,  # 3:random,  2:cycle_crossover, 1: order_crossover, 0: PMX  
                "local_search_iterations": 1,
                "plot_convergence": False,
                "plot_diversity": False,
                "printing": False,
                "debug": False
            },{
                "distance_matrix": fd[0],
                "sorted_adj_matrix": fd[1],
                "lambdaa":  30,
                "alpha": 0.1,
                "beta": 0.1,
                "mu": 100,
                "Kselect": 5,
                "Kelim": 7,
                "init_arg": 2,
                "recomb_method": 2,  # 4:random, 3:order_backtracking, 2:cycle_crossover, 1: order_crossover, 0: PMX  
                "local_search_iterations": 20,
                "plot_convergence": False,
                "plot_diversity": False,
                "printing": False,
                "debug": False
            }]
        
        t0 = time.time()

        model = IslandModelGA(10, config)

        t1 = time.time()

        logger.error("initialisatie tijd")
        logger.error(t1 - t0)

        mymeanObjective = 99999999999999
        mybestObjective = 99999999999999
        mybestSolution = np.array([1, 2, 3, 4, 5])

        timeLeft = 300
        while (timeLeft > 5):
            t3 = time.time()
            meanObjective, bestObjective, bestSolution = model.run_one_iteration()

            if bestObjective < mybestObjective:
                mymeanObjective = meanObjective
                mybestObjective = bestObjective
                mybestSolution  = bestSolution

            bestSolution = move_zero_to_front_np(bestSolution)
            t2 = time.time()

            logger.error("1 iteratie duurde zo lang:")
            logger.error(t2 - t3)

            timeLeft = timeLeft - (t2 - t3)

            logger.error("timeLeft")
            logger.error(timeLeft)

        error.append(mybestObjective - fd[2])

    zipped_data = list(zip(dimensions, error))

    sorted_data = sorted(zipped_data, key=lambda x: x[0])

    sorted_dimensions, sorted_error = zip(*sorted_data)

    print(dimensions, error)
    print(sorted_dimensions, sorted_error)

    plt.clf()
    _, axs = plt.subplots()
    axs.scatter(sorted_dimensions, sorted_error, color='blue', label=f'absolute error for each problem size')

    plt.title(f'absolute error Plot')
    plt.xlabel('problem size')
    plt.ylabel('absolute error')
    plt.legend()
    plt.grid(True)
    
    filename = f'plotting_error.svg'
    plt.savefig(filename, format='svg')
    plt.close()
