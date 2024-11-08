import numpy as np
import matplotlib.pyplot as plt
import coloredlogs
import logging
import numpy as np
import math as m
import random
import copy
import time 
import multiprocessing
import sys
import uuid
import warnings

from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from src.Reporter import Reporter
from src.util import sort_adjacency_list , adjacency_list, replace_infs, move_zero_to_front_np
from src.IslandModelGA import IslandModelGA

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

new_limit = 5000
sys.setrecursionlimit(new_limit)

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

class Run:
    def __init__(self,counter):
        self.reporter = Reporter(self.__class__.__name__+ str(counter))

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        DISTANCE = np.loadtxt(file, delimiter=",")
        file.close()
        sorted_adj_matrix = sort_adjacency_list(
            adjacency_list(DISTANCE), DISTANCE)

        scale = 2
        DISTANCE = replace_infs(DISTANCE,scale)

        config = [{
            "distance_matrix": DISTANCE,
            "sorted_adj_matrix": sorted_adj_matrix,
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
            "distance_matrix": DISTANCE,
            "sorted_adj_matrix": sorted_adj_matrix,
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

        model = IslandModelGA(10, config)

        mymeanObjective = 99999999999999
        mybestObjective = 99999999999999
        mybestSolution = np.array([1, 2, 3, 4, 5])

        timeLeft = 300
        while (timeLeft > 5):
            
            logger.warning("timeLeft")
            logger.warning(timeLeft)
            
            meanObjective, bestObjective, bestSolution = model.run_one_iteration()

            if bestObjective < mybestObjective:
                mymeanObjective = meanObjective
                mybestObjective = bestObjective
                mybestSolution = move_zero_to_front_np(bestSolution)


            bestSolution = move_zero_to_front_np(bestSolution)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            timeLeft = self.reporter.report(mymeanObjective, mybestObjective, mybestSolution)

        logger.info("meanObjective, bestObjective, bestSolution")
        logger.info(mymeanObjective)
        logger.info(mybestObjective)
        logger.info(mybestSolution)
        return 0
