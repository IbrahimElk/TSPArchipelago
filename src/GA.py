import matplotlib.pyplot as plt
import coloredlogs
import logging
import numpy as np
import os
import time 

from src.operators.initialisation   import greedy_initialise, random_valid_initialise, random_and_greedy_initialise
from src.operators.mutation         import random_mutate
from src.operators.selection        import select_tourn
from src.operators.recombination    import random_recombination
from src.operators.elimination      import random_elimination
from src.operators.objective        import constrained_fitness
from src.operators.localsearch      import two_opts
from src.util                       import adjacency_list, sort_adjacency_list, replace_infs

from src.visualisations.stat import stat

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


class GA:
    """
    Genetic Algorithm implementation.

    Attributes:
    - _DISTANCE_MATRIX (numpy.ndarray): Matrix containing distances between cities.
    - _LAMBDA (int): Lambda value for GA.
    - _ALPHA: Alpha value for GA.
    - _BETA: Beta value for GA.
    - _MU (int): Mu value for GA.
    - _K_SELECT (int): K selection value for GA.
    - _K_ELIM (int): K elimination value for GA.
    - _sorted_connected_dict: Dictionary containing sorted adjacency matrix.
    - _ITERATIONS: Number of local search iterations.
    - init_arg: Initialization argument for operators.
    - plot_convergence: Boolean indicating if convergence plots should be generated.
    - plot_diversity: Boolean indicating if diversity plots should be generated.
    - printing: Boolean indicating if debug messages should be printed.
    - _DEBUG: Debugging flag.
    - MAX_ITER (int): Maximum number of iterations.
    - _amount_cities: Number of cities.
    - meanObjective: Mean objective value.
    - bestObjective: Best objective value.
    - bestSolution: Best solution.
    - _OFFSPRING (numpy.ndarray): Offspring array.
    - _POPULATION (numpy.matrix): Population matrix.
    """

    def __init__(self, config):
        """
        Initializes the Genetic Algorithm.

        Args:
        - config (dict): A dictionary containing configuration parameters.
        """
        self._config = config
        self.set_config(config)

        if self.printing:
            logger.debug("Initializing GA...")


        self.MAX_ITER = 1500
        self.meanObjective = 99999999999
        self.bestObjective = 99999999999
        self.bestSolution = []
        self._OFFSPRING: np.ndarray = np.empty(
            (self._MU, self._amount_cities), dtype=int)
        self._POPULATION: np.matrix = np.empty(
            (self._LAMBDA, self._amount_cities), dtype=int)
        
        self.stat = stat(self._DISTANCE_MATRIX.copy(),self._sorted_connected_dict.copy(), self.plot_convergence, self.plot_diversity, self._amount_cities)

    def getConfig(self):
        """
        Retrieves the configuration.

        Returns:
        - dict: Configuration parameters.
        """
        logger.debug("Getting configuration...")
        return self._config

    def getMaxIter(self):
        """
        Retrieves the maximum number of iterations.

        Returns:
        - int: Maximum number of iterations.
        """
        logger.debug("Getting maximum iterations...")
        return self.MAX_ITER

    def getPopulation(self):
        """
        Retrieves the current population.

        Returns:
        - numpy.matrix: Current population matrix.
        """
        logger.debug("Getting population...")
        return self._POPULATION.copy()

    def setPopulation(self, new_pop: np.ndarray):
        """
        Sets a new population.

        Args:
        - new_pop (numpy.ndarray): New population to set.
        """
        logger.debug("Setting population...")
        self._POPULATION = new_pop.copy()

    def getBestSolution(self):
        """
        Retrieves the best solution.

        Returns:
        - list: Best solution.
        """
        logger.debug("Getting best solution...")
        return self.bestSolution

    def setBestSolution(self, best):
        """
        Sets the best solution.

        Args:
        - best: Best solution to set.
        """
        logger.debug("Setting best solution...")
        self.bestSolution = best

    def getMeanObjective(self):
        """
        Retrieves the mean objective value.

        Returns:
        - float: Mean objective value.
        """
        logger.debug("Getting mean objective...")
        return self.meanObjective

    def setMeanObjective(self, mean):
        """
        Sets the mean objective value.

        Args:
        - mean (float): Mean objective value to set.
        """
        logger.debug("Setting mean objective...")
        self.meanObjective = mean

    def getBestObjective(self):
        """
        Retrieves the best objective value.

        Returns:
        - float: Best objective value.
        """
        logger.debug("Getting best objective...")
        return self.bestObjective

    def setBestObjective(self, best):
        """
        Sets the best objective.

        Args:
        - best: The best objective value.
        """
        logger.debug("Setting best objective...")
        self.bestObjective = best

    def setIter(self, iterations):
        """
        Sets the maximum number of iterations.

        Args:
        - iterations (int): Maximum number of iterations to set.
        """
        logger.debug("Setting iterations...")
        self.MAX_ITER = iterations

    def setInitPopulation(self):
        """
        Sets the initial population.
        And should be called before optimize()
        """

        logger.debug("Setting initial population...")
        self._POPULATION = self.init(self.init_arg)

    def set_config(self, config: dict):
        """
        Sets configurations for the Genetic Algorithm.

        Args:
        - config (dict): A dictionary containing configuration parameters.
        """

        # general GA parameters.
        self._DISTANCE_MATRIX = config.get('distance_matrix')
        self._LAMBDA = int(config.get('lambdaa'))
        self._ALPHA = config.get('alpha')
        self._BETA = config.get('beta')
        self._MU = int(config.get('mu'))
        self._K_SELECT = int(config.get('Kselect'))
        self._K_ELIM = int(config.get('Kelim'))
        self._sorted_connected_dict = config.get("sorted_adj_matrix")

        # types of operators to perform.
        self.init_arg = config.get('init_arg')
        self.recomb_method = config.get("recomb_method")
        self.local_search_iterations = config.get("local_search_iterations")

        # administratieve zaken
        self.plot_convergence = config.get('plot_convergence')
        self.plot_diversity = config.get('plot_diversity')
        self.printing = config.get('printing')
        self._DEBUG = config.get('debug')
        self._amount_cities = self._DISTANCE_MATRIX.shape[0]

        
    def generate_offsprings(self):
        """
        Generates offspring solutions using selection, recombination, and mutation.
        """
        logger.debug("Generating offsprings...")
        for offspr_idx in range(self._MU):
            t1 = time.time()
            p1 = self.select(self._POPULATION)
            t2 = time.time()
            p2 = self.select(self._POPULATION)
            t3 = time.time()
            offspr = self.recomb(p1, p2, self.recomb_method)
            t4 = time.time()
            self._OFFSPRING[offspr_idx] = self.mutate(offspr)
            t5 = time.time()

            # print("select1")
            # print(t2-t1)
            # print("select2")
            # print(t3-t2)
            # print("recomb")
            # print(t4-t3)
            # print("mutate")
            # print(t5-t4)


    def optimize(self):
        """
        Runs the optimization process using the Genetic Algorithm.

        Returns:
        - tuple: Tuple containing meanObjective, bestObjective, and bestSolution.
        """
        logger.debug("Starting optimization process...")
        self._ITER = 0

        while self._ITER < self.MAX_ITER:
            logger.debug(f"Iteration {self._ITER}:")
            self.generate_offsprings()
            
            for i in range(len(self._POPULATION)):
                self._POPULATION[i] = self.mutate(self._POPULATION[i])

            self._POPULATION = self.elim(self._POPULATION, self._OFFSPRING)
            
            self.meanObjective, best_obj, best_sol = self.metrics(self._POPULATION)
            
            if best_obj < self.bestObjective:
                self.bestObjective, self.bestSolution = best_obj, best_sol 
            
            self.administratieve_zaken(self.bestObjective, self.meanObjective, self.bestSolution)

            self._ITER += 1

        if self.plot_convergence:
            logger.debug("Closing convergence plot...")
            plt.close()

        logger.debug("Optimization process complete.")
        return self.meanObjective, self.bestObjective, self.bestSolution

    def init(self, method: int):
        """
        Initializes the population using the specified method.

        Args:
        - method (int): Method selection for initialization.

        Returns:
        - numpy.ndarray: Initialized population.
        """

        if self._DEBUG:
            logger.debug("init method")
            logger.debug(method)

        match(method):
            case 0:
                return greedy_initialise(self._LAMBDA, self._amount_cities, self._sorted_connected_dict)
            case 1:
                return random_valid_initialise(self._LAMBDA, self._amount_cities, self._sorted_connected_dict)
            case 2:
                return random_and_greedy_initialise(self._LAMBDA,self._amount_cities,self._sorted_connected_dict, 0.9) # 0.8 random

    def select(self, population: np.matrix):
        """
        Selects individuals from the population.

        Args:
        - population (numpy.matrix): Population matrix.

        Returns:
        - np.ndarray: Selected individual.
        """
        return select_tourn(population, self._K_SELECT, self.fitness)

    def recomb(self, p1: np.ndarray, p2: np.ndarray,recomb_method):
        """
        Performs recombination (crossover) between two parents.

        Args:
        - p1 (np.ndarray): First parent.
        - p2 (np.ndarray): Second parent.

        Returns:
        - np.ndarray: Offspring generated from recombination.
        """
        return random_recombination(p1, p2, self._sorted_connected_dict, self.fitness, recomb_method)

    def mutate(self, ind: np.ndarray):
        """
        Performs mutation on an individual.

        Args:
        - ind (np.ndarray): Individual to mutate.

        Returns:
        - np.ndarray: Mutated individual.
        """

        if np.random.random() < self._ALPHA:
            result = random_mutate(ind)
            ind_result = self.local_search(result)
            return ind_result
        return ind

    def local_search(self, result):
        """
        Conducts local search on a result if a random condition is met.

        Args:
        - result: Result to undergo local search.

        Returns:
        - result: Result after local search (if applied).
        """

        if np.random.random() < self._BETA:
            route = two_opts(result, self._DISTANCE_MATRIX,self.local_search_iterations)
            return route

        return result

    def elim(self, pop: np.matrix, offspr: np.matrix):
        """
        Performs elimination from the population based on fitness.

        Args:
        - pop (numpy.matrix): Current population.
        - offspr (numpy.matrix): Offspring population.

        Returns:
        - np.matrix: Updated population after elimination.
        """
        return random_elimination(pop,
                                offspr,
                                self._LAMBDA,
                                self._K_ELIM,
                                self._amount_cities,
                                self._sorted_connected_dict,
                                self.fitness)

    def fitness(self, ind: np.ndarray):
        """
        Calculates the fitness of an individual.

        Args:
        - ind (numpy.ndarray): Individual to evaluate.

        Returns:
        - float: Fitness value of the individual.
        """
        return constrained_fitness(ind, self._DISTANCE_MATRIX)

    def metrics(self, population: list[np.ndarray]) -> tuple[float, int, np.ndarray]:
        """
        Return meanObjective, bestObjective and bestSolution
        """
        sumObjective = 0
        bestObjective = float("inf")
        bestSolution = None
        length = 0
        for item in population:
            temp = self.fitness(item)
            if temp < bestObjective:
                bestSolution = item
                bestObjective = temp
            sumObjective += temp
            length += 1
        return sumObjective / length, bestObjective, bestSolution

    def administratieve_zaken(self, bestObjective, meanObjective, bestSolution):
        """
        Handles administrative tasks such as printing and plot updates.

        Args:
        - bestObjective: Best objective value.
        - meanObjective: Mean objective value.
        - bestSolution: Best solution.
        """
        if self.printing:
            logger.info("bestObjective: " + str(bestObjective))
            logger.info("meanObjective: " + str(meanObjective))
            # logger.info("the best path: " + str(bestSolution))

        self.stat.update_visualization(self._POPULATION, meanObjective, bestObjective, self.fitness)

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "../assingment_data/tour500.csv")
    file = open(csv_path)
    DISTANCE = np.loadtxt(csv_path, delimiter=",")
    file.close()

    sorted_adj_matrix = sort_adjacency_list(
        adjacency_list(DISTANCE), DISTANCE)

    scale = 2
    DISTANCE = replace_infs(DISTANCE,scale)
    # DISTANCE = Graph.preprocessing(DISTANCE)

    MGA = GA({
        "distance_matrix": DISTANCE,
        "sorted_adj_matrix": sorted_adj_matrix,
        "lambdaa":  30,
        "alpha": 0.05,
        "beta": 0.5,
        "mu": 50,
        "Kselect": 3,
        "Kelim": 3,
        "init_arg": 0,  # 2 : random & greedy, 1 : random , 0 : greedy
        "recomb_method": 1,  # 3:random,  2:cycle_crossover, 1: order_crossover, 0: PMX  
        "local_search_iterations": 10,
        "plot_convergence": False,
        "plot_diversity": False,
        "printing": True,
        "debug": False
    })

    t1 = time.time()
    MGA.setInitPopulation()
    t2 = time.time()
    print(t2 - t1)
    MGA.setIter(100)

    result = MGA.optimize()
    logger.warn(result)