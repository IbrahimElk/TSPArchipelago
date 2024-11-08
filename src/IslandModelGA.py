import time
import numpy as np
import copy
import logging
import coloredlogs
import matplotlib.pyplot as plt
import multiprocessing
import os 

from src.util import replace_infs,sort_adjacency_list,adjacency_list,move_zero_to_front_np
from src.GA import GA
from src.operators.localsearch import two_opts
from src.operators.objective import constrained_fitness

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

# TODO: PROBEER TIJD EFFICIENTER TE ZIJN, EN RUN BAYSIAN OPTIMSATION. 
# TODO: begin dan met verslag. 
# beta moet wat hoog zijn.  
# iets bijgeleerd: beter om verschillende soorten recombinatie operatoren te hebben dan 1 goede die sterk diversification prevention heeft. 

# TODO: voor 500 keer te runnen, check met optimale parameters, met 0,5 proportie of met 0.9 beter is. 

class IslandModelGA:
    def __init__(self,
                 max_iterations: int,
                 config_lijst: list[dict]
                 ):

        self.num_islands: int = len(config_lijst)
        self.islands: list[GA] = [None] * self.num_islands

        self.popMeanObjective: int = 99999999999
        self.popBestObjective: int = 99999999999
        self.popBestSolution: list = []

        for i in range(self.num_islands):
            island: GA = GA(config_lijst[i])
            # After a fixed number of iterations (e.g., 25–150), migrate populations.
            island.setIter(max_iterations)
            island.setInitPopulation()
            self.islands[i] = island

    # CORE FUNCTION
    def run_one_iteration(self):
        logger.warning("running one iteration")
        mean_objectives, best_objectives, best_solutions = self.parrallel_population_computation()

        logger.warning("running migration procedure")
        self.migrate()

        logger.warning("running finding best among subpopulation procedure")
        self.find_best_solutions_among_subpopulations(mean_objectives, best_objectives, best_solutions)

        logger.warning("finshed running one iteration")
        return self.popMeanObjective, self.popBestObjective, np.array(self.popBestSolution)

    def optimize_island(self, shared_data, index, config_lijst, max_iterations):
        island_data = shared_data[index]

        island = GA(config_lijst)
        island.setIter(max_iterations)

        island.setPopulation(np.array(island_data['population']))
        island.setMeanObjective(float(island_data['meanObj'].value))
        island.setBestObjective(float(island_data['bestObj'].value))
        island.setBestSolution(np.array(island_data['bestSol']))

        island.optimize()

        island_data['population'][:] = island.getPopulation()
        island_data['meanObj'].value = island.getMeanObjective()
        island_data['bestObj'].value = island.getBestObjective()
        island_data['bestSol'][:] = island.getBestSolution()

    def parrallel_population_computation(self) -> tuple:
        manager = multiprocessing.Manager()

        shared_data = []
        for i in range(self.num_islands):
            island_data = manager.dict()
            island_data['population'] = manager.list(self.islands[i].getPopulation())
            island_data['meanObj'] = manager.Value('d', self.islands[i].getMeanObjective())
            island_data['bestObj'] = manager.Value('d', self.islands[i].getBestObjective())
            island_data['bestSol'] = manager.list(self.islands[i].getBestSolution())
            shared_data.append(island_data)

        processes = []
        for i in range(self.num_islands):
            args = (shared_data, i, self.islands[i].getConfig(), self.islands[i].getMaxIter())
            process = multiprocessing.Process(target=self.optimize_island, args=args)
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        for index in range(self.num_islands):
            island_data = shared_data[index]

            self.islands[i].setPopulation(np.array(island_data['population']))
            self.islands[i].setMeanObjective(float(island_data['meanObj'].value))
            self.islands[i].setBestObjective(float(island_data['bestObj'].value))
            self.islands[i].setBestSolution(np.array(island_data['bestSol']))

        mean_objectives = [data['meanObj'].value for data in shared_data]
        best_objectives = [data['bestObj'].value for data in shared_data]
        best_solutions = [list(data['bestSol']) for data in shared_data]

        return mean_objectives[:], best_objectives[:], best_solutions

    def find_best_solutions_among_subpopulations(self, mean_objectives: list, best_objectives: list, best_solutions: list):
        for isle in range(self.num_islands):
            currentObjective = best_objectives[isle]
            if currentObjective < self.popBestObjective:
                self.popMeanObjective = mean_objectives[isle]
                self.popBestObjective = currentObjective
                self.popBestSolution = best_solutions[isle]

    def migrate(self):
        for i in range(self.num_islands):
            try : 
                my_island: GA = self.islands[i]

                # Commonly used rates range from 5% to 20% of the population size.
                #  it is better to exchange a small number of solutions between subpopulations –usually 2–5. 
                migration_count: int = int(my_island._LAMBDA * 0.05)            
                migration_target: np.ndarray = np.random.choice([x for x in range(self.num_islands) if x != i])
                
                target_island: GA = self.islands[migration_target]
                individuals_to_migrate: np.ndarray = np.random.choice(range(my_island._LAMBDA), migration_count, replace=False)

                my_pop: np.ndarray = my_island.getPopulation()

                individuals: np.ndarray = my_pop[individuals_to_migrate]
                immigrants: np.ndarray = copy.deepcopy(individuals)

                # kan groter zijn dan lambdaa, maar zou geen groot probleem mogen zijn peisk.
                new_list: np.ndarray = np.concatenate((target_island.getPopulation(), immigrants))
                target_island.setPopulation(new_list)

                pop: np.ndarray = my_island.getPopulation()
                filtered_array: np.ndarray = np.delete(pop, individuals_to_migrate, axis=0)
                my_island.setPopulation(filtered_array)
            except Exception as e: 
                logger.error(f"migrating failed in {i}th population")
                logger.error(e)
                continue


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "../assingment_data/tour1000.csv")
    file = open(csv_path)
    DISTANCE = np.loadtxt(csv_path, delimiter=",")
    file.close()

    sorted_adj_matrix = sort_adjacency_list(
        adjacency_list(DISTANCE), DISTANCE)

    scale = 2
    DISTANCE = replace_infs(DISTANCE,scale)
    
    random_configurations = [{
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
    
    # [{'lambdaa': 30, 'alpha': 0.5, 'beta': 0.3, 'mu': 50, 'Kselect': 7, 'Kelim': 5, 'init_arg': 0, 'recomb_method': 2, 'local_search_iterations': 1}, 
    #  {'lambdaa': 30, 'alpha': 1, 'beta': 1, 'mu': 100, 'Kselect': 5, 'Kelim': 3, 'init_arg': 2, 'recomb_method': 4, 'local_search_iterations': 20}

    # kheb migration rate naar 5% gebracht, doe terug 20% als niet goed gaat .
    # TODO: voor 500 keer te runnen, check met optimale parameters, met 0,5 proportie of met 0.9 beter is. 

    t0 = time.time()

    model = IslandModelGA(10, random_configurations)

    t1 = time.time()

    logger.error("initialisatie tijd")
    logger.error(t1 - t0)

    mymeanObjective = 99999999999999
    mybestObjective = 99999999999999
    mybestSolution = np.array([1, 2, 3, 4, 5])

    timeLeft = 300 #TODO: run this 500 times with 60 seconds
    mean_objective_values = []
    best_objective_values = []
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

        mean_objective_values.append(meanObjective)
        best_objective_values.append(bestObjective)

    print(mybestObjective)

    print(len(mean_objective_values))
    print(len(best_objective_values))
    
    plt.clf()
    fig, axs = plt.subplots()
    axs.plot(range(len(mean_objective_values)), mean_objective_values,
            label='Mean Objective', color='orange')
    axs.plot(range(len(best_objective_values)), best_objective_values,
            label='Best Objective', color='green')
    axs.set_xlabel('Iterations')
    axs.set_ylabel('Objective Values')
    axs.set_title('Convergence Plot')
    axs.legend()

    filename = f'plotting.svg'
    fig.savefig(filename, format='svg')        
    plt.close(fig) 

