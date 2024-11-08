import uuid
import numpy as np
import matplotlib.pyplot as plt

import os

from src.util import adjacency_list
from src.operators.objective import constrained_fitness

import coloredlogs
import logging

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


class stat:
    _COUNTER = 0
    
    def __init__(self,distance_matrix, adj_matrix, plot_convergence, plot_diversity, amount_cities:int) -> None:
        self.DISTANCE = distance_matrix
        self.ADJ_MATRIX = adj_matrix
        self.id = uuid.uuid4()
        logger.debug(self.id)
        self.fileidx = f"figures/{self.id}"    

        self.plot_convergence = plot_convergence
        self.plot_diversity = plot_diversity

        self._amount_cities = amount_cities

        self.unique_individuals = []
        self.unique_individuals_perm_wise = []
        self.dictinct_fitness = []
        self.amount_of_infs = []
        self.amount_of_valid = []

        self.neighbor_counts = {}
        self.non_neighbor_counts = {}

        for city in range(self._amount_cities):
            self.neighbor_counts[city] = {}
            self.non_neighbor_counts[city] = {}
            for another_city in range(self._amount_cities):
                self.neighbor_counts[city][another_city] = 0
                self.non_neighbor_counts[city][another_city] = 0

        self.mean_objective_values = []
        self.best_objective_values = []

    def exploration_measure(population, neighbor_counts, non_neighbor_counts, sorted_connected_dict):
        apath = population[0]
        alength = len(apath)

        for ind in population:
            for city_idx in range(alength):
                city = ind[city_idx]
                neighbors = sorted_connected_dict[city]

                prev_city = ind[city_idx - 1]

                if prev_city in neighbors:

                    neighbor_counts[city][prev_city] += 1
                else:
                    non_neighbor_counts[city][prev_city] += 1

                next_city = ind[(city_idx + 1 )% alength]

                if next_city in neighbors:
                    neighbor_counts[city][next_city] += 1
                else:
                    non_neighbor_counts[city][next_city] += 1
        
        return neighbor_counts, non_neighbor_counts

    def is_valid_path(self,population):
        inf = np.max(self.DISTANCE) 
        counter = 0
        for path in population:
            for i in range(len(path) - 1):
                if self.DISTANCE[path[i]][path[i + 1]] == inf:
                    counter += 1
                    break
            if self.DISTANCE[path[-1]][path[0]] == inf:
                counter += 1
                continue
        return counter

    def count_equal_permutation_wise(population):
        permutations_count = 0
        for i, array1 in enumerate(population):
            array1_str = ''.join(map(str, array1))
            for j in range(i + 1, len(population)):
                array2_str = ''.join(map(str, population[j])) * 2
                if array1_str in array2_str:
                    permutations_count += 1
        return permutations_count

    def calculate_amount_duplicates(population: list):
        my_pop = np.array(population)
        _, counts = np.unique(my_pop, axis=0, return_counts=True)
        duplicate_indices = np.where(counts > 1)[0]
        num_duplicates = len(duplicate_indices)
        return num_duplicates

    def calculate_amount_different_fitness_values(self, population: list, fitness: callable):
        values = []
        for ind in population:
            value = fitness(ind)
            if value not in values:
                values.append(value)

        return len(population) - len(values)

    def update_visualization(self, population, mean_obj, best_obj, fitness: callable):
        # calculating the to be measured data:
        if self.plot_convergence:
            self.mean_objective_values.append(mean_obj)
            self.best_objective_values.append(best_obj)
        
        if self.plot_diversity:

            distinct_fitness_values = self.calculate_amount_different_fitness_values(population, fitness)
            self.dictinct_fitness.append(distinct_fitness_values)

            amount_dinstinct = stat.calculate_amount_duplicates(population)
            self.unique_individuals.append(amount_dinstinct)

            amount_same = stat.count_equal_permutation_wise(population)
            self.unique_individuals_perm_wise.append(amount_same)

            amount_valid = self.is_valid_path(population)
            self.amount_of_valid.append(amount_valid)

            stat.exploration_measure(population,self.neighbor_counts,self.non_neighbor_counts, self.ADJ_MATRIX)
            
        # saving the data to plots:
        if self.plot_convergence:
            self.update_convergenceplot()
        if self.plot_diversity:
            self.update_histogram()
            self.update_diversityplot()

        self._COUNTER += 1
        return
    
    def update_histogram(self):
        if self.plot_diversity:
            for city in range(5):
                if not os.path.exists(f'{self.fileidx}/diversity/histogram/city{city}'):
                    os.makedirs(f'{self.fileidx}/diversity/histogram/city{city}')
                plt.clf()
                _, axs = plt.subplots()
                axs.plot(self.neighbor_counts[city].keys(), self.neighbor_counts[city].values(),linestyle='-', color='blue', label=f'Neighbors of {city}')
                axs.plot(self.non_neighbor_counts[city].keys(), self.non_neighbor_counts[city].values(),linestyle='-', color='red', label=f'Non-Neighbors of {city}')

                plt.title(f'Neighbor and Non-Neighbor Counts for City {city}')
                plt.xlabel('Neighboring City')
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True)
                
                filename = f'{self.fileidx}/diversity/histogram/city{city}/histogram_figure_{self._COUNTER}.svg' 
                plt.savefig(filename,format='svg')
                plt.close()

    def update_diversityplot(self):
        if self.plot_diversity:
            if not os.path.exists(f'{self.fileidx}/diversity/plot_diversity'):
                os.makedirs(f'{self.fileidx}/diversity/plot_diversity')

            plt.clf()
            figs, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].plot(range(self._COUNTER + 1), self.dictinct_fitness, marker='o',
                            label='amount of same fitness values in population', color='blue')
            axs[1].plot(range(self._COUNTER + 1), self.unique_individuals, marker='o',
                            label='amount of duplicate paths in population', color='green')
            axs[2].plot(range(self._COUNTER + 1), self.amount_of_valid, marker='o',
                            label=f'amount of valid individuals in population size of {len(self.DISTANCE)}', color='red')

            axs[0].set_xlabel('Iteration')
            axs[0].set_ylabel('amount of individuals with the same fitness value')
            axs[0].set_title('Duplicate Fitness Values Across Iterations')
            axs[0].legend()

            axs[1].set_xlabel('Iteration')
            axs[1].set_ylabel('amount of distinct individuals')
            axs[1].set_title('Duplicate Individuals Across Iterations')
            axs[1].legend()

            axs[2].set_xlabel('Iteration')
            axs[2].set_ylabel('amount of valid individuals (permutation wise)')
            axs[2].set_title('Amount Of Valid Individuals Across Iterations')
            axs[2].legend()

            filename = f'{self.fileidx}/diversity/plot_diversity/diversity_figure_{self._COUNTER}.svg'
            figs.savefig(filename,format='svg')
            plt.close(figs)

    def update_convergenceplot(self):
        if self.plot_convergence:
            if not os.path.exists(f'{self.fileidx}/convergence'):
                os.makedirs(f'{self.fileidx}/convergence')
    
            plt.clf()
            fig, axs = plt.subplots()
            axs.plot(range(self._COUNTER+1), self.mean_objective_values,
                    label='Mean Objective', color='orange')
            axs.plot(range(self._COUNTER+1), self.best_objective_values,
                    label='Best Objective', color='green')
            axs.set_xlabel('Iterations')
            axs.set_ylabel('Objective Values')
            axs.set_title('Convergence Plot')
            axs.legend()

            filename = f'{self.fileidx}/convergence/convergence_plot_{self._COUNTER}.svg'
            fig.savefig(filename, format='svg')        
            plt.close(fig) 

if __name__ == "__main__":

    distance_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    population = [
        [0, 1, 2, 3],  
        [0, 2, 1, 3],  
        [3, 2, 1, 0],  
        [1, 0, 3, 2]
    ]
    iterations = 10
    adj_matrix = adjacency_list(distance_matrix)

    vis = stat(distance_matrix,adj_matrix,True,True,4)
    vis.update_visualization(population, constrained_fitness)