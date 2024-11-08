import numpy as np
import os
import json
# import coloredlogs
# import logging
from hyperopt import fmin, tpe, hp, Trials
from sklearn.base import BaseEstimator

from src.util import adjacency_list,sort_adjacency_list,replace_infs
from src.IslandModelGA import IslandModelGA
import time

# TODO: parameters baysian nog eens runnen: kijken of die cycle crossover van 500 geen fluke is ofzo. 
# verslag scrhijven dan en gwn indienen. 
# TODO: alles verplaatsen naar 1 file, ongebruikte verwijderen en pythondoc toevoegen aan functies. 
# TODO: r0855183;r0855183() laten runnnen in een for loop met verschillende probleem groottes om te checken of er geen errors zijn. 

class CustomEstimator(BaseEstimator):
    """
    CustomEstimator class for hyperparameter tuning using a machine learning approach.

    This class implements a CustomEstimator, a scikit-learn compatible estimator, 
    to perform hyperparameter tuning using a machine learning approach. 
    It utilizes a genetic algorithm (GA) from the 'GA' class and hyperparameter optimization techniques from 'hyperopt'.

    The fit method within this class applies the GA with specified hyperparameters to a set of training data, 
    aiming to optimize the performance based on a specific objective function.

    Attributes:
        - TRAINING: Input training data for hyperparameter optimization.
        - BATCH_SIZE: Size of batches used during hyperparameter optimization.

    Methods:
        - __init__(training_data): Initializes the CustomEstimator with training_data.
        - fit(hyperparameters): Performs hyperparameter optimization using GA and specified hyperparameters.

    References:
        - Hyperparameter Tuning with Automated Machine Learning:
            - Medium Article: https://insaid.medium.com/automated-hyperparameter-tuning-988b5aeb7f2a
            - Towards Data Science Article: https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a
    """

    def __init__(self, training_data):
        self.TRAINING = training_data

    def fit(self, hyperparameters):
        t1 = time.time()

        selected_matrix = self.TRAINING[0][0]
        sorted_adj_matrix = self.TRAINING[0][1]
        print(
        str([{   "lambdaa":                  hyperparameters['lambdaa1'],
            "alpha":                    hyperparameters['alpha1'],
            "beta":                     hyperparameters['beta1'],
            "mu":                       hyperparameters['mu1'],
            "Kselect":                  hyperparameters['Kselect1'],
            "Kelim":                    hyperparameters['Kelim1'],
            "init_arg":                 hyperparameters["init_arg1"],
            "recomb_method":            hyperparameters['recomb_method1'],
            "local_search_iterations": hyperparameters['local_search_iterations1'],

        },
        {
            "lambdaa":                  hyperparameters['lambdaa2'],
            "alpha":                    hyperparameters['alpha2'],
            "beta":                     hyperparameters['beta2'],
            "mu":                       hyperparameters['mu2'],
            "Kselect":                  hyperparameters['Kselect2'],
            "Kelim":                    hyperparameters['Kelim2'],
            "init_arg":                 hyperparameters["init_arg2"],
            "recomb_method":            hyperparameters['recomb_method2'],
            "local_search_iterations":  hyperparameters['local_search_iterations2'],

        }])
        )


        config = [{
            "distance_matrix":          selected_matrix,
            "sorted_adj_matrix":        sorted_adj_matrix,
            "lambdaa":                  hyperparameters['lambdaa1'],
            "alpha":                    hyperparameters['alpha1'],
            "beta":                     hyperparameters['beta1'],
            "mu":                       hyperparameters['mu1'],
            "Kselect":                  hyperparameters['Kselect1'],
            "Kelim":                    hyperparameters['Kelim1'],
            "init_arg":                 hyperparameters["init_arg1"],
            "recomb_method":            hyperparameters['recomb_method1'],
            "local_search_iterations":  hyperparameters['local_search_iterations1'],
            "plot_convergence":         False,
            "plot_diversity":           False,
            "printing":                 True,
            "debug":                    False
        },{
            "distance_matrix":          selected_matrix,
            "sorted_adj_matrix":        sorted_adj_matrix,
            "lambdaa":                  hyperparameters['lambdaa2'],
            "alpha":                    hyperparameters['alpha2'],
            "beta":                     hyperparameters['beta2'],
            "mu":                       hyperparameters['mu2'],
            "Kselect":                  hyperparameters['Kselect2'],
            "Kelim":                    hyperparameters['Kelim2'],
            "init_arg":                 hyperparameters["init_arg2"],
            "recomb_method":            hyperparameters['recomb_method2'],
            "local_search_iterations":  hyperparameters['local_search_iterations2'],
            "plot_convergence":         False,
            "plot_diversity":           False,
            "printing":                 True,
            "debug":                    False
        }]

        self.INSTANCE = IslandModelGA(10, config)
        TimeLeft = 300
        best = 999999999
        oldscore = 999999999
        bestsol = None
        while TimeLeft > 5:
            oldscore = best

            t1 = time.time()
            mean, score, sol = self.INSTANCE.run_one_iteration()  # best obj, index 1
            if score < best:
                best = score
                bestsol = sol
            t2 = time.time()
            TimeLeft = TimeLeft - (t2 - t1)

            print("best")
            print(best)
            print("score")
            print(score)
        
        if TimeLeft < 0: 
            score = oldscore
        
        print("result")
        print(score)

        print("solution")
        print(bestsol)
        return score                


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "../../assingment_data/tour1000.csv")
    file = open(csv_path)
    DISTANCE = np.loadtxt(csv_path, delimiter=",")
    file.close()

    sorted_adj_matrix = sort_adjacency_list(
        adjacency_list(DISTANCE), DISTANCE)

    scale = 2
    DISTANCE = replace_infs(DISTANCE,scale)

    filtered_distance_matrices = []
    filtered_distance_matrices.append(
        (DISTANCE, sorted_adj_matrix))

    objective_function = CustomEstimator(filtered_distance_matrices)

    lambd   = [30,50]
    alpha   = [0.1, 0.3, 0.5, 0.7, 1]
    beta    = [0.1, 0.3, 0.5, 0.7, 1]
    mu      = [50,100]
    Kselect = [3, 5, 7]
    Kelim   = [3, 5, 7]
    init    = [
                0,   # 2 : random & greedy, 1 : random , 0 : greedy
                1, 
                2]
    recomb  = [
                0,   # 4:random, 3:order_backtracking, 2:cycle_crossover, 1: order_crossover, 0: PMX
                1, 
                2,
                3,
                4]
    local_search_iterations = [1, 5, 20]

    param_space = {
        "lambdaa1": hp.choice("lambdaa1", lambd),
        "alpha1": hp.choice("alpha1", alpha),
        "beta1": hp.choice("beta1", beta),
        "mu1": hp.choice("mu1", mu),
        "Kselect1": hp.choice("Kselect1", Kselect),
        "Kelim1": hp.choice("Kelim1", Kelim),
        "init_arg1": hp.choice("init_arg1", init),
        "recomb_method1": hp.choice("recomb_method1", recomb),
        "local_search_iterations1": hp.choice("local_search_iterations1",local_search_iterations),
        #-------------------------------------------------------------------------------
        "lambdaa2": hp.choice("lambdaa2", lambd),
        "alpha2": hp.choice("alpha2", alpha),
        "beta2": hp.choice("beta2",beta),
        "mu2": hp.choice("mu2", mu),
        "Kselect2": hp.choice("Kselect2", Kselect),
        "Kelim2": hp.choice("Kelim2", Kselect),
        "init_arg2": hp.choice("init_arg2", init),
        "recomb_method2": hp.choice("recomb_method2", recomb),
        "local_search_iterations2": hp.choice("local_search_iterations2",local_search_iterations)
    }

    trials = Trials()
    best = fmin(fn=objective_function.fit, space=param_space,
                algo=tpe.suggest, max_evals=100, trials=trials)

    best_hyperparameters = {
        "lambdaa1":                 lambd[best["lambdaa1"]],
        "alpha1":                   alpha[best["alpha1"]],
        "beta1":                    beta[best["beta1"]],
        "mu1":                      mu[best["mu1"]],
        "Kselect1":                 Kselect[best["Kselect1"]],
        "Kelim1":                   Kelim[best["Kelim1"]],
        "init_arg1":                init[best["init_arg1"]],
        "recomb_method1":           recomb[best["recomb_method1"]],
        "local_search_iterations1": local_search_iterations[best["local_search_iterations1"]],
        #-------------------------------------------------------------------------------
        "lambdaa2":                 lambd[best["lambdaa2"]],
        "alpha2":                   alpha[best["alpha2"]],
        "beta2":                    beta[best["beta2"]],
        "mu2":                      mu[best["mu2"]],
        "Kselect2":                 Kselect[best["Kselect2"]],
        "Kelim2":                   Kelim[best["Kelim2"]],
        "init_arg2":                init[best["init_arg2"]],
        "recomb_method2":           recomb[best["recomb_method2"]],
        "local_search_iterations2": local_search_iterations[best["local_search_iterations2"]],
    }

    print("Best Hyperparameters:", best_hyperparameters)
    print("Optimal Score:", trials.best_trial['result']['loss'])

    result = {
        'hyperparameters': best_hyperparameters,
        'optimal_score':   trials.best_trial['result']['loss']
    }
    




# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------



# TOUR 200
# [{'lambdaa': 30, 'alpha': 0.5, 'beta': 0.5, 'mu': 100, 'Kselect': 5, 'Kelim': 7, 'init_arg': 1, 'recomb_method': 0, 'local_search_iterations': 5},
#  {'lambdaa': 50, 'alpha': 0.5, 'beta': 0.5, 'mu': 50,  'Kselect': 3, 'Kelim': 5, 'init_arg': 0, 'recomb_method': 2, 'local_search_iterations': 1}]
# 36372.69542241105

#TOUR 500
# [{'lambdaa': 30, 'alpha': 0.5, 'beta': 0.3, 'mu': 50,  'Kselect': 7, 'Kelim': 5, 'init_arg': 0, 'recomb_method': 2, 'local_search_iterations': 1}, 
#  {'lambdaa': 30, 'alpha': 1,   'beta': 1,   'mu': 100, 'Kselect': 5, 'Kelim': 3, 'init_arg': 2, 'recomb_method': 4, 'local_search_iterations': 20}]
# 132369.1570006687

#TOUR 750
# [{'lambdaa': 30, 'alpha': 0.7, 'beta': 0.1, 'mu': 50,  'Kselect': 5, 'Kelim': 7, 'init_arg': 2, 'recomb_method': 2, 'local_search_iterations': 5},
#  {'lambdaa': 30, 'alpha': 0.3, 'beta': 0.5, 'mu': 100, 'Kselect': 7, 'Kelim': 5, 'init_arg': 2, 'recomb_method': 0, 'local_search_iterations': 1}]
# 197541.09839542626

#TOUR 1000
# [{'lambdaa': 50, 'alpha': 0.5, 'beta': 0.1, 'mu': 50,  'Kselect': 5, 'Kelim': 7, 'init_arg': 2, 'recomb_method': 4, 'local_search_iterations': 5}, 
#  {'lambdaa': 30, 'alpha': 0.1, 'beta': 0.1, 'mu': 100, 'Kselect': 5, 'Kelim': 3, 'init_arg': 0, 'recomb_method': 4, 'local_search_iterations': 20}]    
# 196618.27939421375











# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------

# TOUR 200
# {'lambdaa': 50, 'alpha': 0.1, 'beta': 1, 'mu': 50, 'Kselect': 5, 'Kelim': 7, 'init_arg': 2, 'recomb_method': 3}                                                                              
# {'lambdaa': 50, 'alpha': 0.7, 'beta': 0.5, 'mu': 100, 'Kselect': 3, 'Kelim': 7, 'init_arg': 0, 'recomb_method': 0}

# TOUR 500
# {'lambdaa': 30, 'alpha': 0.3, 'beta': 0.3, 'mu': 100, 'Kselect': 5, 'Kelim': 5, 'init_arg': 2, 'recomb_method': 2}                                                                                    
# {'lambdaa': 50, 'alpha': 0.7, 'beta': 1, 'mu': 50, 'Kselect': 5, 'Kelim': 5, 'init_arg': 0, 'recomb_method': 1}

# TOUR 750
# {'lambdaa': 30, 'alpha': 0.3, 'beta': 0.3, 'mu': 100, 'Kselect': 7, 'Kelim': 5, 'init_arg': 0, 'recomb_method': 0}                                                                               
# {'lambdaa': 50, 'alpha': 0.1, 'beta': 0.5, 'mu': 50, 'Kselect': 7, 'Kelim': 7, 'init_arg': 2, 'recomb_method': 4}

# TOUR 1000 
# {'lambdaa': 30, 'alpha': 0.3, 'beta': 0.3, 'mu': 100, 'Kselect': 7, 'Kelim': 7, 'init_arg': 0, 'recomb_method': 1}
# {'lambdaa': 30, 'alpha': 1, 'beta': 0.1, 'mu': 100, 'Kselect': 5, 'Kelim': 5, 'init_arg': 1, 'recomb_method': 4}

# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------

# TOUR 200
# Best Hyperparameters: 
    # {'lambdaa1': 30, 'alpha1': 0.35, 'beta1': 0.7, 'mu1': 50, 'Kselect1': 7, 'Kelim1': 3, 'init_arg1': 2, 'recomb_method1': 3, 
    #  'lambdaa2': 30, 'alpha2': 0.05, 'beta2': 0.7, 'mu2': 50, 'Kselect2': 7, 'Kelim2': 3, 'init_arg2': 2, 'recomb_method2': 1}
# Optimal Score: 
    # 36372.695422411045

# TOUR 500
# Best Hyperparameters: 
    # {'lambdaa1': 30, 'alpha1': 0.25, 'beta1': 0.7, 'mu1': 50, 'Kselect1': 3, 'Kelim1': 5, 'init_arg1': 2, 'recomb_method1': 2, 
    # 'lambdaa2': 30, 'alpha2': 0.35, 'beta2': 0.7, 'mu2': 50, 'Kselect2': 5, 'Kelim2': 5, 'init_arg2': 0, 'recomb_method2': 3}
# Optimal Score: 
    # 132115.57261199848
    

# TOUR 750
# Best Hyperparameters: 
    # {'lambdaa1': 30, 'alpha1': 0.35, 'beta1': 0.5, 'mu1': 50, 'Kselect1': 7, 'Kelim1': 3, 'init_arg1': 0, 'recomb_method1': 3, 
    # 'lambdaa2': 30, 'alpha2': 0.05, 'beta2': 0.3, 'mu2': 50, 'Kselect2': 5, 'Kelim2': 3, 'init_arg2': 0, 'recomb_method2': 3}
# Optimal Score: 
    # 197394.68710917485


# TOUR 1000
# Best Hyperparameters: 
    # {'lambdaa1': 30, 'alpha1': 0.05, 'beta1': 0.5, 'mu1': 50, 'Kselect1': 3, 'Kelim1': 3, 'init_arg1': 0, 'recomb_method1': 1, 
    # 'lambdaa2': 30, 'alpha2': 0.05, 'beta2': 0.5, 'mu2': 50, 'Kselect2': 3, 'Kelim2': 3, 'init_arg2': 0, 'recomb_method2': 3}
# Optimal Score: 
    # 195847.53037735604
