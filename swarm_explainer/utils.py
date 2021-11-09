import pandas as pd
import numpy as np 
import math 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score

def f(particle, X, y, klass, model, arg_best, metric):
    """
    Computes the optimization function 

        Parameters:
        - particle (float): the weight assumed by a particle
        - X (np.array): the input dataset
        - y (np.array): the corresponding training labels
        - klass (int): the class in which the particles are looking
        - model (sklearn-based): the model being explained
        - arg_best (argmax or argmin): the function that specifies which prediction to consider
        - metric (sklearn-based): the metric used for explanation (e.g., accuracy_score)

        Returns:
        - float: the differente between the accuracy with and without perturbation (for the specified class)    
    """
    X_class = particle * X[y == klass].copy()
    
    X_final = np.concatenate((X_class, X[y != klass]), axis=0)
    y_predicted = arg_best(model.predict_proba(X_final), axis=1)
    y_true = y[y == klass]
    y_final = np.concatenate((y_true, y[y != klass]), axis=None)

    return abs(1-metric(y_final, y_predicted)) 

def define_neighbors(N, k):
    """
    Defines the neighborhood for each particle of PSO following the ring pattern

        Parameters:
        - N (int): number of particles
        - k (int): number of neighbors

        Returns:
        - np.array (N, k): the indices of k neighbors for each particle (row)    
    """

    neighborhood = np.zeros((N, k*2))
    
    for i in range(N):
        index = 0
        for j in range(1, k+1):
            neighborhood[i][index] = int(-j+i)
            index += 1
        for j in range(1, k+1):
            neighborhood[i][index] = int((j+i) % N)
            index += 1
            
    # TODO use set so the neighbors are not duplicated
    return neighborhood

def particle_swarm_optimization(max_it, N, m, model, X, y, klass,
                                min_value, max_value, 
                                AC1, AC2, Vmin, Vmax, feature, arg_best, metric=accuracy_score, init_strategy='ones', k=1, 
                                constriction=0.729, verbose=True):
    x = None

    # TODO use a design pattern
    if init_strategy == 'ones':
        x = np.ones((N, m))
    elif init_strategy == 'random':
        x = np.random.uniform(min_value, max_value, (N, m))
    else:
        x = np.random.uniform(0.8, 1.2, (N, m))
    
    p = x.copy()
    v = np.random.uniform(Vmin, Vmax, (N, m))
    
    neighborhood = define_neighbors(N, k)
    epoch = 0
    
    importances = []
    
    for epoch in range(max_it):
        
        if verbose and (epoch+1) % 10 == 0:
            print(f"Feature {feature} => epoch {epoch+1}/{max_it}")
        
        x[:, :feature] = 1.0
        p[:, :feature] = 1.0
        
        x[:, feature+1:] = 1.0
        p[:, feature+1:] = 1.0
        
        
        importances.append(p.copy())
        
        for i in range(N):
            
            if (f(x[i], X, y, klass, model, arg_best, metric) > f(p[i], X, y, klass, model, arg_best, metric)) or \
               (f(x[i], X, y, klass, model, arg_best, metric) != 0.0 and \
                f(x[i], X, y, klass, model, arg_best, metric) == f(p[i], X, y, klass, model, arg_best, metric) and \
                math.sqrt((1.0 - x[i][feature])**2.0) < math.sqrt((1.0 - p[i][feature])**2.0)):
            
                p[i] = x[i].copy()
            
            g = i
            
            for j in neighborhood[i]:
                j = int(j)
                if (f(p[j], X, y, klass, model, arg_best, metric) > f(p[g], X, y, klass, model, arg_best, metric)) or \
                   (f(p[j], X, y, klass, model, arg_best, metric) != 0.0 and \
                    f(p[j], X, y, klass, model, arg_best, metric) == f(p[g], X, y, klass, model, arg_best, metric) and \
                    math.sqrt((1.0 - p[j][feature])**2.0) < math.sqrt((1.0 - p[g][feature])**2.0 )):
                    g = j
                    
            fi_1 = np.random.uniform(0, AC1, m) #random_vector(m, AC1)
            fi_2 = np.random.uniform(0, AC2, m) #random_vector(m, AC2)
                    
            v[i] = constriction*(v[i] + fi_1 * (p[i]-x[i]) + fi_2 * (p[g]-x[i]))
            v[i] = np.clip(v[i], Vmin, Vmax)
            
            x[i] = x[i] + v[i]
            x[i] = np.clip(x[i], min_value, max_value)
            
            
    return p, importances


def extract_weights(importances, feature):
    """
    Gets the importance values for a specific feature

        Parameters:
        - importances (np.array): 3-dimensional array containing the importances for each epoch
                                  associated to the features and particles
        - feature (int): the index of the feature 

        Returns:
        - np.array: matrix containing the importance for each epoch and feature
    
    """
    lines = np.zeros((importances.shape[1], importances.shape[0]))
    
    for i in range(importances.shape[1]):        
        for j in range(importances.shape[0]):            
            lines[i][j] = importances[j][i][feature]
            
    return np.array(lines)