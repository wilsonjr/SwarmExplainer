import pandas as pd
import numpy as np 
import math 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score

def _optim_function(particle, X, y, klass, model, arg_best, metric):
    """Computes the optimization function 

    Parameters
    ----------
    particle: float
        The weight assumed by the particle.
    
    X: array (n samples, m dimensions)
        The input dataset.

    y: array (n samples) 
        The test labels.

    klass: int
        The class in which the particles are looking.

    model: sklearn classifier
        The model being explained.

    arg_best: function (np.argmax or np.argmin)

        The function that specifies which prediction to consider.

    metric: function (sklearn-based classification metric)
        The metric used for explanation (e.g., accuracy_score).

    Returns
    -------
    float, the differente between the accuracy with and without perturbation (for the specified class)    
    """

    X_class = particle * X[y == klass].copy()
    
    X_final = np.concatenate((X_class, X[y != klass]), axis=0)
    y_predicted = arg_best(model.predict_proba(X_final), axis=1)
    y_true = y[y == klass]
    y_final = np.concatenate((y_true, y[y != klass]), axis=None)

    return abs(1-metric(y_final, y_predicted)) 

def _define_neighbors(N, k):
    """Defines the neighborhood for each particle of PSO following the ring pattern

    Parameters
    ----------
    N: int
        Number of particles.

    k: int 
        Size of neighborhood.

    Returns
    -------
    array (N particles, k neighbors)
        The indices of k neighbors for each particle (row).
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

def _particle_swarm_optimization(max_it, N, m, model, X, y, klass,
                                min_value, max_value, 
                                AC1, AC2, Vmin, Vmax, feature, arg_best, metric=accuracy_score, init_strategy='ones', k=1, 
                                constriction=0.729, verbose=True):
    """Computes the perturbing weights for a pair (class, dimenion).
    
    Parameters
    ----------
    max_it: int
        The number of epochs to run the algorithm. In general, values should
        be between 30 and 100.

    N: int
        The number of particles using in the optimization. Values should be
        between 5 and 15.

    m: int
        The number of features of the dataset.
    
    model: sklearn-based classifier
        The model to be explained.

    X: array (n samples, m dimensions)
        The test dataset.
    
    y: array (n samples)
        The test labels.

    klass: int
        The class to be explained.

    min_value: float (default 0)
        The minimum value assumed by the perturbing weights.
    
    max_value: float (default 10)
        The maximum value assumed by the perturbing weights.
    
    AC1: float (default 2.05)
        Limit in which samples will be drawn to update the 
        position of particles based on its current position.

    AC2: float (default 2.05)
        Limit in which samples will be drawn to update the 
        position of particles based on the current position of
        its most similar neighbor.

    Vmin: float (default -1)
        Minimum velocity of the particles.

    Vmax: float (default 1)
        Maximum velocity of the particles.

    feature: int
        The feature to be explained.

    arg_best: function (default np.argmax)
        The function specifying the best element according to the metric.

    metric: function (default accuracy_score)
        A function that returns a scalar representing the performance of the model.

    init_strategy: str (default 'ones')
        The weights initialization strategy.
        If 'ones', all weights will be initialized as 1.
        If 'random', all weights will be initialized as random numbers between 0.8 and 1.2.

    k: int (default 1)
        The size of neighborhood for each particle (weight).

    constriction: float (default 0.729)
        A factor that limits the overall particle velocity.

    verbose: bool (default True)
        Controls the verbosity of the process.
    """
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
    
    neighborhood = _define_neighbors(N, k)
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
            
            if (_optim_function(x[i], X, y, klass, model, arg_best, metric) > _optim_function(p[i], X, y, klass, model, arg_best, metric)) or \
               (_optim_function(x[i], X, y, klass, model, arg_best, metric) != 0.0 and \
                _optim_function(x[i], X, y, klass, model, arg_best, metric) == _optim_function(p[i], X, y, klass, model, arg_best, metric) and \
                math.sqrt((1.0 - x[i][feature])**2.0) < math.sqrt((1.0 - p[i][feature])**2.0)):
            
                p[i] = x[i].copy()
            
            g = i
            
            for j in neighborhood[i]:
                j = int(j)
                if (_optim_function(p[j], X, y, klass, model, arg_best, metric) > _optim_function(p[g], X, y, klass, model, arg_best, metric)) or \
                   (_optim_function(p[j], X, y, klass, model, arg_best, metric) != 0.0 and \
                    _optim_function(p[j], X, y, klass, model, arg_best, metric) == _optim_function(p[g], X, y, klass, model, arg_best, metric) and \
                    math.sqrt((1.0 - p[j][feature])**2.0) < math.sqrt((1.0 - p[g][feature])**2.0 )):
                    g = j
                    
            fi_1 = np.random.uniform(0, AC1, m) 
            fi_2 = np.random.uniform(0, AC2, m)
                    
            v[i] = constriction*(v[i] + fi_1 * (p[i]-x[i]) + fi_2 * (p[g]-x[i]))
            v[i] = np.clip(v[i], Vmin, Vmax)
            
            x[i] = x[i] + v[i]
            x[i] = np.clip(x[i], min_value, max_value)
            
            
    return p, importances


def _extract_weights(importances, feature):
    """Gets the importance values for a specific feature

    Parameters  
    ----------
    importances: array (N particles, m features, max_it iterations)
        3-dimensional array containing the importances for each epoch
        associated to the features and particles.

    feature: int
        The index of the feature.

    Returns
    -------
    array, matrix containing the importance for each epoch and feature.    
    """
    lines = np.zeros((importances.shape[1], importances.shape[0]))
    
    for i in range(importances.shape[1]):        
        for j in range(importances.shape[0]):            
            lines[i][j] = importances[j][i][feature]
            
    return np.array(lines)