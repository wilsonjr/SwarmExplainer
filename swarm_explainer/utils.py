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


def show_importance_lines(threads, n_features, feature_names, model, X, y, X_test, y_test, klass, MAX_VALUE=10):
    
    max_x = -1
    min_x = 100000
    for i in range(len(threads)):
        for feature in range(n_features):
            max_x = max(max_x, np.max(threads[feature].importances))
            min_x = min(min_x, np.min(threads[feature].importances))

    features_I = []
    mean_w = []
    mean_acc = []
    for thread in threads:
        
        importances = thread.feature_importances
        feature = thread.feature
        
        sum_diff_acc = 0.0
        sum_diff_w = 0.0

        sum_w = 0
        sum_acc = 0
        
        for i, weight in enumerate(importances):
            
            w_ij = weight[feature]
            
            # X_class = weight * X_test[y_test == klass]     
            # y_predicted_wij = np.argmax(model.predict_proba(X_class), axis=1)
            # acc_wij = accuracy_score(y_test[y_test == klass], y_predicted_wij)
                    
            # y_predicted_1 = np.argmax(model.predict_proba(X_test[y_test == klass]), axis=1)
            # acc_w1 = accuracy_score(y_test[y_test == klass], y_predicted_1)

            X_class = weight * X_test[y_test == klass]
            X_final = np.concatenate((X_class, X_test[y_test != klass]), axis=0)
            y_predicted_wij = np.argmax(model.predict_proba(X_final), axis=1)
            y_true = y_test[y_test == klass]
            y_final = np.concatenate((y_true, y_test[y_test != klass]), axis=None)        
            acc_wij = accuracy_score(y_final, y_predicted_wij)
            
            y_predicted_1 = np.argmax(model.predict_proba(X_test), axis=1)
            acc_w1 = accuracy_score(y_test, y_predicted_1)
        
            sum_diff_acc += abs(acc_w1 - acc_wij)
            sum_diff_w += abs(1 - w_ij)

            sum_w += w_ij
            sum_acc += acc_wij
            
        if sum_diff_w == 0.0:
            features_I.append(0.0)
            mean_w.append(sum_diff_w/len(importances))
            mean_acc.append(sum_diff_acc/len(importances))
        else:
            mean_w.append(sum_diff_w/len(importances))
            mean_acc.append(sum_diff_acc/len(importances))
            features_I.append((sum_diff_acc/len(importances))/(sum_diff_w/len(importances)))
        
    features_I = np.array(features_I)


    names = feature_names
    mean_w = np.array(mean_w)
    mean_acc = np.array(mean_acc)
    importances = np.zeros(len(mean_w))

    for i in range(len(importances)):
        if mean_w[i] == 0:
            importances[i] = 0.0
        else:
            # importances[i] = mean_acc[i]*( MAX - ((mean_w[i]/np.max(mean_w[mean_acc == mean_acc[i]])) /MAX)) + mean_acc[i]
            importances[i] = mean_acc[i]*( MAX_VALUE - ((mean_w[i]/np.max(mean_w[mean_acc == mean_acc[i]])) /MAX_VALUE))

    df = pd.DataFrame({
        'names': names,
        'w': mean_w,
        'acc': mean_acc,
        'importances': importances,
    })

    df = df.sort_values(by=['importances'], ascending=False)
    ordered_features = df.index.values

    fig, axs = plt.subplots(n_features, 3, figsize=(12,n_features*1.1), 
                            gridspec_kw={'width_ratios': [3, 1, 1]})
    
    for j, feature in enumerate(ordered_features):
        swarmImportance = threads[feature]        
        lines = extract_weights(swarmImportance.importances, feature)
        
        axs[j, 0].spines['top'].set_visible(False)
        axs[j, 0].spines['left'].set_visible(False)
        axs[j, 0].spines['right'].set_visible(False)
        axs[j, 0].set_ylim([-1.2, 1.2])
        axs[j, 0].set_xlim([min_x-0.5, max_x+0.5])        
        axs[j, 0].get_yaxis().set_ticks([])
        
        if j < n_features-1:
            axs[j, 0].spines['bottom'].set_visible(False)
            axs[j, 0].get_xaxis().set_visible(False)
            
            
        for index in range(lines.shape[0]):
            weights = np.sort(np.unique(lines[index]))
            accs = []
            evolution = []
            y_jitter = np.zeros(len(weights))
            y_jitter = y_jitter + np.random.normal(0, 0.10, len(y_jitter))
            for i, weight in enumerate(weights):
                vec = np.ones(X_test.shape[1])
                vec[feature] = weight

                # X_class = vec * X_test[y_test == klass]
                # y_predicted = np.argmax(model.predict_proba(X_class), axis=1)
                # acc = accuracy_score(y_test[y_test == klass], y_predicted)

                X_class = vec * X_test[y_test == klass].copy()
                X_final = np.concatenate((X_class, X_test[y_test != klass]), axis=0)

                y_predicted = np.argmax(model.predict_proba(X_final), axis=1)
                y_true = y_test[y_test == klass]
                y_final = np.concatenate((y_true, y_test[y_test != klass]), axis=None)

                acc = accuracy_score(y_final, y_predicted)

                accs.append(acc)
                evolution.append(i)

            accs = np.array(accs)
            axs[j, 0].scatter(weights, y_jitter, alpha=0.8, s=50, c = accs, cmap='viridis', vmin=0, vmax=1)
            
        min_importance = swarmImportance.importances[-1][:,feature].min()
        max_importance = swarmImportance.importances[-1][:,feature].max()
        
        axs[j, 0].set_ylabel(feature_names[feature].upper(), rotation='horizontal', fontsize=12, ha='right', va='center')
        axs[j, 0].plot([min_importance, max_importance], [-.7,-.7], c='red', lw=1, zorder=100, marker='o', ms=6)
#         axs[j].plot([mean_importances[j]],[-0.7], c='red', 
#                     lw=1, zorder=100, marker='o', 
#                     ms=6)
        if j < n_features-1:
            axs[j, 0].axvline(x=1, ymin=-1.2, ymax=1,
                               c='gray', lw=1, zorder=-1, clip_on=False)
        else:
            axs[j, 0].axvline(x=1, ymin=0, ymax=1,
                               c='gray', lw=1, zorder=-1, clip_on=False)
            axs[j, 0].set_xlabel('Feature weight', ha='center', fontsize=11)
            
    for j, feature in enumerate(ordered_features):
        axs[j, 1].spines['top'].set_visible(False)
        axs[j, 1].spines['right'].set_visible(False)
        with sns.axes_style('white'):
            sns.distplot(X[y == klass, feature], kde=False, ax=axs[j, 1], color='gray')
        if j == len(ordered_features)-1:
            axs[j, 1].set_xlabel('Distribution values', fontsize='11', ha='center')

    for k, j in enumerate(ordered_features):
        axs[k, 2].spines['top'].set_visible(False)
        axs[k, 2].spines['right'].set_visible(False)
        with sns.axes_style('white'):
            clrs = ['blue' if (x == df['names'][j]) else 'gray' for x in df['names'].values]
            ax = sns.barplot(x='names', y='importances', data=df, ax=axs[k, 2], palette=clrs)
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
        if k == len(ordered_features)-1:
            axs[k, 2].set_xlabel('Importance Value')


    # fig.subplots_adjust(hspace = 0.5, wspace=0.9, right=0.8)
    
    
    cbar_ax = fig.add_axes([0.0, 0.08, 0.005, 0.87])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), cax=cbar_ax)
    cbar.ax.set_ylabel('ACCURACY SCORE', fontsize=10)
    cbar.ax.get_yaxis().set_ticks([0.0,0.5,1.0])
    cbar.ax.get_yaxis().set_ticks_position('left')
    cbar.ax.get_yaxis().set_label_position('left')
    cbar.outline.set_visible(False) 
    
    fig.tight_layout()
        