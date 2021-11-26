from . import utils

from sklearn.metrics import accuracy_score

import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

from tqdm import tqdm

class SwarmExplainer():

    """Model-agnostic explanations using feature perturbations

    Uses PSO algorithm to compute perturbing weights and build a visualization
    for explanations.

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
    
    feature_names: list
        A list of strs containing the feature names.

    n_classes: int
        The number of classes.

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

    V1: float (default -1)
        Minimum velocity of the particles.

    V2: float (default 1)
        Maximum velocity of the particles.

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
    def __init__(self, max_it, N, m, model, feature_names, n_classes, min_value=0, max_value=10, AC1=2.05, AC2=2.05, 
        Vmin=-1, Vmax=1, arg_best = np.argmax, metric=accuracy_score, init_strategy='ones', k=1, constriction=0.729, verbose=True):

        self.max_it = max_it
        self.N = N
        self.m = m
        self.model = model
        self.X = None
        self.y = None
        self.feature_names = feature_names
        self.n_classes = n_classes
        self.min_value = min_value
        self.max_value = max_value
        self.AC1 = AC1
        self.AC2 = AC2
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.arg_best = arg_best
        self.init_strategy = init_strategy
        self.k = k
        self.constriction = constriction
        self.verbose = verbose
        self.metric = metric 


        self.class_importances = []
        self.class_information = []


    def fit_transform(self, X, y, strategy='mean'):
        """Compute explanations

        Parameters
        ----------        
        X: array (n samples, m dimensions)
            The test data.
        
        y: array (n samples)
            The test labels.

        strategy: str (default 'mean') 
            Strategy to compute feature importance according to the optimization function.
            If "mean", it uses all of the perturbing weights.
            If "best", it uses the best perturbing weight.
        """

        self.X = X
        self.y = y
        self.strategy = strategy

        for klass in tqdm(range(self.n_classes)):

            threads = []
            if self.verbose:
                print("Finding feature weights for class %d..." % (klass))


            for feature in range(self.X.shape[1]):
                finder = ParticleImportance(self.max_it, self.N, X.shape[1], self.model, self.X, self.y, klass, 
                                        self.min_value, self.max_value, self.AC1, self.AC2, self.Vmin, self.Vmax, 
                                        feature, arg_best = self.arg_best, metric = self.metric, k=self.k, init_strategy=self.init_strategy, verbose=self.verbose)
                finder.start()
                threads.append(finder)
                
            for thread in threads:
                thread.join()

            if self.verbose:
                print("Done finding weights!")

            self.class_importances.append(threads)

            if self.verbose:
                print("Now computing importances!")

            self.class_information.append(self._compute_information(threads, self.X, self.y, self.max_value, klass, self.model))

            if self.verbose:
                print()

    def important_features(self, normalized=False, klass=None):
        """Gets the feature importance 

        Parameters
        ----------
        normalized: bool (default False)
            True for normalized [0, 1] importances when klass == None.
            False for non-normalized importances.

        klass: int (default None) 
            When retrieving importances for a specific class.

        Returns
        -------
        dataframe containing feature names and their respective importances
        """

        if klass == None:

            aggregated_importance = np.zeros(self.X.shape[1])
            name_features = self.class_information[0].sort_values(by=['names'])['names'].values

            for k in range(self.n_classes):

                sorted_df = self.class_information[k].sort_values(by=['names'])
                values= None
                if normalized:
                    values = sorted_df['importances'].values / (np.max(sorted_df['importances'].values)+1e-10)
                else:
                    values = sorted_df['importances'].values 

                for i, value in enumerate(values):
                    aggregated_importance[i] += value

            df = pd.DataFrame({'names': name_features, 'importances': aggregated_importance})            
            df = df.sort_values(by=['importances'], ascending=False)
            return df
        else:
            return self.class_information[klass]

    def _compute_information(self, particles, X_test, y_test, MAX_VALUE, klass, model):
        """Computes the feature importances

        Parameters
        ----------
        particles: list
            The list of ParticleImportance used for computing explanations.

        X_test: array
            The test set with data points.

        y_test: array
            The test labels.

        MAX_VALUE: float
            The hyperparameter that limits feature importance.

        klass: int
            The class being explained.

        model: sklearn classifier
            The model being explained.


        Returns
        -------
        dataframe containing feature importances in descending order        
        """
        max_x = -1
        min_x = 100000
        for i in range(len(particles)):
            for feature in range(X_test.shape[1]):
                max_x = max(max_x, np.max(particles[feature].importances))
                min_x = min(min_x, np.min(particles[feature].importances))

        mean_w = []
        mean_acc = []
        for thread in particles:
            
            importances = thread.feature_importances
            feature = thread.feature
            
            sum_diff_acc = 0.0
            sum_diff_w = 0.0

            sum_w = 0
            sum_acc = 0

            best_weight = 0
            best_acc = 0
            
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
                acc_wij = self.metric(y_final, y_predicted_wij)
                
                y_predicted_1 = np.argmax(model.predict_proba(X_test), axis=1)
                acc_w1 = self.metric(y_test, y_predicted_1)

                if abs(acc_w1 - acc_wij) > best_acc:
                    best_acc = abs(acc_w1 - acc_wij)
                    best_weight = abs(1 - w_ij)
                elif abs(acc_w1 - acc_wij) == best_acc and abs(1 - w_ij) < best_weight and best_acc != 0.0:
                    best_weight = abs(1 - w_ij)
                    best_acc = abs(acc_w1 - acc_wij)
            
                sum_diff_acc += abs(acc_w1 - acc_wij)
                sum_diff_w += abs(1 - w_ij)

                sum_w += w_ij
                sum_acc += acc_wij
                
            
            if self.strategy == 'mean':
                mean_w.append(sum_diff_w/len(importances))
                mean_acc.append(sum_diff_acc/len(importances))
            else:
                mean_w.append(best_weight)
                mean_acc.append(best_acc)
            
       

        names = self.feature_names
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

        return df 

    def plot_importance(self, klass, X, y, plot_execution=True, show_best=True, filepath=None):
        """Creates a visualization to interpret results

        Parameters
        ----------
        klass: int
            The class to visualize the explanations.

        X: array
            The dataset instances.

        y: array 
            The labels.

        plot_execution: bool (default True) 
            If the visualization will encode all epochs.

        show_best: bool (default True) 
            If the visualization will encode the best particle/weight.

        filepath: str (default None): 
            To save the representation.
        """

        if len(self.class_information) == 0:
            print("Did you fit the data?")
            return None
        elif len(self.class_information) <= klass:
            print("I cannot recognize this class!")
            return None 

        df = self.class_information[klass]
        n_features = self.X.shape[1]

        ordered_features = df.index.values

        max_x = -1
        min_x = 100000
        for i in range(len(self.class_importances[klass])):
            for feature in range(n_features):
                max_x = max(max_x, np.max(self.class_importances[klass][feature].importances))
                min_x = min(min_x, np.min(self.class_importances[klass][feature].importances))

        fig, axs = plt.subplots(n_features, 3, figsize = (12, n_features*1.1), gridspec_kw = {'width_ratios': [3, 1, 1]})

        X_test = self.X
        y_test = self.y 
        
        for j, feature in enumerate(ordered_features):
            swarmImportance = self.class_importances[klass][feature]        

            lines = None 

            if not plot_execution:
                lines = utils._extract_weights(np.array([swarmImportance.importances[-1]]), feature)
            else:
                lines = utils._extract_weights(swarmImportance.importances, feature)
            
            axs[j, 0].spines['top'].set_visible(False)
            axs[j, 0].spines['left'].set_visible(False)
            axs[j, 0].spines['right'].set_visible(False)
            axs[j, 0].set_ylim([-1.2, 1.2])
            axs[j, 0].set_xlim([min_x-0.5, max_x+0.5])        
            axs[j, 0].get_yaxis().set_ticks([])
            
            if j < n_features-1:
                axs[j, 0].spines['bottom'].set_visible(False)
                axs[j, 0].get_xaxis().set_visible(False)
            
            best_acc = 1
            best_weight = 0

            best_diff_weight = 0
            best_diff_acc = 0

            final_weight = 1
            
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

                    y_predicted = np.argmax(self.model.predict_proba(X_final), axis=1)
                    y_true = y_test[y_test == klass]
                    y_final = np.concatenate((y_true, y_test[y_test != klass]), axis=None)

                    acc = self.metric(y_final, y_predicted)


                    y_predicted_1 = np.argmax(self.model.predict_proba(X_test), axis=1)
                    acc_w1 = self.metric(y_test, y_predicted_1)

                    if index == lines.shape[0]-1:
                        if abs(acc_w1 - acc) > best_diff_acc and acc < acc_w1:
                            best_diff_acc = abs(acc_w1 - acc)
                            best_diff_weight = abs(1 - weight)
                            final_weight = weight 
                            best_acc = acc 
                        elif abs(acc_w1 - acc) == best_diff_acc and abs(1 - weight) < best_diff_weight and best_diff_acc != 0.0:
                            best_diff_weight = abs(1 - weight)
                            final_weight = weight



                    accs.append(acc)
                    evolution.append(i)

                accs = np.array(accs)
                axs[j, 0].scatter(weights, y_jitter, alpha=0.8, s=50, c = accs, cmap='viridis', vmin=0, vmax=1)
                
            min_importance = swarmImportance.importances[-1][:,feature].min()
            max_importance = swarmImportance.importances[-1][:,feature].max()
            
            axs[j, 0].set_ylabel(self.feature_names[feature].upper(), rotation='horizontal', fontsize=12, ha='right', va='center')
            
            if not show_best:
                axs[j, 0].plot([min_importance, max_importance], [-.7,-.7], c='red', lw=1, zorder=100, marker='o', ms=6)
            else:
                axs[j, 0].plot([final_weight, final_weight], [-.7, -.7], c='red', lw=1, zorder=100, marker='o', ms=6)
   
            if j < n_features-1:
                axs[j, 0].axvline(x=1, ymin=-1.2, ymax=1, c='gray', lw=1, zorder=-1, clip_on=False)
            else:
                axs[j, 0].axvline(x=1, ymin=0, ymax=1, c='gray', lw=1, zorder=-1, clip_on=False)
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

        if filepath:
            fig.savefig(filepath, bbox_inches = "tight")    


class ParticleImportance(threading.Thread):
    """ParticleImportance

    Uses PSO algorithm to compute perturbing weights for a specific class.

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
    def __init__(self, max_it, N, m, model, X, y, klass, min_value, max_value, 
                AC1, AC2, Vmin, Vmax, feature, arg_best = np.argmax, metric = accuracy_score, 
                init_strategy='ones', k=1, constriction=0.729, verbose=True):
        
        threading.Thread.__init__(self)
        
        self.max_it = max_it
        self.N = N
        self.m = m
        self.model = model
        self.X = X
        self.y = y
        self.klass = klass
        self.min_value = min_value
        self.max_value = max_value
        self.AC1 = AC1
        self.AC2 = AC2
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.feature = feature
        self.arg_best = arg_best
        self.metric = metric
        self.init_strategy = init_strategy
        self.k = k
        self.constriction = constriction
        self.verbose = verbose

        self.feature_importances = None

        self.importances = None

    def run(self):
        """Computes explanations for a pair of (class, feature)        
        """
        self.feature_importances, self.importances = utils._particle_swarm_optimization(self.max_it, self.N, self.m, self.model, 
                                                                                 self.X, self.y, self.klass,
                                                                                 self.min_value, self.max_value, self.AC1, self.AC2, self.Vmin, 
                                                                                 self.Vmax, self.feature, self.arg_best, self.metric, self.init_strategy, 
                                                                                 self.k, self.constriction, self.verbose)

        if self.verbose:
            print("Done for feature %d." % (self.feature))

        self.importances = np.array(self.importances)

