.. -*- mode: rst -*-

|pypi_version|_ |pypi_downloads|_

.. |pypi_version| image:: https://img.shields.io/pypi/v/swarm-explainer.svg
.. _pypi_version: https://pypi.python.org/pypi/swarm-explainer/

.. |pypi_downloads| image:: https://pepy.tech/badge/swarm-explainer/month
.. _pypi_downloads: https://pepy.tech/project/swarm-explainer

==============
SwarmExplainer
==============

SwarmExplainer is a model-agnostic technique to explain machine learning results using visualization of feature perturbations generated by a nature-inspired algorithm. Read the `preprint <https://arxiv.org/abs/2101.10502>`_ for further details.

------------
Installation
------------

You can install SwarmExplainer using pip:

.. code:: bash

    pip install swarm-explainer

--------------
Usage examples
--------------

SwarmExplainer uses feature perturbations to explain how a trained model rects to changes on the features.

**Training a model**

In its current version, SwarmExplainer handles sklearn-based classification models.

.. code:: python 

    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names

    # preprocess the dataset so feature perturbation will take effect
    X = swarm_explainer.utils.preprocess(X)
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LogisticRegression()
    model.fit(X_train, y_train)


**Computing explanations**

.. code:: python

    swarm = SwarmExplainer(
        max_it=100, 
        N=10, 
        m=X_test.shape[1], 
        model=model, 
        feature_names=feature_names, 
        n_classes=len(np.unique(y)), 
        verbose=True
    )

Although there are other few parameters you can tune, to explain a model using SwarmExplainer you should control the following:

-  ``max_it``: The number of iterations of the PSO algorithm;

-  ``N``: The number of particles searching for the feature perturbations;

-  ``m``: The dimensionality of the dataset;

-  ``model``: The model to be explained;

-  ``feature_names``: The feature names for generating the visualization;

-  ``n_classes``: The number of classes;

-  ``verbose``: Controls the verbosity of the technique.


**Interpreting the results**

After computing the explanations, you can generate graphs to interpret the results.

Suppose you want to explain class ``0`` in terms of the feature perturbations:


.. code:: python

    swarm.plot_importance(0, X, y)

The code produces the following plot:

.. image:: docs/artwork/iris-class0.png
	:alt: Explanations for the class 0

The above visualization shows the particle weights (perturbations) and their correspondent change on the performance of the model.

- ``1.`` The features are organized according to their importance for the model. Most important on top;
- ``2.`` The optimal weight for each feature is indicated by a red dot.
- ``3.`` The best weights are the ones that reduce the performance and are close to one.

The visualization also shows the distribution of values to help interpreting the results, and a summary importance value.

In summary, the model learned that classify instances as in class ``0`` when they have *high* sepal width and *low* sepal and petal length.


**Retrieving feature importance**

To retrieve a numerical representation of the feature importance, SwarmExplainer offers two methods. You can retrieve the feature importance within a class or among all classes.

.. code:: python 

    swarm.important_features(klass=0)
    swarm.important_features()


There is a complete example in the *notebooks/* folder.



--------
Citation
--------

Please, use the following reference to further details and to cite ClusterShapley in your work:

.. code:: bibtex

    @misc{MarcilioJr2021_SwarmExplainer,
      title={Model-agnostic interpretation by visualization of feature perturbations}, 
      author={Wilson E. Marcílio-Jr and Danilo M. Eler and Fabrício Breve},
      year={2021},
      eprint={2101.10502},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }

-----------
Support 
-----------

Please, if you have any questions feel free to contact me at wilson_jr at outlook dot com.

-------
License
-------

SwarmExplainer follows the 3-clause BSD license.


