# -*- coding: utf-8 -*-
# Author: Wilson Estécio Marcílio Júnior <wilson_jr@outlook.com>
#
# License: BSD 3 clause

from setuptools import setup


__version__ = "0.0.1"

with open('README.md', 'r') as f:
	long_description = f.read()

setup(
    name="swarm-explainer",
    version=__version__,
    author="Wilson E. Marcílio-Jr",
    author_email="wilson_jr@outlook.com",
    url="https://github.com/wilsonjr/SwarmExplainer",
    description="Model-agnostic explanations using feature perturbations",
    long_description=long_description,
    extras_require={"test": "pytest"},
    install_requires=['pandas', 'sklearn', 'numpy', 'shap', 'seaborn'],
    packages=['swarm_explainer'],
    zip_safe=False,
)