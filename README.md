# Py-Go
[![Build Status](https://app.travis-ci.com/phantom820/Py-Go.svg?branch=master)](https://app.travis-ci.com/phantom820/Py-Go)
[![codecov](https://codecov.io/gh/phantom820/Py-Go/branch/master/graph/badge.svg?token=VJ6J4DM859)](https://codecov.io/gh/phantom820/Py-Go)

A python global optimization library for single-objective optimization, in which we have inequality constraints charcterized by  constants.

### Requirements
- python 3.7 or greater
- pip

### Installation instructions 
- Download the following file https://github.com/phantom820/Py-Go/blob/master/dist/pygo-0.1.0-py3-none-any.whl and run the following command where you downloaded the file
- pip install pygo-0.1.0-py3-none-any.whl 

### Usage Examples
- for relevant usage examples see https://github.com/phantom820/Py-Go/tree/master/examples
- for machine learning related examples see https://github.com/phantom820/Py-Go/tree/master/machine_learning_examples

### Note 
With use with machine learning models a callable cost function must be implemented which will return the results of evaluating the function at a specified number of solutions , that is it accepts a 2d array (m,d) where m is the number of candidate solutions of dimension d. See 
https://github.com/phantom820/Py-Go/blob/master/machine_learning_examples/models.py for example how to implement.

### Implemented optimization algorithms
- particle swarm optimization (PSO)
- genetic algorithm (GA)
- differential evolution (DE)



