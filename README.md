# Py-Go
[![Build Status](https://app.travis-ci.com/phantom820/Py-Go.svg?branch=master)](https://app.travis-ci.com/phantom820/Py-Go)
[![codecov](https://codecov.io/gh/phantom820/Py-Go/branch/master/graph/badge.svg?token=VJ6J4DM859)](https://codecov.io/gh/phantom820/Py-Go)

A python global optimization library for single-object optimization of continuous functions. That is given a possibly nonlinear and non-convex continuous function **f** that attains some global minima/maxima **f<sub>min</sub>**/**f<sub>max</sub>**. The goal is to find **x** in the domain of **f** where the minima/maxima occurs. See https://en.wikipedia.org/wiki/Global_optimization for more detailed description. 

### Implemented global optimization algorithms
- particle swarm optimization (PSO)
- genetic algorithm (GA)
- differential evolution (DE)

The implementation of the above algorithms is based on the material that can be found here
 
### Installation instructions
Note it is recommended that you create a virtual env for installing. 
- `git clone https://github.com/phantom820/Py-Go`
- `cd Py-Go`
- `pip install -r requirements.txt`
- `bash build.bash`

### Usage Examples
- for relevant usage examples see https://github.com/phantom820/Py-Go/tree/master/examples



