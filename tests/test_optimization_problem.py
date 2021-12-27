from pygo import global_optimizers
from pygo import optimization_problem as op
import pytest
import numpy as np

def test_optimization_problem():
		optimization_problem = op.OptimizationProblem('min',lambda x:x[:,0]**2-4*x[:,0]+4,
                              [0],[6])

		# # just check things	and functions												
		assert optimization_problem.mode=='min'
		assert optimization_problem.lower_constraints == np.array([0])
		assert optimization_problem.upper_constraints == np.array([6])
		random_input = np.random.randn(2,2)
		assert optimization_problem.f(random_input)[0] == (lambda x:x[:,0]**2-4*x[:,0]+4)(random_input)[0]
