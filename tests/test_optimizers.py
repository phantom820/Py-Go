from pygo import global_optimizers as go
from pygo import optimization_problem as op
import numpy as np

def test_pso():
		optimization_problem = op.OptimizationProblem('lambda x:x[:,0]**2-4*x[:,0]+4','min',
                              [0],[6])
		# check pso with given problem we should be close to true solution
		global_optimizer = go.GlobalOptimizer()
		x,f = global_optimizer.pso(optimization_problem)
		assert abs(2-x[0])<1e-2									