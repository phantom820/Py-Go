from pygo import global_optimizers
from pygo import optimization_problem as op
import numpy as np

def test_optimization_problem():
		optimization_problem = op.OptimizationProblem('lambda x:x[:,0]**2-4*x[:,0]+4','min',
                              [0],[6])
		# just check things	and functions												
		assert optimization_problem.expression == 'lambda x:x[:,0]**2-4*x[:,0]+4','min'
		assert optimization_problem.mode=='min'
		assert optimization_problem.lower_constraints == np.array([0])
		assert optimization_problem.upper_constraints == np.array([6])
		random_input = np.random.randn(2,2)
		assert optimization_problem.objective_function(random_input)[0] == eval('lambda x:x[:,0]**2-4*x[:,0]+4')(random_input)[0]
		assert optimization_problem.objective_function(random_input)[1] == eval('lambda x:x[:,0]**2-4*x[:,0]+4')(random_input)[1]
		expression = str(optimization_problem).split('\n')
		other_expressions = 'Minimize \nlambda x:x[:,0]**2-4*x[:,0]+4\nSubject to\n0<= x <=6'.split('\n')
		assert expression==other_expressions

		# change the problem to maximization
		optimization_problem.mode = 'max'
		expression = str(optimization_problem).split('\n')
		other_expressions = 'Maximize \nlambda x:x[:,0]**2-4*x[:,0]+4\nSubject to\n0<= x <=6'.split('\n')
		assert expression==other_expressions
