from pygo import global_optimizers
from pygo import optimization_problem as op
import pytest
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
		other_expression = 'Maximize \nlambda x:x[:,0]**2-4*x[:,0]+4\nSubject to\n0<= x <=6'.split('\n')
		assert expression==other_expression

		# problem with several constraints
		optimization_problem = op.OptimizationProblem('lambda x:(x[:,0]+2*x[:,1]-7)**2+(2*x[:,0]+x[:,1]-5)**2','min',
                              [-10,-10],[10,10])
		expression = str(optimization_problem).split('\n')
		other_expression = 'Minimize \nlambda x:(x[:,0]+2*x[:,1]-7)**2+(2*x[:,0]+x[:,1]-5)**2\nSubject to\n-10<= x0 <=10\n-10<= x1 <=10\n'
		other_expression = other_expression.split('\n')
		assert expression==other_expression

		# errors and exceptions
		with pytest.raises(TypeError) as e_info:
			optimization_problem = op.OptimizationProblem(2,'min',
                              [0],[6])
		
		with pytest.raises(Exception) as e_info:
			optimization_problem = op.OptimizationProblem('lambda x:x[:,0]**2-4*x[:,0]+4','min',
                              [0],[6,4])

		with pytest.raises(TypeError) as e_info:
			optimization_problem = op.OptimizationProblem('lambda x:x[:,0]**2-4*x[:,0]+4','m',
                              [0],[6])
		