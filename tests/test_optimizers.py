from pygo import global_optimizers as go
from pygo import optimization_problem as op
import numpy as np

# test pso
def test_pso():
		opp = op.OptimizationProblem('min',lambda x:x[:,0]**2-4*x[:,0]+4,
                              [0],[6])
		# check pso with given 1d minimization problem we should be close to true solution
		global_optimizer = go.GlobalOptimizer()
		x_star,f_star = global_optimizer.pso(opp)
		x_star = np.round(x_star,2)
		assert abs(2-x_star[0])<=1e-1	

		# check pso with given 1d maximization problem we should be close to true solution
		opp.mode = 'max'
		global_optimizer = go.GlobalOptimizer()
		x_star,f_star = global_optimizer.pso(opp)
		x_star_true = 6
		assert abs(x_star_true-x_star[0])<=1e-1

		# check pso with 2d rosenbrock minimization problem	
		opp = op.OptimizationProblem('min',lambda x:(1-x[:,0])**2 + 100*(x[:,1]-x[:,0]**2)**2,
                              [-5,-5],[10,10])

		global_optimizer = go.GlobalOptimizer()
		x_star,f_star = global_optimizer.pso(opp)
		x_star = np.round(x_star,2)		
		x_star_true = np.array([1,1])
		err = np.round(max(abs(x_star_true-x_star)),2)

		assert err<=1e-1

# test adaptive pso
def test_adaptive_pso():

		opp = op.OptimizationProblem('min',lambda x:x[:,0]**2-4*x[:,0]+4,
                              [0],[6])
		# check pso with given 1d minimization problem we should be close to true solution
		global_optimizer = go.GlobalOptimizer()
		x_star,f_star = global_optimizer.adaptive_pso(opp)
		x_star_true = 2
		err = np.round(max(abs(x_star_true-x_star)),2)

		assert err<=1e-1

		# check pso with given 1d maximization problem we should be close to true solution
		opp.mode = 'max'
		global_optimizer = go.GlobalOptimizer()
		x_star,f_star = global_optimizer.adaptive_pso(opp)
		x_star = np.round(x_star,2)
		x_star_true = 6
		err = np.round(max(abs(x_star_true-x_star)),2)
		assert err<=1e-1

		# check pso with 2d rosenbrock minimization problem	
		opp = op.OptimizationProblem('min',lambda x:np.sin(x[:,0]+x[:,1])+(x[:,0]-x[:,1])**2-1.5*x[:,0]+2.5*x[:,1]+1,
                              [-1.5,-3],[4,4])

		global_optimizer = go.GlobalOptimizer()
		x_star,f_star = global_optimizer.adaptive_pso(opp)
		x_star = np.round(x_star,2)		
		x_star_true = np.array([-0.54,-1.54])
		err = np.round(max(abs(x_star_true-x_star)),2)
		assert err<=1e-1

# test genetic algorithm
def test_ga():

		opp = op.OptimizationProblem('min',lambda x:x[:,0]**2-4*x[:,0]+4,
                              [0],[6])
		# check pso with given 1d minimization problem we should be close to true solution
		global_optimizer = go.GlobalOptimizer()
		x_star,f_star = global_optimizer.ga(opp)
		x_star_true = 2
		err = np.round(max(abs(x_star_true-x_star)),2)
		assert err<=1e-1

		# check pso with given 1d maximization problem we should be close to true solution
		opp.mode = 'max'
		global_optimizer = go.GlobalOptimizer()
		x_star,f_star = global_optimizer.ga(opp)
		x_star = np.round(x_star,2)
		x_star_true = 6
		err = np.round(max(abs(x_star_true-x_star)),2)
		assert err<=1e-1

		# check pso with 2d rosenbrock minimization problem	
		opp = op.OptimizationProblem('min',lambda x:np.sin(x[:,0]+x[:,1])+(x[:,0]-x[:,1])**2-1.5*x[:,0]+2.5*x[:,1]+1,
                              [-1.5,-3],[4,4])

		global_optimizer = go.GlobalOptimizer()
		x_star,f_star = global_optimizer.ga(opp)
		x_star = np.round(x_star,2)		
		x_star_true = np.array([-0.54,-1.54])
		err = np.round(max(abs(x_star_true-x_star)),2)
		assert err<=1e-1

# test differential evolution
def test_de():
		opp = op.OptimizationProblem('min',lambda x:x[:,0]**2-4*x[:,0]+4,
                              [0],[6])
		# check pso with given 1d minimization problem we should be close to true solution
		global_optimizer = go.GlobalOptimizer()
		x_star,f_star = global_optimizer.de(opp,N=300)
		x_star_true = 2
		err = np.round(max(abs(x_star_true-x_star)),2)
		assert err<=1e-1

		# check pso with given 1d maximization problem we should be close to true solution
		opp.mode = 'max'
		global_optimizer = go.GlobalOptimizer()
		x_star,f_star = global_optimizer.de(opp,N=400)
		x_star = np.round(x_star,2)
		x_star_true = 6
		err = np.round(max(abs(x_star_true-x_star)),2)
		assert err<=1e-1

		# check pso with 2d rosenbrock minimization problem	
		opp = op.OptimizationProblem('min',lambda x:np.sin(x[:,0]+x[:,1])+(x[:,0]-x[:,1])**2-1.5*x[:,0]+2.5*x[:,1]+1,
                              [-1.5,-3],[4,4])

		global_optimizer = go.GlobalOptimizer()
		x_star,f_star = global_optimizer.de(opp,N=400)
		x_star = np.round(x_star,2)		
		x_star_true = np.array([-0.54,-1.54])
		err = np.round(max(abs(x_star_true-x_star)),2)
		assert err<=1e-1


					