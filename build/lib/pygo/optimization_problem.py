import numpy as np


class OptimizationProblem:
    '''
        mode[str] 'min' / 'max'->  specifies type of optimization problem
        f[callable] -> objective function that returns function evaluation at given input
        lower_constraints[List] -> lower bound on each variable
        upper_constraints[List] -> upper bound on each variable
         
    '''    
    def __init__(self,mode,f,lower_constraints,upper_constraints):
        self.mode = mode
        self.lower_constraints = np.array(lower_constraints)
        self.upper_constraints = np.array(upper_constraints)
        self.f = f
        self.n = len(lower_constraints)


    
# f = lambda x:(x**2).ravel()
# op = OptimizationProblem('max',f,[0],[29])
# go = GlobalOptimizer()
# x_star,f_star = go.de(op)
# print(x_star,f_star)
