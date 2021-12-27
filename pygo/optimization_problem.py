from typing import List
import numpy as np


class OptimizationProblem:
    '''
        mode[str] 'min' / 'max'->  specifies type of optimization problem
        f[callable] -> objective function that returns function evaluation at given input
        lower_constraints[List] -> lower bound on each variable
        upper_constraints[List] -> upper bound on each variable
        feasibility[bool] -> specifies whether to keep solutions in constraints (usefule for rather unbounded problems)
         
    '''    
    def __init__(self,mode,f,lower_constraints,upper_constraints,feasibility=True):

        if type(mode) is not str:
            raise TypeError('mode must be str')

        if mode!='min' and mode!='max':
            raise ValueError('mode must be str min or max')

        if type(lower_constraints) is not list and type(lower_constraints) is not np.ndarray:
            raise TypeError("lower_constaints must be a list or ndarray")

        if type(upper_constraints) is not list and type(upper_constraints) is not np.ndarray:
            raise TypeError("upper_constaints must be a list or ndarray")

        if len(lower_constraints)!=len(upper_constraints):
            raise ValueError('constrainst must have same length')
        
        if not callable(f):
            raise TypeError('f objectove function must be callable')
            
        self.mode = mode
        self.lower_constraints = np.array(lower_constraints)
        self.upper_constraints = np.array(upper_constraints)
        self.f = f
        self.n = len(lower_constraints)
        self.feasibility = feasibility