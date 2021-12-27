from typing import List
import numpy as np


class OptimizationProblem:
    '''
        mode[str] 'min' / 'max'->  specifies type of optimization problem
        f[callable] -> objective function that returns function evaluation at given input
        lower_constraints[List] -> lower bound on each variable
        upper_constraints[List] -> upper bound on each variable
         
    '''    
    def __init__(self,mode,f,lower_constraints,upper_constraints):

        if type(mode) is not str:
            raise TypeError('mode must be str')

        if mode!='min' and mode!='max':
            raise ValueError('mode must be str min or max')

        if type(lower_constraints) is not list:
            raise TypeError("lower_constaints must be a list")

        if type(upper_constraints) is not list:
            raise TypeError("upper_constaints must be a list")

        if len(lower_constraints)!=len(upper_constraints):
            raise ValueError('constrainst must have same length')
        
        if not callable(f):
            raise TypeError('f objectove function must be callable')
            
        self.mode = mode
        self.lower_constraints = np.array(lower_constraints)
        self.upper_constraints = np.array(upper_constraints)
        self.f = f
        self.n = len(lower_constraints)
