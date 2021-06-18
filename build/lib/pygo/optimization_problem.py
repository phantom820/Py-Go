import numpy as np

class OptimizationProblem:

    ''' expression (str) has to be an objective function we wish to optimize passed as a string and all 
        optimization variables for instance x,y,z and so on must be x[:,0],x[:,1],x[:,2] and so on. The :
        part is used by the optimization algorithms when evaluating a number of soultions for the objective 
        function. A few examples are shown below
        
        i.e f(x) = x^2 -> lambda x:x[:,0]**2 in 1 dimensional case
        i.e f(x,y) = x+y -> lambda x:x[:,0]+x[:,1] where x = (x[0],x[1])
        i.e f(x1,x2,x3)= x1*x2-x2^2 -> lambda x: x[:,0]*x[:,1]-x[:,1] and so on
        
        mode (int) indicates whether we minimizing or mazimizing
            0 -> minimization problem
            1 -> maximazation pproblem
        
        lower_constraints list/array of floats/array corresponding to lower constraints on optimization
            optimization variables
            i.e parsing 1.4 in min/max f(x) mean x>=1.4
            i.e parsing [1.4,2,5.4] ub min/max f(x) means x0>=1.4, x1>=2 , x2>=5.4 where x=[x0,x1,x2]
        
        upper_constraints float/list of floats/array corresponding to lower constraints on optimization
            optimization variables
            i.e parsing 1.4 in min/max f(x) mean x<=1.4
            i.e parsing [1.4,2,5.4] ub min/max f(x) means x0<=1.4, x1<=2 , x2<=5.4 where x=[x0,x1,x2]
            
    '''
    
    def __init__(self,expression,mode,lower_constraints,upper_constraints):
        if type(expression)!= str:
            raise TypeError('Only strings are allowed for function expressions')
            
        elif len(lower_constraints)!=len(upper_constraints):
            raise Exception('Constraint lists must be of the same size')
            
        elif type(mode)!=str or mode!='min' and mode!='max':
            raise TypeError('Only min or max str values allowed for mode')
        else:
            self.expression = expression
            self.mode = mode
            self.objective_function = eval(expression)
            self.lower_constraints = np.array(lower_constraints)
            self.upper_constraints = np.array(upper_constraints)
            
    def __str__(self):
        if self.mode=='min':
            line = 'Minimize \n' + self.expression + '\nSubject to\n'
            if len(self.lower_constraints)>1:
                for i in range(len(self.lower_constraints)):
                    line = line + str(self.lower_constraints[i])+"<= x"+str(i)+" <=" + str(self.upper_constraints[i])+"\n"
                return line
            else:
                line = line + str(self.lower_constraints[0])+'<= x <=' + str(self.upper_constraints[0])
                return line
        else:
            line = 'Maximize \n' + self.expression + '\nSubject to\n'
            if len(self.lower_constraints)>1:
                for i in range(len(self.lower_constraints)):
                    line = line + str(self.lower_constraints[i])+'<= x'+str(i)+' <=' + str(self.upper_constraints[i])+"\n"
                return line
            else:
                line = line + str(self.lower_constraints[0])+"<= x <=" + str(self.upper_constraints[0])
                return line


class CostProblem:

    def __init__(self,n,f):
        self.state = 0
        if type(n)!=type(int):
            TypeError('dimensions must be int')
        if callable(f):
            self.n = n
            self.objective_function = f
            self.mode = 'min'
            self.lower_constraints = np.random.uniform(-1,0,n)
            self.upper_constraints = np.random.uniform(0.1,1,n)
