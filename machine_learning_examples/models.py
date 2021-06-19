from pygo import optimization_problem as op
from pygo import global_optimizers as go
import numpy as np

class LogisticRegression:
    def __init__(self,seed):
        np.random.seed(seed)
        
    def sigmoid(self,X,theta):
        z = X@theta
        return 1/(1+np.exp(-z))
    
    def cost_function(self,weights):
        X = self.X_train
        y = self.y_train
        lambda_ = self.lambda_
        h = self.sigmoid(X,weights.T)
        y = y[:,None]
        entropy=-y*np.log(h+1e-5)-(1-y)*np.log(1-h+1e-5)+0.5*lambda_*(weights**2).sum(axis=1)
        J = np.mean(entropy,axis=0)
        return J
    
    def fit(self,X,y,optimizer={}):
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.d = len(X[0])
       
        global_optimizer = go.GlobalOptimizer()
        p = op.CostProblem(self.d,self.cost_function)
        
        # default params all algorithms
        w = 0.6
        w_init = 0.9
        N = 30
        max_iter = 200
        tol = 1e-4
        trajectory = True
        mutation = True
        F = 2
        CR = 0.7
        self.lambda_ = 0
        
        algo = 'pso'
        
        if 'algo' in optimizer:
            algo = optimizer['algo']
            
        if 'w' in optimizer:
            w = optimizer['w']

        if 'w_init' in optimizer:
            w_init = optimizer['w_init']

        if 'N' in optimizer:
            N = optimizer['N']

        if 'max_iter' in optimizer:
            max_iter = optimizer['max_iter']

        if 'tol' in optimizer:
            tol = optimizer['tol']

        if 'trajectory' in optimizer:
            trajectory = optimizer['trajectory']

        if 'mutation' in optimizer:
            mutation = optimizer['mutation']

        if 'F' in optimizer:
            F = optimizer['F']

        if 'CR' in optimizer:
            CR = optimizer['CR']
        
        if 'lambda_' in optimizer:
            self.lambda_ = optimizer['lambda_']
            
        if algo == 'pso':
            x_star,J_star, x_path,y_path = global_optimizer.pso(p,trajectory=True,max_iter=max_iter,
                                          N=N,tol=tol,w=w)
            
        elif algo == 'apso':
            x_star,J_star, x_path,y_path = global_optimizer.adaptive_pso(p,trajectory=True,max_iter=max_iter,
                                          N=N,tol=tol,w_init=w_init)
            
        elif algo == 'de':
            x_star,J_star, x_path,y_path = global_optimizer.de(p,trajectory=True,max_iter=max_iter,
                                          N=N,tol=tol,F=F,CR=CR)
        
        elif algo == 'ga':
            x_star,J_star, x_path,y_path = global_optimizer.ga(p,trajectory=True,max_iter=max_iter,
                                          N=N,tol=tol,mutation=mutation)
            
        self.weights = x_star
        self.cost_function_values = y_path
        
    def predict(self,X):
        theta = self.weights
        y_pred = self.sigmoid(X,theta)
        y_pred[y_pred>0.5] = 1
        y_pred[y_pred<=0.5] = 0
        return y_pred.astype(int)
    
    def score(self,X,y):
        y_pred = self.predict(X)
        return 100*(np.count_nonzero(y_pred==y)/len(y))
        
        
