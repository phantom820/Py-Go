import numpy as np

class GlobalOptimizer:
    
    def pso(self,optimization_problem,N=100,c1=2,c2=2,w = 0.6 ,max_iter =300,tol=1e-4):
        ''' 
            basic pso optimization we have fixed inertia the parameters are as follows
            
            optimization_problem (OptimizationProblem) optimization problem object with objective function and
            constraints
            N (int) : number of particles in swarm default = 100
            c1 (int) range(0,4]: personal component of particle scales in direction of personal best default = 2,
            c2 (int) range(0,4]: social component of particle scales in direction of global best default = 2
            w (float) between 0 and 1 which indicates the inertial of the particle or what fraction of prior 
            velocity we add to get to next velocity values default 0.6
            max_iter (int): maximum number of iterations to run the algorithm for default = 300
            tol (float) : used as convergence criteria check if we have not converged to a solution uses default 1e-4
            max of abs difference  g(k)-g(k-1)
        '''
        
        # just check input parameters are ok
        if c1<0 or c1>4:
            raise Exception('c1 must be in range (0,4])')
        
        if c2<0 or c2>4:
            raise Exception('c2 must be in range(0,4]')
        
        if w<0 or w>1:
            raise Exception('w(inertia) must be in range (0,1]')
            
            
        n = len(optimization_problem.lower_constraints)
        l = optimization_problem.lower_constraints
        u = optimization_problem.upper_constraints
        f = optimization_problem.objective_function
        mode = optimization_problem.mode
        t = 0
        
        # minimization problem
        if mode =='min':
            x = np.zeros((N,n+1))
            x[:,:n] = (l)+(u-l)*np.random.uniform(0,1,(N,n))
            x[:,n] = f(x[:,:n])

            v = 0.1*np.random.uniform(0,1,(N,n))
            p = x.copy()
            g_k = p[np.argsort(p[:,-1])[0]][:n]
            
            # best function value so far
            f_k = p[np.argsort(p[:,-1])[0]][-1]
         
            while True:
                r1 = np.random.uniform(0,1,(N,n))
                r2 = np.random.uniform(0,1,(N,n))
                v = w*v+c1*r1*(p[:,:n]-x[:,:n])+c1*r2*(g_k-x[:,:n])
                x[:,:n] = x[:,:n]+v

                idl = np.where((x[:,:n]<l))[0]
                idu = np.where((x[:,:n]>u))[0]

                # maintain feasibility
                if len(idl)>0:
                    x[idl,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idl),n))

                if len(idu)>0:
                    x[idu,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idu),n))

                x[:,n] = f(x[:,:n])
                
                # update personal best
                idx = (x[:,n]<p[:,n])
                p[idx] = x[idx]
                
                # update global best if we have an improvement
                if p[np.argsort(p[:,-1])[0]][-1]<f_k:
                    g_k_copy = np.copy(g_k)
                    g_k = p[np.argsort(p[:,-1])[0]][:n] 
                    f_k = p[np.argsort(p[:,-1])[0]][-1]
                    
                    # compute difference for convergence
                    err = np.max(np.abs(g_k-g_k_copy)) 

                    if err!=0 and err<tol:
                        return g_k,f_k

                    if t>max_iter:
                        print('Failed to converge after',max_iter,'iterations')
                        return g_k,f_k

                t = t+1
                
        # maximization problem
        elif mode=='max':
            x = np.zeros((N,n+1))
            x[:,:n] = (l)+(u-l)*np.random.uniform(0,1,(N,n))
            x[:,n] = f(x[:,:n])

            v = 0.1*np.random.uniform(0,1,(N,n))
            p = x.copy()
            g_k = p[np.argsort(p[:,-1])[-1]][:n]
            
            # best function value so far
            f_k = p[np.argsort(p[:,-1])[-1]][-1]
         
            while True:
                r1 = np.random.uniform(0,1,(N,n))
                r2 = np.random.uniform(0,1,(N,n))
                v = w*v+c1*r1*(p[:,:n]-x[:,:n])+c1*r2*(g_k-x[:,:n])
                x[:,:n] = x[:,:n]+v

                idl = np.where((x[:,:n]<l))[0]
                idu = np.where((x[:,:n]>u))[0]

                # maintain feasibility
                if len(idl)>0:
                    x[idl,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idl),n))

                if len(idu)>0:
                    x[idu,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idu),n))

                x[:,n] = f(x[:,:n])
                
                # update personal best
                idx = (x[:,n]>p[:,n])
                p[idx] = x[idx]
                    
                # update global best if we have an improvement
                if p[np.argsort(p[:,-1])[-1]][-1]>f_k:
                    g_k_copy = g_k.copy()
                    g_k = p[np.argsort(p[:,-1])[-1]][:n] 
                    f_k = p[np.argsort(p[:,-1])[-1]][-1]
                    
                    # compute difference for convergence
                    err = np.max(np.abs(g_k-g_k_copy)) 
                    if err!=0 and err<tol:
                        return g_k,f_k
                
                if t>max_iter:
                    print('Failed to converge after',max_iter,'iterations')
                    return g_k,f_k
                
                t = t+1
                
    def adaptive_pso(self,optimization_problem,N=100,c1=2,c2=2,w_init = 0.9 ,max_iter =300,tol=1e-4):
        ''' 
            adaptive pso optimization we have linearly dropping inertia the parameters are as follows
            
            optimization_problem (OptimizationProblem) optimization problem object with objective function and
            constraints
            N (int) : number of particles in swarm default = 50
            c1 (int) range(0,4]: personal component of particle scales in direction of personal best default = 2,
            c2 (int) range(0,4]: social component of particle scales in direction of global best default = 2
            w_init (float) initial weight value close to 1 and the w will be linearly decreased using equal 
            sized steps from w_init to 0.4 where each step (w_init-0.6)/max_iter 
            max_iter (int): maximum number of iterations to run the algorithm for default 300 
            tol (float) : used as convergence criteria check if we have not converged to a solution uses default 1e-4
            max of abs difference og g(k)-g(k-1)
        '''
        
        # just check input parameters are ok
        if c1<0 or c1>4:
            raise Exception('c1 must be in range (0,4])')
        
        if c2<0 or c2>4:
            raise Exception('c2 must be in range(0,4]')
        
        if w_init<=0.6 and w_init>1:
            raise Exception('w(inertia) must be in range (0.4,1]')
        
        if max_iter<=0:
            raise Exception('max iterations must be integer greater than zero')
            
            
        n = len(optimization_problem.lower_constraints)
        l = optimization_problem.lower_constraints
        u = optimization_problem.upper_constraints
        f = optimization_problem.objective_function
        mode = optimization_problem.mode
        w = np.linspace(w_init,0.4,max_iter)
        
        t = 0
        
        # minimization problem
        if mode =='min':
            x = np.zeros((N,n+1))
            x[:,:n] = (l)+(u-l)*np.random.uniform(0,1,(N,n))
            x[:,n] = f(x[:,:n])

            v = 0.1*np.random.uniform(0,1,(N,n))
            p = x.copy()
            g_k = p[np.argsort(p[:,-1])[0]][:n]
            
            # best function value so far
            f_k = p[np.argsort(p[:,-1])[0]][-1]
         
            while True:
                r1 = np.random.uniform(0,1,(N,n))
                r2 = np.random.uniform(0,1,(N,n))
                v = w[t]*v+c1*r1*(p[:,:n]-x[:,:n])+c1*r2*(g_k-x[:,:n])
                x[:,:n] = x[:,:n]+v

                idl = np.where((x[:,:n]<l))[0]
                idu = np.where((x[:,:n]>u))[0]

                # maintain feasibility
                if len(idl)>0:
                    x[idl,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idl),n))

                if len(idu)>0:
                    x[idu,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idu),n))

                x[:,n] = f(x[:,:n])
                
                # update personal best
                idx = (x[:,n]<p[:,n])
                p[idx] = x[idx]
                
                # update global best if we have an improvement
                if p[np.argsort(p[:,-1])[0]][-1]<f_k:
                    g_k_copy = np.copy(g_k)
                    g_k = p[np.argsort(p[:,-1])[0]][:n] 
                    f_k = p[np.argsort(p[:,-1])[0]][-1]
                    
                    # compute difference for convergence
                    err = np.max(np.abs(g_k-g_k_copy)) 

                    if err!=0 and err<tol:
                        return g_k,f_k
                t = t+1
                if t>=max_iter:
                    print('Failed to converge after',max_iter,'iterations')
                    return g_k,f_k

            
                
        # maximization problem
        elif mode=='max':
            x = np.zeros((N,n+1))
            x[:,:n] = (l)+(u-l)*np.random.uniform(0,1,(N,n))
            x[:,n] = f(x[:,:n])

            v = 0.1*np.random.uniform(0,1,(N,n))
            p = x.copy()
            g_k = p[np.argsort(p[:,-1])[-1]][:n]
            
            # best function value so far
            f_k = p[np.argsort(p[:,-1])[-1]][-1]
         
            while True:
                r1 = np.random.uniform(0,1,(N,n))
                r2 = np.random.uniform(0,1,(N,n))
                v = w[t]*v+c1*r1*(p[:,:n]-x[:,:n])+c1*r2*(g_k-x[:,:n])
                x[:,:n] = x[:,:n]+v

                idl = np.where((x[:,:n]<l))[0]
                idu = np.where((x[:,:n]>u))[0]

                # maintain feasibility
                if len(idl)>0:
                    x[idl,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idl),n))

                if len(idu)>0:
                    x[idu,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idu),n))

                x[:,n] = f(x[:,:n])
                
                # update personal best
                idx = (x[:,n]>p[:,n])
                p[idx] = x[idx]
                    
                # update global best if we have an improvement
                if p[np.argsort(p[:,-1])[-1]][-1]>f_k:
                    g_k_copy = g_k.copy()
                    g_k = p[np.argsort(p[:,-1])[-1]][:n] 
                    f_k = p[np.argsort(p[:,-1])[-1]][-1]
                    
                    # compute difference for convergence
                    err = np.max(np.abs(g_k-g_k_copy)) 
                    if err!=0 and err<tol:
                        return g_k,f_k
                t = t+1
                if t>=max_iter:
                    print('Failed to converge after',max_iter,'iterations')
                    return g_k,f_k
                
                
                