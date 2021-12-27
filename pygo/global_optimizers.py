import numpy as np

class GlobalOptimizer:
    def pso(self,optimization_problem,N=100,c1=2,c2=2,w = 0.6 ,max_iter =300,tol=1e-4,trajectory=False):
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
            trajectory (bool) : whether to return a 2d array of optimal solution and f value after each iteration 
        '''
        
        # just check input parameters are ok
        if c1<0 or c1>4:
            raise Exception('c1 must be in range (0,4])')
        
        if c2<0 or c2>4:
            raise Exception('c2 must be in range(0,4]')
        
        if w<0 or w>1:
            raise Exception('w(inertia) must be in range (0,1]')
        
        

        n = optimization_problem.n
        f = optimization_problem.f
        mode = optimization_problem.mode
        l = optimization_problem.lower_constraints
        u = optimization_problem.upper_constraints
        feasibility = optimization_problem.feasibility
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

            # function values
            x_path = np.zeros((1,len(g_k)))
            x_path[0] = g_k
            y_path = np.array([f_k]) 

            while True:
                r1 = np.random.uniform(0,1,(N,n))
                r2 = np.random.uniform(0,1,(N,n))
                v = w*v+c1*r1*(p[:,:n]-x[:,:n])+c1*r2*(g_k-x[:,:n])
                x[:,:n] = x[:,:n]+v

            
                if feasibility:
                    idl = np.where((x[:,:n]<l))[0]
                    idu = np.where((x[:,:n]>u))[0]

                    if len(idl)>0:
                        x[idl,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idl),n))

                    if len(idu)>0:
                        x[idu,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idu),n))

                x[:,n] = f(x[:,:n])
                
                # update personal best
                idx = (x[:,n]<p[:,n])
                p[idx] = x[idx]

                # new potential solution
                g_k_new = p[np.argsort(p[:,-1])[0]][:n]
                f_k_new = p[np.argsort(p[:,-1])[0]][-1]

                # add to path
                x_path = np.vstack((x_path,g_k_new))
                y_path = np.append(y_path,f_k_new)

                # update global best if we have an improvement
                if f_k_new<f_k:
                    g_k_copy = np.copy(g_k)
                    g_k = g_k_new 
                    f_k = f_k_new
                    

                    # compute difference for convergence
                    err = np.max(np.abs(g_k-g_k_copy)) 

                    if err!=0 and err<tol:
                        if trajectory:
                            return g_k,f_k,x_path,y_path
                        else:
                            return g_k,f_k

                    if t>max_iter:
                        print('Failed to converge after',max_iter,'iterations')
                        if trajectory:
                            return g_k,f_k,x_path,y_path
                        else:
                            return g_k,f_k

                t = t+1
        else:
            def f_new(x):
                return -1*f(x)
            optimization_problem_new = optimization_problem
            optimization_problem_new.mode = 'min'
            optimization_problem_new.f = f_new
            g_k,f_k = self.pso(optimization_problem=optimization_problem_new,N=N,c1=c1,c2=c2,w = w ,max_iter=max_iter,tol=tol,trajectory=trajectory)
            return g_k,-1*f_k

                
    def adaptive_pso(self,optimization_problem,N=100,c1=2,c2=2,w_init=0.9,max_iter=300,tol=1e-4,trajectory=False):
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
             trajectory (bool) : whether to return a 2d array of optimal solution and f value after each iteration 
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
        f = optimization_problem.f
        mode = optimization_problem.mode
        feasibility = optimization_problem.feasibility
        
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


            # function values
            x_path = np.zeros((1,len(g_k)))
            x_path[0] = g_k
            y_path = np.array([f_k]) 
         
            while True:
                r1 = np.random.uniform(0,1,(N,n))
                r2 = np.random.uniform(0,1,(N,n))
                v = w[t]*v+c1*r1*(p[:,:n]-x[:,:n])+c1*r2*(g_k-x[:,:n])
                x[:,:n] = x[:,:n]+v

                # maintain feasibility
                if feasibility:

                    idl = np.where((x[:,:n]<l))[0]
                    idu = np.where((x[:,:n]>u))[0]
                    if len(idl)>0:
                        x[idl,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idl),n))

                    if len(idu)>0:
                        x[idu,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idu),n))

                x[:,n] = f(x[:,:n])

                # update personal best
                idx = (x[:,n]<p[:,n])
                p[idx] = x[idx]
                

                # new potential solution
                g_k_new = p[np.argsort(p[:,-1])[0]][:n]
                f_k_new = p[np.argsort(p[:,-1])[0]][-1]
                
                 # add to path
                x_path = np.vstack((x_path,g_k_new))
                y_path = np.append(y_path,f_k_new)

                # update global best if we have an improvement
                if f_k_new<f_k:
                    g_k_copy = np.copy(g_k)
                    g_k = g_k_new 
                    f_k = f_k_new
                    # compute difference for convergence
                    err = np.max(np.abs(g_k-g_k_copy)) 

                    if err!=0 and err<tol:
                        if trajectory:
                            return g_k,f_k,x_path,y_path
                        
                        else:
                            return g_k,f_k
                t = t+1
                if t>=max_iter:
                    print('Failed to converge after',max_iter,'iterations')
                    if trajectory:
                        return g_k,f_k,x_path,y_path
                    
                    else:
                        return g_k,f_k
    
        # maximization problem
        else:
            def f_new(x):
                return -1*f(x)
            optimization_problem_new = optimization_problem
            optimization_problem_new.mode = 'min'
            optimization_problem_new.f = f_new
            g_k,f_k = self.adaptive_pso(optimization_problem=optimization_problem,N=N,c1=c1,c2=c2,w_init=w_init,max_iter=max_iter,tol=tol,trajectory=trajectory)
            return g_k,-1*f_k

            
                
    def ga(self,optimization_problem,N=100,m=10,mutation=False,max_iter=300,tol=1e-3,trajectory=False):
        
        ''' genetic algorithm we evolve the population by replacing m least fit individuals with offspring
            of mating pair using tournament selection parameters are as follows
            
            optimization_problem (OptimizationProblem) optimization problem object with objective function and
                constraints
            
            N (int): number of indivduals in population default 100
            m (int): number of individuals that are replaced each generation default 10
            mutation (bool): indicates whether mutation occurs in the population or not
            max_iter (int): indicates maximum number of generations that can occur
            tol (float): used to check convergence by using standard deviation of function values
            
        '''
        n = optimization_problem.n
        l = optimization_problem.lower_constraints
        u = optimization_problem.upper_constraints
        f = optimization_problem.f
        mode = optimization_problem.mode
        feasibility = optimization_problem.feasibility
        
        p = np.zeros((N,n+1))
        p[:,:n] = l+(u-l)*np.random.uniform(0,1,(N,n))
        p[:,n] = f(p[:,:n])
        c = np.zeros((m,n+1))
        
        x_path = np.zeros((0,n))
        y_path = np.array([])

        # minimization problem
        if mode=='min':
            for g in range(max_iter):
                # sorting in descending order so we later replace top m
                p = p[np.argsort(p[:,-1])[::-1]]

                x_g = p[-1,:n]
                y_g = p[-1,-1]

                x_path = np.vstack((x_path,x_g))
                y_path = np.append(y_path,y_g)
                
                for k in range(0,m,2):
                    # we will pick the pair using this
                    L = np.zeros(2).astype(int) 
                    n1 , n2 = 1,1
                    
                    # tournament selection generate 2 indices and compare
                    for j in range(2):
                        while n1==n2:
                            n1 = int(np.random.uniform(0,1)*N)
                            n2 = int(np.random.uniform(0,1)*N)

                        if p[n1,n]<p[n1,n]:
                            L[j] = n1

                        else:
                            L[j] = n2
                        n1 = n2

                    # now do the crossing over
                    alpha = -1/2+(2)*np.random.uniform(0,1,n)
                    beta = -1/2+(2)*np.random.uniform(0,1,n)
                  
                    x = alpha*(p[L[0],:n])+(1-alpha)*(p[L[1],:n])
                    y = (1-beta)*(p[L[0],:n])+beta*(p[L[1],:n])
                    
                    # if mutation is allowed
                    if mutation==True:
                        r1 = np.random.uniform(0,1)
                        r2 = np.random.uniform(0,1)
                        
                        if r1<0.1:
                            idx = int(np.random.uniform(0,1)*n)
                            x[idx] = x[idx]*np.random.uniform(0,1)
                        
                        if r2<0.1:
                            idy = int(np.random.uniform(0,1)*n)
                            y[idy] = y[idy]*np.random.uniform(0,1)
                        
                    # children       
                    c[k,:n] = x
                    c[k+1,:n] = y
                    
                    if feasibility:
                        # infeasible solutions
                        idl = np.where((c[:,:n]<l))[0]
                        idu = np.where((c[:,:n]>u))[0]

                        # maintain feasibility
                        if len(idl)>0:
                            c[idl,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idl),n))

                        if len(idu)>0:
                            c[idu,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idu),n))

                    c[k:k+2,n] = f(c[k:k+2,:n])            

                p[:m,:] = c
                p[:,n] = f(p[:,:n])
                
                if np.std(p[:,-1])<tol:
                    p = p[np.argsort(p[:,-1])[::-1]]  
                    x_g = p[-1,:n]
                    y_g = p[-1,-1]
                    x_path = np.vstack((x_path,x_g))
                    y_path = np.append(y_path,y_g)
                    if trajectory:
                        return x_g,y_g,x_path,y_path
                    
                    else:
                        return x_g,y_g
                    
            
            print('Failed to converge after',max_iter,'iterations')
            p = p[np.argsort(p[:,-1])[::-1]]

            x_g = p[-1,:n]
            y_g = p[-1,-1]
            x_path = np.vstack((x_path,x_g))
            y_path = np.append(y_path,y_g)
            if trajectory:
                return x_g,y_g,x_path,y_path
            
            else:
                return x_g,y_g  
            
        # maximization problem
        else:
            def f_new(x):
                return -1*f(x)
            optimization_problem_new = optimization_problem
            optimization_problem_new.mode = 'min'
            optimization_problem_new.f = f_new
            x_g,y_g = self.ga(optimization_problem=optimization_problem_new,N=N,m=m,mutation=mutation,max_iter=max_iter,tol=tol,trajectory=trajectory)
            return x_g,-1*y_g
    
    def de(self,optimization_problem,N=100,F=0.8,CR=0.9,max_iter=300,tol=1e-4,trajectory=False):        
        ''' differential evolution 
            optimization_problem (OptimizationProblem) optimization problem object with objective function and
            constraints
            N (int) : number of candidate solutions
            F (float) range(0,2): differential weight default 0.8
            CR (float) range(0,1): crossover probability
            max_iter (int): maximum number of iterations to run the algorithm for default = 300
            tol (float) : used as convergence criteria check if we have not converged to a solution  1e-4
            checks standard deviation
        
        '''
        
        if F<0 or F>2:
            raise Exception('F must be in range [0,2] (differential weight)')
            
        if CR<0 or CR>1:
            raise Exception('CR muste be in range [0,1]')
            
        n = optimization_problem.n
        l = optimization_problem.lower_constraints
        u = optimization_problem.upper_constraints
        f = optimization_problem.f
        mode = optimization_problem.mode

        x_path = np.zeros((1,n))
        y_path = np.array([])

        if mode == 'min':
            p = np.zeros((N,n+1))
            p[:,:n] = (l)+(u-l)*np.random.uniform(0,1,(N,n))
            p[:,n] = f(p[:,:n])
            x_path[0] = p[0,:n]
            y_path = np.append(y_path,p[0,-1])
            for g in range(max_iter):
                y = np.zeros((N,n+1))
                for i in range(N):
                    # generate candidates
                    while True:
                        c = np.random.uniform(0,N,3).astype(int)    
                        if len(np.unique(c))==3 and i not in c:
                            break
                    # compute the agent's new position
                    a = p[c[0],:n]
                    b = p[c[1],:n]
                    c = p[c[2],:n]

                    R = int(np.random.uniform(0,n))
                    r = np.random.uniform(0,1,n)
                    idl = np.where(r<CR)[0]
                    idu = np.where(r>CR)[0]
                    y[i,idl] = a[idl]+F*(b[idl]-a[idl])
                    y[i,idu] = p[i,idu]
                    y[i,R] = a[R]+F*(b[R]-a[R]) 

                idl = np.where((y[:,:n]<l))[0]
                idu = np.where((y[:,:n]>u))[0]

            
                if len(idl)>0:
                    y[idl,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idl),n))

                if len(idu)>0:
                    y[idu,:n] = (l)+(u-l)*np.random.uniform(0,1,(len(idu),n))

                # compute new function values
                y[:,n] =f(y[:,:n])
                idx = (y[:,n]<=p[:,n])
                p[idx] = y[idx]
                

                p = p[np.argsort(p[:,-1])]
                
                x_g = p[0,:n]
                y_g = p[0,-1]
                x_path = np.vstack((x_path,x_g))
                y_path = np.append(y_path,y_g)
                
                if np.std(p[:,-1])<tol:
                   
                    if trajectory:
                        return x_g,y_g,x_path,y_path

                    else:
                        return x_g,y_g


            print('Failed to converger after',max_iter,'iterations')
            if trajectory:
                return x_g,y_g,x_path,y_path
            else:
                return x_g,y_g

        # maximization problem
        elif mode == 'max':
            def f_new(x):
                return -1*f(x)
            optimization_problem_new = optimization_problem
            optimization_problem_new.mode = 'min'
            optimization_problem_new.f = f_new
            x_g,y_g = self.de(optimization_problem=optimization_problem_new,N=N,F=F,CR=CR,max_iter=max_iter,tol=tol,trajectory=trajectory)
            return x_g,-1*y_g
                    