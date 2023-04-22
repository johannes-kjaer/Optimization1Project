import numpy as np

def stepLength(f_val,f_grad, X_k, p_k, initAlpha = 1.0, c1 = 1e-2, c2 = 0.99, maxExtItr=50, maxBisItr=20):
    '''
    Function for calculating the step length alpha_k, provided a current X_k and a search_direction p_k.
    Using extrapolation and bisectioning to find a step length satisfying the strong Wolfe conditions.
    Input:
    f_val: The objective function to minimize
    f_grad: The gradient of the objective function
    X_k: The current iterate candidate solution
    p_k: The search direction to calculate a step in
    Optional:
    initAlpha: The step length to first test if is satisfying the strong Wolfe conditions
    c1: A constant pertaining to the Aramijo condition, satisfying 0 < c1 < 1
    c2: A constant pertaining to the curvature condition, satisfying c1 < c2 < 1
    Output:
    alpha_k: A step length along p_k satisfying the strong Wolfe conditions
    '''

    multiplier = 2.0            # If the interval is to small, it will be expanded by mulitplying the upper bound with this
    alphaUpper = initAlpha      # Initial upper bound for alpha, set equal to the default step length
    alphaLower = 0              # Lower bound for alpha
    X_k1 = X_k + alphaUpper*p_k # Precomputing the candidate step
    val_k = f_val(X_k)       # Precomputing the value at X_k
    grad_k =f_grad(X_k)      # Precomputing the current gradient
    grad_k1 = f_grad(X_k1)   # Precomputing the gradient at the candidate step

    initDescent = np.inner(grad_k,p_k) # Computing the initial descent along p_k
    descent_k = np.inner(grad_k1,p_k)


    #print(grad_k)
    ### Precomputing the Strong Wolfe conditions as boolean values ###
    #print(f.getVal(X_k1),val_k+c1*alphaUpper*np.inner(grad_k,p_k))
    armijo = (f_val(X_k1) <= val_k+c1*alphaUpper*initDescent)
    curvatureLow = (descent_k >= c2*initDescent)
    curvatureHigh = (descent_k <= -c2*initDescent)
    #print(armijo,curvatureHigh,curvatureLow)

    ##### Finding an interval whose upper end satisfies the armijo condition and the curvatureLow condition, which means the interval is long enough #####
    iterations = 0
    while iterations<maxExtItr and (armijo and (not curvatureLow)):
        alphaLower = alphaUpper  # If the interval is to short, then we set the new lower bound to be the former upper bound
        alphaUpper *= multiplier # Expanding the interval to hopefully now be able to find a satisfactory alpha_k

        ### Updating values at X_k1 ###
        X_k1 = X_k + alphaUpper*p_k # Computing the new candidate step
        grad_k1 = f_grad(X_k1)   # Computing the gradient at the new candidate step

        ### Recomputing the strong wolfe conditions for our new interval upper bound ###
        armijo = f_val(X_k1) <= val_k+c1*alphaUpper*np.inner(p_k,grad_k)
        curvatureLow = np.inner(p_k,grad_k1) >= c2*initDescent
        curvatureHigh = np.inner(p_k,grad_k1) <= -c2*initDescent
        #print(armijo,curvatureHigh,curvatureLow)

        iterations += 1
    
    #if iterations == maxExtItr:
    #    print(f'After {iterations} iterations, the interval was still to small to find a suitable alpha_k.')

    #### Using the bisection method to find an alpha_k satisfying the strong Wolfe conditions #####
    alpha_k = alphaUpper # alpha_k is to be our solution, knowing alphaUpper satisfies two of the three conditions, the first alpha_k candidate is set to be alphaUpper

    iterations = 0 #           
    while iterations<maxBisItr and (not (armijo and curvatureHigh and curvatureLow)):
        if not (armijo and curvatureLow):   # Checking if the candidate step length is to small
            alphaLower = alpha_k            # Increasing the lower bound
        else:
            alphaUpper = alpha_k    # If the candidate step length is to large, the upper bound is decreased
        alpha_k = (alphaUpper-alphaLower)/2 # Updating the candidate step length so that it is in the middle of the interval

        ### Updating values at X_k1 ###
        X_k1 = X_k + alpha_k*p_k # Computing the new candidate step
        grad_k1 = f_grad(X_k1)   # Computing the gradient at the new candidate step

        ### Recomputing the strong wolfe conditions for our new interval upper bound ###
        armijo = (f_val(X_k1) <= val_k+c1*alphaUpper*np.inner(grad_k,p_k))
        curvatureLow = np.inner(p_k,grad_k1) >= c2*initDescent
        curvatureHigh = np.inner(p_k,grad_k1) <= -c2*initDescent
        #print(armijo,curvatureHigh,curvatureLow)

        iterations += 1

    #if iterations == maxBisItr:
    #    print(f'After {iterations} iterations, bisectioning the interval still yielded no step length satisfying the strong Wolfe conditions.')

    return alpha_k


def BFGS(f_val,f_grad,X_0,
         tol=1e-6, maxItr=50):
    
    I = np.identity(X_0.size)
    k, X_k = 0, X_0                                         # Setting initial
    val_k, grad_k, H_k = f_val(X_k), f_grad(X_k), I   # values

    while np.linalg.norm(grad_k) > tol and k < maxItr:
        p_k = -H_k @ grad_k     # Updating the search direction
        descent_k = np.inner(p_k,grad_k)
        alpha_k = stepLength(f_val,f_grad,X_k,p_k,val_k,descent_k) # Finding a step length satisfying the Wolfe conditions
        X_k1 = X_k + alpha_k * p_k  # Taking the step along p_k with step length alpha_k
        
        grad_k1 = f_grad(X_k1)   # Finding the gradient at the new X_k

        s_k = X_k1 - X_k            # Finding what the step actually is
        y_k = grad_k1 - grad_k      # Finding the difference between the gradient ahead of the step and after the step
        rho_k = 1/(np.inner(y_k,s_k))   # A recurring expression, precomputed to reduce computations
        H_k1 = (I-rho_k*np.outer(s_k,y_k)) @ (H_k @ (I-rho_k*np.outer(y_k,s_k))) + (rho_k * np.outer(s_k,s_k)) # Updating H_k the BFGS way
        
        k, X_k, val_k, grad_k, H_k = k+1, X_k1, f_val(X_k1), grad_k1, H_k1 # Updating values for each step

    if k == maxItr:
        print(f'The algorithm did not converge after {k} iterations. norm={np.linalg.norm(grad_k)}>{tol}=tol')
    else:
        print(f'The algorithm converged after {k} iterations.')

    return X_k

