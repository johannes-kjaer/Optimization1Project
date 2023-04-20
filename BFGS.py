import numpy as np

def StrongWolfe(f,x,p,
                initial_value,
                initial_descent,
                initial_step_length = 1.0,
                c1 = 1e-2,
                c2 = 0.9,
                max_extrapolation_iterations = 50,
                max_interpolation_iterations = 20,
                rho = 2.0):
    '''
    Implementation of a bisection based bracketing method
    for the strong Wolfe conditions
    '''

    # initialise the bounds of the bracketing interval
    alphaR = initial_step_length
    alphaL = 0.0
    # Armijo condition and the two parts of the Wolfe condition
    # are implemented as Boolean variables
    next_x = x+alphaR*p
    next_value = f.getVal(next_x)
    next_grad = f.getGrad(next_x)
    Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
    descentR = np.inner(p,next_grad)
    curvatureLow = (descentR >= c2*initial_descent)
    curvatureHigh = (descentR <= -c2*initial_descent)
    # We start by increasing alphaR as long as Armijo and curvatureHigh hold,
    # but curvatureLow fails (that is, alphaR is definitely too small).
    # Note that curvatureHigh is automatically satisfied if curvatureLow fails.
    # Thus we only need to check whether Armijo holds and curvatureLow fails.
    itnr = 0
    while (itnr < max_extrapolation_iterations and (Armijo and (not curvatureLow))):
        itnr += 1
        # alphaR is a new lower bound for the step length
        # the old upper bound alphaR needs to be replaced with a larger step length
        alphaL = alphaR
        alphaR *= rho
        # update function value and gradient
        next_x = x+alphaR*p
        next_value = f.getVal(next_x)
        next_grad = f.getGrad(next_x)
        # update the Armijo and Wolfe conditions
        Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
        descentR = np.inner(p,next_grad)
        curvatureLow = (descentR >= c2*initial_descent)
        curvatureHigh = (descentR <= -c2*initial_descent)
    # at that point we should have a situation where alphaL is too small
    # and alphaR is either satisfactory or too large
    # (Unless we have stopped because we used too many iterations. There
    # are at the moment no exceptions raised if this is the case.)

    if itnr == max_extrapolation_iterations:
        print('Extrapolation yielded no suitable interval for the step length')
    alpha = alphaR
    itnr = 0
    # Use bisection in order to find a step length alpha that satisfies
    # all conditions.
    while (itnr < max_interpolation_iterations and (not (Armijo and curvatureLow and curvatureHigh))):
        itnr += 1
        if (Armijo and (not curvatureLow)):
            # the step length alpha was still too small
            # replace the former lower bound with alpha
            alphaL = alpha
        else:
            # the step length alpha was too large
            # replace the upper bound with alpha
            alphaR = alpha
        # choose a new step length as the mean of the new bounds
        alpha = (alphaL+alphaR)/2
        # update function value and gradient
        next_x = x+alphaR*p
        next_value = f.getVal(next_x)
        next_grad = f.getGrad(next_x)
        # update the Armijo and Wolfe conditions
        Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
        descentR = np.inner(p,next_grad)
        curvatureLow = (descentR >= c2*initial_descent)
        curvatureHigh = (descentR <= -c2*initial_descent)
    if itnr == max_interpolation_iterations:
        print('Bisectioning yielded no suitable interval for the step length')
    # return the next iterate as well as the function value and gradient there
    # (in order to save time in the outer iteration; we have had to do these
    # computations anyway)
    return alphaR #next_x,next_value,next_grad

def stepLength(f, X_k, p_k, initAlpha = 1.0, c1 = 1e-2, c2 = 0.9, maxExtItr=50, maxBisItr=20):
    '''
    Function for calculating the step length alpha_k, provided a current X_k and a search_direction p_k.
    Using extrapolation and bisectioning to find a step length satisfying the strong Wolfe conditions.
    Input:
    f: The objective function to minimize
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
    val_k = f.getVal(X_k)       # Precomputing the value at X_k
    grad_k =f.getGrad(X_k)      # Precomputing the current gradient
    grad_k1 = f.getGrad(X_k1)   # Precomputing the gradient at the candidate step

    initDescent = np.inner(p_k,grad_k) # Computing the initial descent along p_k
    descent_k = np.inner(p_k,grad_k1)

    #print(grad_k)
    ### Precomputing the Strong Wolfe conditions as boolean values ###
    #print(f.getVal(X_k1),val_k+c1*alphaUpper*np.inner(grad_k,p_k))
    armijo = (f.getVal(X_k1) <= val_k+c1*alphaUpper*initDescent)
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
        grad_k1 = f.getGrad(X_k1)   # Computing the gradient at the new candidate step

        ### Recomputing the strong wolfe conditions for our new interval upper bound ###
        armijo = f.getVal(X_k1) <= val_k+c1*alphaUpper*np.dot(grad_k,p_k)
        curvatureLow = np.inner(p_k,grad_k1) >= c2*initDescent
        curvatureHigh = np.inner(p_k,grad_k1) <= -c2*initDescent
        #print(armijo,curvatureHigh,curvatureLow)

        iterations += 1
    
    if iterations == maxExtItr:
        print(f'After {iterations} iterations, the interval was still to small to find a suitable alpha_k.')

    #### Using the bisection method to find an alpha_k satisfying the strong Wolfe conditions #####
    alpha_k = alphaUpper # alpha_k is to be our solution, knowing alphaUpper satisfies two of the three conditions, the first alpha_k candidate is set to be alphaUpper

    iterations = 0 #           and (not (Armijo and curvatureLow and curvatureHigh))
    while iterations<maxBisItr and (not (armijo and curvatureHigh and curvatureLow)):
        if armijo and (not curvatureLow):   # Checking if the candidate step length is to small
            alphaLower = alpha_k            # Increasing the lower bound
        else:
            alphaUpper = alpha_k    # If the candidate step length is to large, the upper bound is decreased
        alpha_k = (alphaUpper-alphaLower)/2 # Updating the candidate step length so that it is in the middle of the interval

        ### Updating values at X_k1 ###
        X_k1 = X_k + alpha_k*p_k # Computing the new candidate step
        grad_k1 = f.getGrad(X_k1)   # Computing the gradient at the new candidate step

        ### Recomputing the strong wolfe conditions for our new interval upper bound ###
        armijo = f.getVal(X_k1) <= val_k+c1*alphaUpper*np.dot(grad_k,p_k)
        curvatureLow = np.inner(p_k,grad_k1) >= c2*initDescent
        curvatureHigh = np.inner(p_k,grad_k1) <= -c2*initDescent
        #print(armijo,curvatureHigh,curvatureLow)

        iterations += 1
    
    if iterations == maxBisItr:
        print(f'After {iterations} iterations, bisectioning the interval still yielded no step length satisfying the strong Wolfe conditions.')

    return alpha_k

def BFGS(f,X_0,
         tol=1e-12, maxItr=20):
    
    I = np.identity(X_0.size)
    k, X_k = 0, X_0                                         # Setting initial
    val_k, grad_k, H_k = f.getVal(X_k), f.getGrad(X_k), I   # values

    while np.linalg.norm(grad_k) > tol and k < maxItr:
        p_k = -H_k @ grad_k     # Updating the search direction
        descent_k = np.inner(p_k,grad_k)
        alpha_k = stepLength(f,X_k,p_k,val_k,descent_k) # Finding a step length satisfying the Wolfe conditions
        X_k1 = X_k + alpha_k * p_k  # Taking the step along p_k with step length alpha_k
        
        grad_k1 = f.getGrad(X_k1)   # Finding the gradient at the new X_k

        s_k = X_k1 - X_k            # Finding what the step actually is
        y_k = grad_k1 - grad_k      # Finding the difference between the gradient ahead of the step and after the step
        rho_k = 1/(np.inner(y_k,s_k))   # A recurring expression, precomputed to reduce computations
        H_k1 = (I-rho_k*np.outer(s_k,y_k)) @ (H_k @ (I-rho_k*np.outer(y_k,s_k))) + (rho_k * np.outer(s_k,s_k)) # Updating H_k the BFGS way
        
        k, X_k, val_k, grad_k, H_k = k+1, X_k1, f.getVal(X_k1), grad_k1, H_k1 # Updating values for each step

    return X_k

