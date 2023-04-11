import numpy as np

##### Defining constants #####
#g = 9.81 # m/s^2 # Gravitational acceleration constant
k = 3 # Constant pertaining to the cable elasticity

N = 8 # Number of nodes in total
M = 4 # Number of fixed nodes

##### Making an edge matrix matching the the parameters of the given analytical solution #####
edgesMatrix = np.zeros((N,N))
for i in range(N-1):
    for j in range(i+1,N):
        if j == i+4: # Connecting all fixed nodes with their free node
            edgesMatrix[i][j] = 3
            edgesMatrix[j][i] = 3
        if i >= 4 and j == i+1: # Connecting the free nodes from 5 through 7 to the next free node
            edgesMatrix[i][j] = 3
            edgesMatrix[j][i] = 3
        if i == 4 and j == 7:   # Connecting the nodes 5 and 8
            edgesMatrix[i][j] = 3
            edgesMatrix[j][i] = 3
#print(edgesMatrix)

##### Making an external mass array #####
extMassArr = np.zeros(N)
for i in range(N-M,N): # Loading all free nodes with a mass
    extMassArr = 1/6 # m_i * g

##### Positioning the fixed nodes, as well as giving the free nodes initial positions #####
pArray = np.array([ [5, 5, 0]
                    [-5, 5, 0]
                    [-5, -5, 0]
                    [5, -5, 0]
                ])
xInitialPos = np.zeros((4,3)) # As the problem is convex the inital guess does not matter much, so making it simple the free nodes are all placed at the origin

class objectiveFunction:
    '''
    Class to make objective functions in, containing the function as well as maybe its gradient and hessian.
    
    Attributes:
    val: The function in question, returns the value of our function
    grad: The gradient of our function
    hess: The hessian of our function
    '''
    def __init__(self,
                edges = None,
                extWeights = None,
                nFixedNodes = 0,
                val = lambda: None,
                grad = lambda: None,
                hess = lambda: None
        ):
        self.edges = edges
        self.extWeights = extWeights
        self.nFixedNodes = nFixedNodes
        self.val = val
        self.grad = grad
        self.hess = hess
    
    def getVal(self,X):
        self.val(X,self.edges,self.extWeights)
    
    def getGrad(self,X):
        self.grad(X,self.nFixedNodes,self.edges,self.extWeights)
    
    def getHess(self,X):
        self.hess(X,self.edges)
        
def valEP5(X,edges,extWeights):
    '''
    Input:
    X: 3xN array of N node positions
    edges: NxN array of lengths of cables connecting nodes. Zero lengths means there is no cable there.
    extWeights: Nx1 array of external mass at each node
    
    Output: The energy of the system in the given configuration X
    '''
    n = edges.shape()[0] # Saving the number of nodes in a variable n

    cableElasticityEnergy = 0 # Variable to store the potential energy of the stretched cables
    for i in range(n-1):        # For each node,
        for j in range(i+1,n):  # access all other nodes once, no double counting
            l_ij = edges[i][j]  # Check the resting length of the cable
            if l_ij > 0:        # Checking if there is a cable between these two nodes - a zero resting length implies no cable between them
                norm_ij = np.linalg.norm(X[i]-X[j]) # Check the current length between the nodes
                if norm_ij > l_ij: # If the current length between the nodes is greater than the resting length of the cable, then the cable is stretched and is contributing potential energy
                    cableElasticityEnergy += (norm_ij - l_ij)**2 * k/(2*l_ij**2) #Calculate the potential energy contribution of the cable
                #else:
                #    cableElasticityEnergy += 0 # If the cable is not stretched, it is contributing zero potential energy 

    x_3Coordinates = X[:,2] # Extracting the column of the x_3 coordinates
    extMassEnergy = np.dot(extWeights,x_3Coordinates) # Multiplying and summing all masses multiplied with g, and multiplying with the height above (or below) ground, giving the potential energy contribution from the external masses

    return cableElasticityEnergy + extMassEnergy

def gradEP5(X,nFixNode,edges,extWeights):
    '''
    Input:
    X: 3xN array of n node position
    nFixNode: The number of fixed nodes
    edges: NxN array of lengths of cables connecting nodes. Zero lengths means there is no cable there.
    extWeights: Nx1 array of external mass at each node
    
    Output: The 3X1 array with the gradient of the function at X
    '''
    n = edges.shape()[0] # Saving the number of nodes in a variable n

    cableElasticityForce = np.zeros((n,3)) # Array to store the force from the stretched cables / the gradient of the energy function
    for i in range(n-1):        # For each node,
        for j in range(i+1,n):  # access all other nodes once, no double counting
            l_ij = edges[i][j]  # Check the resting length of the cable
            if l_ij > 0:        # Checking if there is a cable between these two nodes - a zero resting length implies no cable between them
                norm_ij = np.linalg.norm(X[i]-X[j]) # Check the current length between the nodes
                if norm_ij > l_ij: # If the current length between the nodes is greater than the resting length of the cable, then the cable is stretched and is contributing a force
                    forceContribution_ij = (norm_ij - l_ij)*(X[i]-X[j]) * k/l_ij**2 # Calculate the force contribution of the cable
                    cableElasticityForce[i] += - forceContribution_ij   # As x_i - x_j is the vector from j to i, the pull on i from j is going to be the negative of the calculated force
                    cableElasticityForce[j] += forceContribution_ij
                #else:
                #    cableElasticityForce += 0 # If the cable is not stretched, it is contributing no force 

    extMassForce = np.zeros((n,3))
    extMassForce[:,2] = extWeights

    nodesGradient = cableElasticityForce + extMassForce
    if nFixNode > 0:
        nodesGradient[0:nFixNode,:] = np.zeros(nFixNode,3)
    
    return nodesGradient

E = objectiveFunction(edgesMatrix,extMassArr,M,valEP5,gradEP5)

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

    ### Precomputing the Strong Wolfe conditions as boolean values ###
    armijo = f.getval(X_k1) <= val_k+c1*alphaUpper*np.dot(grad_k,p_k)
    curvatureLow = np.inner(p_k,grad_k1) >= c2*initDescent
    curvatureHigh = np.inner(p_k,grad_k1) <= -c2*initDescent

    ##### Finding an interval whose upper end satisfies the armijo condition and the curvatureLow condition, which means the interval is long enough #####
    iterations = 0
    while iterations<maxExtItr and ((armijo and curvatureHigh) and not curvatureLow):
        alphaLower = alphaUpper  # If the interval is to short, then we set the new lower bound to be the former upper bound
        alphaUpper *= multiplier # Exapnding the interval to hopefully now be able to find a satisfactory alpha_k

        ### Updating values at X_k1 ###
        X_k1 = X_k + alphaUpper*p_k # Computing the new candidate step
        grad_k1 = f.getGrad(X_k1)   # Computing the gradient at the new candidate step

        ### Recomputing the strong wolfe conditions for our new interval upper bound ###
        armijo = f.getval(X_k1) <= val_k+c1*alphaUpper*np.dot(grad_k,p_k)
        curvatureLow = np.inner(p_k,grad_k1) >= c2*initDescent
        curvatureHigh = np.inner(p_k,grad_k1) <= -c2*initDescent

        iterations += 1
    
    if iterations == maxExtItr:
        print(f'After {iterations} iterations, the interval was still to small to find a suitable alpha_k.')

    #### Using the bisection method to find an alpha_k satisfying the strong Wolfe conditions #####
    alpha_k = alphaUpper # alpha_k is to be our solution, knowing alphaUpper satisfies two of the three conditions, the first alpha_k candidate is set to be alphaUpper

    iterations = 0
    while iterations<maxBisItr and not (armijo and curvatureHigh and curvatureLow):
        if armijo and (not curvatureLow):   # Checking if the candidate step length is to small
            alphaLower = alpha_k            # Increasing the lower bound
        else:
            alphaUpper = alpha_k    # If the candidate step length is to large, the upper bound is decreased
        alpha_k = (alphaUpper-alphaLower)/2 # Updating the candidate step length so that it is in the middle of the interval

        ### Updating values at X_k1 ###
        X_k1 = X_k + alpha_k*p_k # Computing the new candidate step
        grad_k1 = f.getGrad(X_k1)   # Computing the gradient at the new candidate step

        ### Recomputing the strong wolfe conditions for our new interval upper bound ###
        armijo = f.getval(X_k1) <= val_k+c1*alphaUpper*np.dot(grad_k,p_k)
        curvatureLow = np.inner(p_k,grad_k1) >= c2*initDescent
        curvatureHigh = np.inner(p_k,grad_k1) <= -c2*initDescent

        iterations += 1
    
    if iterations == maxBisItr:
        print(f'After {iterations} iterations, bisectioning the interval still yielded no step length satisfying the strong Wolfe conditions.')

    return alpha_k

def hessianBFGSapprox(H_k, s_k, y_k):
    '''
    A function for updating the hessian BFGS approximation for each step k.
    
    Input:
    H_k: The current BFGS approximation to the hessian
    s_k: The 'step vector', x_k1-x_k
    y_k: The difference in the gradient over the step, del(f_k1) - del(f_k)
    Output:
    H_k1: The updated hessian BFGS approximation
    '''

    I = np.identity(np.size(H_k)[0]) # Constructing the Identity matrix of the same size as our hessian approximation matrix
    rho_k = 1/(y_k.T @ s_k)          # Computing the constant rho_k once
    H_k1 = (I-rho_k* s_k@y_k.T) @ H_k @ (I-rho_k* y_k@s_k.T) + rho_k* s_k@s_k.T # Returning the updated hessian BFGS approximation
    return H_k1

def BFGS(f, X_0, tol=1e-12, maxItr=100):
    '''
    The BFGS optimization method, minimizing a given function, using step lenghts satisfying the strong Wolfe conditions.
    
    Input:
    f: The objective function to minimize
    X_0: An initial guess at the solution
    tol: The tolerance for the gradient deviating from the optimality condition of the gradient being equal to zero.
    maxItr: The maximum number of iterations before the algorithm gives up.
    '''

    X_k, H_k, k = X_0, np.identity(X_0.size()), 0 # Setting initial values

    grad_k = f.getGrad(X_k) # Precomputing the gradient at X_0

    iterations = 0
    while (np.norm(grad_k) > tol) and (maxItr>iterations):
        p_k = -H_k @ grad_k         # Computing the search direction
        alpha_k = stepLength(f,X_k,p_k)     # Computing a step length satisfying the strong Wolfe conditions
        X_k1 = X_k + alpha_k*p_k            # Finding the next candidate solution X
        grad_k1 = f.getGrad(X_k1)           # Computing the gradient at X_{k+1}
        H_k1 = hessianBFGSapprox(H_k,X_k1-X_k,grad_k1)  # Computing the next hessian BFGS approximation

        k, X_k, grad_k, H_k = k+1, X_k1, grad_k1, H_k1  # Updating the variables for a possible next iteration
    if (iterations==maxItr):
        print(f'The algorithm did not converge to any X within the tolerance {tol} in the course of {maxItr} iterations.')
    
    return X_k



    