import numpy as np

##### Making a objectiveFunction class for all the particularities of the given problems #####
class objectiveFunction:
    '''
    Class to make objective functions, containing the function as well as maybe its gradient and hessian.
    
    Attributes:
    val: The function in question, returns the value of our function
    grad: The gradient of our function
    hess: The hessian of our function
    '''
    def __init__(self,
                k = 1,
                edges = None,
                extWeights = None,
                fixedNodes = None,
                val = lambda: None,
                grad = lambda: None
        ):
        self.k = k
        self.edges = edges
        self.extWeights = extWeights
        self.fixedNodes = fixedNodes
        self.val = val
        self.grad = grad
    def getVal(self,X):
        return self.val(X,self.k,self.fixedNodes,self.edges,self.extWeights)
    def getGrad(self,X):
        return self.grad(X,self.k,self.fixedNodes,self.edges,self.extWeights)

########## Problem 5 ##########
def P5val(X,k,fixedNodes,edges,extWeights):
    def P5cables(X,k,fixedNodes,edges):
        M = fixedNodes.size//3 # The number of fixed nodes
        N = X.size//3 + M
        allNodes = np.zeros(3*N)        # Gathering the free
        allNodes[fixedNodes.size:] = X           # and the fixed nodes
        allNodes[:fixedNodes.size] = fixedNodes  # into one array

        E_pot = 0   # Initializing a variable for storing the potential energy
        for i in range(N-1):
            for j in range(i+1,N):
                l_ij = edges[i][j]                          # Extracting the resting length of the cable
                if l_ij > 0:                                # Only proceed if there is a cable between the nodes
                    norm_ij = np.linalg.norm(allNodes[3*i:3*i+3]-allNodes[3*j:3*j+3])   # Compute the length between the nodes
                    #print(f'i:{i}, j:{j}, norm_ij:{norm_ij}, l_ij:{l_ij}, ||{allNodes[3*i:3*i+3]}-{allNodes[3*j:3*j+3]}||')
                    if norm_ij > l_ij:                      # Only continue if the cable is stretched, and is creating a force
                        E_pot += ((norm_ij-l_ij)/l_ij)**2   # Add the potential energy contribution from the stretched cable
        return E_pot * k/2      # Return the potential energy from the stretched cables
    
    def P5weights(X,extWeights):
        X_3coordinates = (X.reshape((3,X.size//3)))[2]  # Retrieving the z coordinates of the free nodes
        return np.inner(extWeights,X_3coordinates)      # Return the sum of the potential energy due to gravity (Assuming the weights are already mulitplied by g)
    
    return P5cables(X,k,fixedNodes,edges) + P5weights(X,extWeights)   # Return the sum of the potential energy from both the stretched cables, and gravity's pull on the loaded nodes
def P5grad(X,k,fixedNodes,edges,extWeights):
    def P5cables(X,k,fixedNodes,edges):
        M = fixedNodes.size//3  # The number of fixed nodes
        N = M + X.size//3       # The total number of nodes

        allNodes = np.zeros(3*N)        # Gathering the free
        allNodes[fixedNodes.size:] = X           # and the fixed nodes
        allNodes[:fixedNodes.size] = fixedNodes  # into one array

        force_cables = np.zeros(X.size) # The gradient of potential energy is a force
        for i in range(M,N):
            subGradientk = np.zeros(3)  # For each node k (i) the gradient may be calculated separately by calculating the force on the node
            for j in range(N):
                l_ij = edges[i][j]  # Extracting the resting length of the cable between node i and j
                if l_ij > 0:        # If there is a cable between the nodes, then proceed
                    norm_ij = np.linalg.norm(allNodes[i*3:i*3+3] - allNodes[j*3:j*3+3])
                    if norm_ij > l_ij:
                        forceContribkj = (allNodes[i*3:i*3+3]-allNodes[j*3:j*3+3]) *(1-l_ij/norm_ij) *k/l_ij**2
                        subGradientk += forceContribkj
            force_cables[3*(i-M):3*(i-M+1)] = subGradientk
        return force_cables 

    def P5weights(extWeights):
        gravitationalPull = np.zeros((3,extWeights.size))   # Creating an array for the gravitational pull on each node, in each direction (all except the z direction will be zero)
        gravitationalPull[2] += extWeights  # Extracting the z coordinate for each node and adding the gravitational pull
        return (gravitationalPull.T).reshape(3*extWeights.size) # Returning the gravitational pull array
    
    return P5cables(X,k,fixedNodes,edges) + P5weights(extWeights)
def P5edges():
    '''
    Function for assembling a matrix A of the "edges" in the system, 
    more specifically the resting length of the cable or bar connecting the nodes i and j is stored at A[i][j] (And A[j][i]).
    Assumes the fixed nodes are indexed 1 through 4 (0 through 3), and the free nodes 5 through 8 (4 through 7).
    Output:
    edgesMatrix: Matrix of the edges in the system
    '''
    N = 8
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
    return edgesMatrix
def P5weights():
    '''
    Function for assembling an array of the weights on the free nodes
    Output: Array of weights
    '''
    freeNodes = 4
    return np.ones(freeNodes) * 1/6
def P5fixedNodes():
    '''
    Function for assembling the array of fixed nodes, as a one dimensional array.
    Output: Array of fixed nodes'''
    M = 4
    return np.array([[5, 5, 0],
                     [-5, 5, 0],
                     [-5, -5, 0],
                     [5, -5, 0]
    ]).reshape(M*3)
P5 = objectiveFunction(3,P5edges(),P5weights(),P5fixedNodes(),P5val,P5grad) # Gathering the different stationary parts of our problem into one objective function
########## --------- ##########

########## Problem 9 ##########
def P9val(X,c,grho,k,fixedNodes,bars,cables,extWeights):
    ### The potential energy from the cables and external loads are the same as if the system consisted of only cables
    EcablesAndExternalMasses = P5val(X,k,fixedNodes,cables,extWeights)

    def P9bars(X,c,grho,fixedNodes,bars):
        M = fixedNodes.size//3 # The number of fixed nodes
        N = X.size//3 + M
        allNodes = np.zeros(3*N)                 # Gathering the free
        allNodes[fixedNodes.size:] = X           # and the fixed nodes
        allNodes[:fixedNodes.size] = fixedNodes  # into one array

        E_pot = 0   # Initializing a variable for storing the potential energy
        for i in range(N-1):
            for j in range(i+1,N):
                l_ij = bars[i][j]                           # Extracting the resting length of the bar
                if l_ij > 0:                                # Only proceed if there is a bar between the nodes
                    norm_ij = np.linalg.norm(allNodes[3*i:3*i+3]-allNodes[3*j:3*j+3])   # Compute the length between the nodes
                    E_pot += c/(2*l_ij**2) * (norm_ij - l_ij)**2 + grho*l_ij/2 * (allNodes[3*i+2]+allNodes[3*j+2])
        return E_pot
    Ebars = P9bars(X,c,grho,fixedNodes,bars)

    return Ebars + EcablesAndExternalMasses

########## Test function ##########
def testFunction():
    X_star = np.array([2, 2, -3/2,-2, 2, -3/2,-2, -2, -3/2,2, -2, -3/2]) # The analytical solution to the problem.
    print(f'----------\nE_P5={P5.getVal(X_star)}=={7/6}')  # The value of the function at X^* should be about 7/6
    print(P5.getGrad(X_star),'\n----------') # The gradient should equal zero at X^*


#testFunction()

testOF = objectiveFunction(3,np.array([[0,0,1],[0,0,1],[1,1,0]]),np.array([1]),np.array([1,0,3,-1,0,3]),P5val,P5grad)