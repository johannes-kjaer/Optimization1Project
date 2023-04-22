import numpy as np

##### Making a objectiveFunction class for all the particularities of the given problems #####
class objectiveFunctionP5:
    '''
    Class to make objective functions, containing the function as well as maybe its gradient and hessian.
    
    Attributes:
    val: The function in question, returns the value of our function
    grad: The gradient of our function
    hess: The hessian of our function
    '''
    def __init__(self, k=1, cables=None, extWeights=None, fixedNodes=None, val=lambda: None, grad=lambda: None):
        self.k = k
        self.cables = cables
        self.extWeights = extWeights
        self.fixedNodes = fixedNodes
        self.val = val
        self.grad = grad
    def getVal(self,X):
        return self.val(X,self.k,self.fixedNodes,self.cables,self.extWeights)
    def getGrad(self,X):
        return self.grad(X,self.k,self.fixedNodes,self.cables,self.extWeights)

class objectiveFunctionP9(objectiveFunctionP5):
    def __init__(self, c=1, grho=1, k=1, bars=None, cables=None, extWeights=None, fixedNodes=None, val=lambda : None, grad=lambda : None):
        super().__init__(k, cables, extWeights, fixedNodes, val, grad)
        self.c = c
        self.grho=grho
        self.bars = bars
    def getVal(self,X):
        return self.val(X,self.c,self.grho,self.k,self.fixedNodes,self.bars,self.cables,self.extWeights)
    def getGrad(self,X):
        return self.grad(X,self.c,self.grho,self.k,self.fixedNodes,self.bars,self.cables,self.extWeights)


########## Problem 5 ##########
def P5val(X,k,fixedNodes,edges,extWeights):
    '''
    Function for calculating the potential energy of the system in a given configuration X.
    Input:
    X: A 3n array with the configuration of nodes to be evaluated, where n is the number of free nodes
    k: A constant pertaining to the elasticity of the cables
    fixedNodes: A 3m array with the positions of the fixed nodes, where m is the number of fixed nodes
    edges: A NxN matrix holding the resting lengths of the cables, where N is the total number of nodes
    extWeights: An array of size n holding the external loads on the free nodes, multiplied by g
    Output:
    The potential energy of the system in the configuration X, float number
    '''
    def P5cables(X,k,fixedNodes,edges):
        '''
        Function for calculating the potential energy due to stretched cables
        Input:
        X: A 3n array with the configuration of nodes to be evaluated, where n is the number of free nodes
        k: A constant pertaining to the elasticity of the cables
        fixedNodes: A 3m array with the positions of the fixed nodes, where m is the number of fixed nodes
        edges: A NxN matrix holding the resting lengths of the cables, where N is the total number of nodes
        Output:
        The potential energy of the system due to stretched cables, float number
        '''
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
        '''
        Function for calculating the potential energy due to external loads
        Input:
        X: A 3n array with the configuration of nodes to be evaluated, where n is the number of free nodes
        extWeights: An array of size n holding the external loads on the free nodes, multiplied by g
        Output:
        The potential energy of the system due to external loads, float number
        '''
        X_3coordinates = (X.reshape((3,X.size//3)))[2]  # Retrieving the z coordinates of the free nodes
        return np.inner(extWeights,X_3coordinates)      # Return the sum of the potential energy due to gravity (Assuming the weights are already mulitplied by g)
    
    return P5cables(X,k,fixedNodes,edges) + P5weights(X,extWeights)   # Return the sum of the potential energy from both the stretched cables, and gravity's pull on the loaded nodes
def P5grad(X,k,fixedNodes,edges,extWeights):
    '''
    Function for calculating the gradient of the system in X.
    Input:
    X: A 3n array with the configuration of nodes to be evaluated, where n is the number of free nodes
    k: A constant pertaining to the elasticity of the cables
    fixedNodes: A 3m array with the positions of the fixed nodes, where m is the number of fixed nodes
    edges: A NxN matrix holding the resting lengths of the cables, where N is the total number of nodes
    extWeights: An array of size n holding the external loads on the free nodes, multiplied by g
    Output:
    The gradient of the system due to stretched cables and external loads
    '''
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
P5 = objectiveFunctionP5(3,P5edges(),P5weights(),P5fixedNodes(),P5val,P5grad) # Gathering the different stationary parts of our problem into one objective function

testOF = objectiveFunctionP5(3,np.array([[0,0,1],[0,0,1],[1,1,0]]),np.array([1]),np.array([1,0,3,-1,0,3]),P5val,P5grad)
########## --------- ##########

########## Problem 9 ##########
def P9val(X,c,grho,k,fixedNodes,bars,cables,extWeights):
    '''
    Function for calculating the potential energy of the system in a given configuration X.
    Input:
    X: A 3n array with the configuration of nodes to be evaluated, where n is the number of free nodes
    c: A constant pertaining to the elasticity of the bars
    grho: The density of the bars multiplied by the gravitational acceleration at earths surface
    k: A constant pertaining to the elasticity of the cables
    fixedNodes: A 3m array with the positions of the fixed nodes, where m is the number of fixed nodes
    bars: A NxN matrix holding the resting lengths of the bars, where N is the total number of nodes
    cables: A NxN matrix holding the resting lengths of the cables, where N is the total number of nodes
    extWeights: An array of size n holding the external loads on the free nodes, multiplied by g
    Output:
    The potential energy of the system in the configuration X, float
    '''

    ### The potential energy from the cables and external loads are the same as if the system consisted of only cables
    EcablesAndExternalMasses = P5val(X,k,fixedNodes,cables,extWeights)

    def P9bars(X,c,grho,fixedNodes,bars):
        '''
        Function for calculating the potential energy due to stretched bars and their weight
        Input:
        X: A 3n array with the configuration of nodes to be evaluated, where n is the number of free nodes
        c: A constant pertaining to the elasticity of the bars
        fixedNodes: A 3m array with the positions of the fixed nodes, where m is the number of fixed nodes
        bars: A NxN matrix holding the resting lengths of the bars, where N is the total number of nodes
        Output:
        The potential energy of the system due to the bars, float
        '''
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
def P9grad(X,c,grho,k,fixedNodes,bars,cables,extWeights):
    '''
    Function for calculating the gradient of the system in X.
    Input:
    X: A 3n array with the configuration of nodes to be evaluated, where n is the number of free nodes
    c: A constant pertaining to the elasticity of the bars
    grho: The density of the bars multiplied by the gravitational acceleration at earths surface
    k: A constant pertaining to the elasticity of the cables
    fixedNodes: A 3m array with the positions of the fixed nodes, where m is the number of fixed nodes
    bars: A NxN matrix holding the resting lengths of the bars, where N is the total number of nodes
    cables: A NxN matrix holding the resting lengths of the cables, where N is the total number of nodes
    extWeights: An array of size n holding the external loads on the free nodes, multiplied by g
    Output:
    The gradient of the system
    '''
    ### The gradient contribution from stretched cables and external loads is the same as calculated for a cable net
    gradCablesAndExtWeight = P5grad(X,k,fixedNodes,cables,extWeights)

    def P9bars(X,c,grho,fixedNodes,bars):
        '''
        Function for calculating the gradient in X due to stretched or compressed bars
        Input:
        X: A 3n array with the configuration of nodes to be evaluated, where n is the number of free nodes
        c: A constant pertaining to the elasticity of the bars
        fixedNodes: A 3m array with the positions of the fixed nodes, where m is the number of fixed nodes
        bars: A NxN matrix holding the resting lengths of the bars, where N is the total number of nodes
        Output:
        The force on X contributed by the stretched and compressed bars
        '''
        M = fixedNodes.size//3  # The number of fixed nodes
        N = M + X.size//3       # The total number of nodes

        allNodes = np.zeros(3*N)        # Gathering the free
        allNodes[fixedNodes.size:] = X           # and the fixed nodes
        allNodes[:fixedNodes.size] = fixedNodes  # into one array

        force_bars = np.zeros(X.size) # The gradient of potential energy is a force
        for i in range(M,N):
            subGradientk = np.zeros(3)  # For each node k (i) the gradient may be calculated separately by calculating the force on the node
            for j in range(N):
                l_ij = bars[i][j]  # Extracting the resting length of the bar between node i and j
                if l_ij > 0:        # If there is a bar between the nodes, then proceed
                    norm_ij = np.linalg.norm(allNodes[i*3:i*3+3] - allNodes[j*3:j*3+3])
                    if norm_ij != 0: # Making sure, one hundred percent, that we are not going to divide by zero
                        forceContribkj = (allNodes[i*3:i*3+3]-allNodes[j*3:j*3+3])  * (1-l_ij/norm_ij) * c/l_ij**2  +  grho*l_ij/2 * np.array([0,0,1])
                        subGradientk += forceContribkj
            force_bars[3*(i-M):3*(i-M+1)] = subGradientk
        return force_bars
    
    gradBars = P9bars(X,c,grho,fixedNodes,bars)
    return gradBars + gradCablesAndExtWeight
def P9edges():
    '''
    Function for producing the matrices containing the resting lengths of the bars and cables
    Output:
    bars: 8x8 matrix containing the resting lenght of the bars
    cables: 8x8 matrix containing the resting lenght of the cables
    '''
    bars = np.zeros((8,8))
    cables = np.zeros((8,8))

    bars[0][4], bars[1][5], bars[2][6], bars[3][7], bars[4][0], bars[5][1], bars[6][2], bars[7][3] = 10, 10, 10, 10, 10, 10, 10, 10
    cables[0][7], cables[1][4], cables[2][5], cables[3][6], cables[7][0], cables[4][1], cables[5][2], cables[6][3]  = 8,8,8,8,8,8,8,8
    cables[4][5], cables[5][6], cables[6][7], cables[4][7], cables[5][4], cables[6][5], cables[7][6], cables[7][4] = 1,1,1,1,1,1,1,1

    return (bars, cables)
def P9weights():
    return np.ones(4) * 0
def P9fixedNodes():
    return np.array(   [[1, 1, 0],
                        [-1, 1, 0],
                        [-1, -1, 0],
                        [1, -1, 0]]
                    ).reshape(3*4)
P9 = objectiveFunctionP9(1,0,0.1,P9edges()[0],P9edges()[1],P9weights(),P9fixedNodes(),P9val,P9grad)

########## Test function ##########
def testFunction(P,X):
    print(f'----------\nE_P5={P.getVal(X)}=={7/6}')  # The value of the function at X^* should be about 7/6
    print(P.getGrad(X),'\n----------') # The gradient should equal zero at X^*


#testFunction()
