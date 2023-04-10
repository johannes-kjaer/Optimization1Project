import numpy as np

##### Defining constants #####
g = 9.81 # m/s^2 # Gravitational acceleration constant
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
                val = lambda: None,
                grad = lambda: None,
                hess = lambda: None
        ):
        self.edges = edges
        self.extWeights = extWeights
        self.val = val
        self.grad = grad
        self.hess = hess
    
    def getVal(self,X):
        self.val(X,self.edges,self.extWeights)
    
    def getGrad(self,X):
        self.grad(X,self.edges,self.extWeights)
    
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

    return cableElasticityForce + extMassForce
    
    

