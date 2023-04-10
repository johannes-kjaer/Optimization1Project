import numpy as np

##### Defining constants #####
g = 9.81 # m/s^2 # Gravitational acceleration constant
k = 1 # Constant pertaining to the cable elasticity


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
    X: 3Xn array of n node positions
    edges: nXn array of lengths of cables connecting nodes. Zero lengths means there is no cable there.
    extWeights: nx1 array of external mass at each node
    
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
    extMassEnergy = g * np.dot(extWeights,x_3Coordinates) # Multiplying and summing all masses times the height above (or below) ground, and multiplying with g, giving the potential energy contribution from the external masses

    return cableElasticityEnergy + extMassEnergy

def gradEP5(X,edges,extWeights):
    '''
    Input:
    X: 3Xn array of n node positions
    edges: nXn array of lengths of cables connecting nodes. Zero lengths means there is no cable there.
    extWeights: nx1 array of external mass at each node
    
    Output: The 3X1 array with the gradient of the function at X
    '''
    n = edges.shape()[0] # Saving the number of nodes in a variable n

    cableElasticityForce = np.zeros(3) # Vector to store the force from the stretched cables / the gradient of the energy function
    for i in range(n-1):        # For each node,
        for j in range(i+1,n):  # access all other nodes once, no double counting
            l_ij = edges[i][j]  # Check the resting length of the cable
            if l_ij > 0:        # Checking if there is a cable between these two nodes - a zero resting length implies no cable between them
                norm_ij = np.linalg.norm(X[i]-X[j]) # Check the current length between the nodes
                if norm_ij > l_ij: # If the current length between the nodes is greater than the resting length of the cable, then the cable is stretched and is contributing a force
                    cableElasticityForce += (norm_ij - l_ij)*(X[i]-X[j]) * k/l_ij**2 # Calculate the force contribution of the cable
                #else:
                #    cableElasticityForce += 0 # If the cable is not stretched, it is contributing no force 

    extMassForce = np.sum(extWeights[:,2]) * g # Summing the weights multiplied by g, which is the force, which is the derivative of the energy with respect to the spatial domain

    return cableElasticityForce + np.array([0,0,extMassForce])


