import numpy as np
import BFGS
import objectiveFunctions

def main():
    ### Testing that the function is yielding the correct values at the analytical solution ###
    objectiveFunctions.testFunction()

    print(BFGS.BFGS(objectiveFunctions.testOF,np.array([0,0,0])))
    
    X_0 = np.zeros(4*3)
    #print(BFGS.BFGS(objectiveFunctions.P5,X_0))

main()