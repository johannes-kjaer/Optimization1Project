import numpy as np
import BFGS
import objectiveFunctions

def main():
    objectiveFunctions.testFunction()
    #X_0 = np.zeros(4*3)
    #print(BFGS.BFGS(objectiveFunctions.P5,X_0))

    #print(BFGS(tesF,np.array([0,0,0])))

main()