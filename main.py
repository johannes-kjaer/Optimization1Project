import numpy as np
import BFGS
import objectiveFunctions
import gradientDescent

def main():
    ### Testing that the function is yielding the correct values at the analytical solution ###
    #objectiveFunctions.testFunction()

    #print(BFGS.BFGS(objectiveFunctions.testOF,np.array([0,0,0])))
    
    X_0 = np.array([1, 1, 0,-1, 1, 0,-1, -1, 0, 1, -1, 0]) # An initial guess, which should not matter to much as the problem is convex
    #X_star = np.array([2, 2, -3/2,-2, 2, -3/2,-2, -2, -3/2,2, -2, -3/2])    # The analytical solution
    print(BFGS.BFGS(objectiveFunctions.P5.getVal,objectiveFunctions.P5.getGrad,X_0))                          # Calling the method and printing the result

    #print(gradientDescent.gradientDescent(objectiveFunctions.P5.getVal,objectiveFunctions.P5.getGrad,X_0,maxSteps=5000))

    #print(BFGS.BFGS(objectiveFunctions.testOF.getVal,objectiveFunctions.testOF.getGrad,np.zeros(3))) # This problem seems to produce actual results
    #print(gradientDescent.gradientDescent(objectiveFunctions.testOF.getVal,objectiveFunctions.testOF.getGrad,np.zeros(3)))

main()