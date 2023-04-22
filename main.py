import numpy as np
import BFGS
import objectiveFunctions
import gradientDescent

def P5():
    ### Testing that the function is yielding the correct values at the analytical solution ###
    #objectiveFunctions.testFunction()

    #print(BFGS.BFGS(objectiveFunctions.testOF,np.array([0,0,0])))
    
    X_0 = np.array([1, 1, 0,-1, 1, 0,-1, -1, 0, 1, -1, 0]) # An initial guess, which should not matter to much as the problem is convex
    X_0 = np.zeros(3*4)
    #X_star = np.array([2, 2, -3/2,-2, 2, -3/2,-2, -2, -3/2,2, -2, -3/2])    # The analytical solution
    print(BFGS.BFGS(objectiveFunctions.P5.getVal,objectiveFunctions.P5.getGrad,X_0))                          # Calling the method and printing the result

    #print(gradientDescent.gradientDescent(objectiveFunctions.P5.getVal,objectiveFunctions.P5.getGrad,X_0,maxSteps=5000))

    #print(BFGS.BFGS(objectiveFunctions.testOF.getVal,objectiveFunctions.testOF.getGrad,np.zeros(3))) # This problem seems to produce actual results
    #print(gradientDescent.gradientDescent(objectiveFunctions.testOF.getVal,objectiveFunctions.testOF.getGrad,np.zeros(3)))
    #print(gradientDescent.gradientDescent(objectiveFunctions.P5.getVal,objectiveFunctions.P5.getGrad,X_0,maxSteps=10000))

def P9():
    X_0 = np.zeros(3*4)
    X_star = np.array([-0.70970,0,9.54287,0,-0.70970,9.54287,0.70970,0,9.54287,0,0.70970,9.54287])
    X_starPertubations = np.array([-1,0,9,0,-1,9,1,0,9,0,1,9])
    X_something = np.array([-0.3,1,8,-1,-0.7,9,7,3,5,1,1,9])
    print(BFGS.BFGS(objectiveFunctions.P9.getVal,objectiveFunctions.P9.getGrad,X_something,maxItr=10000))
    #print(gradientDescent.gradientDescent(objectiveFunctions.P9.getVal,objectiveFunctions.P9.getGrad,X_starPertubations,maxSteps=10000))
    #print(objectiveFunctions.P9.getVal(X_star))
    #print(objectiveFunctions.P9.getGrad(X_star))
    #print(objectiveFunctions.P9.bars)
    #print(objectiveFunctions.P9.cables)
    #print(objectiveFunctions.P9.fixedNodes)
    #print(objectiveFunctions.P9.extWeights)

def main():
    P5()
    P9()

main()