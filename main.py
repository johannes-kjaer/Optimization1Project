import numpy as np
import BFGS
import objectiveFunctions
import gradientDescent

def P5():
    print(f'--------------------------------------------------\nProblem 5\n--------------------------------------------------')
    ### Testing that the function is yielding the correct values at the analytical solution ###
    #objectiveFunctions.testP5()

    ### Testing the numerical algorithm on problem 5
    X_0 = np.zeros(3*4)
    print('\nUsing BFGS on problem 5.')
    print('X_sol:\n',BFGS.BFGS(objectiveFunctions.P5.getVal,objectiveFunctions.P5.getGrad,X_0))
    print('\nUsing gradient descent on problem 5.')
    print('X_sol:\n',gradientDescent.gradientDescent(objectiveFunctions.P5.getVal,objectiveFunctions.P5.getGrad,X_0,maxSteps=1000,rho=0.9))
    print(f'--------------------------------------------------')

    #print(BFGS.BFGS(objectiveFunctions.testOF.getVal,objectiveFunctions.testOF.getGrad,np.zeros(3))) # This problem seems to produce actual results
    #print(gradientDescent.gradientDescent(objectiveFunctions.testOF.getVal,objectiveFunctions.testOF.getGrad,np.zeros(3)))
    #print(gradientDescent.gradientDescent(objectiveFunctions.P5.getVal,objectiveFunctions.P5.getGrad,X_0,maxSteps=10000))
    

def P9():
    print(f'Problem 9\n--------------------------------------------------')
    #objectiveFunctions.testP9()
    X_0 = np.zeros(3*4)
    X_something = np.array([-0.3,1,8,-1,-0.7,9,7,3,5,1,1,9])
    print(f'\nUsing BFGS on problem 9, with the the inital X_0=\n{X_something}')
    print('X_sol:\n',BFGS.BFGS(objectiveFunctions.P9.getVal,objectiveFunctions.P9.getGrad,X_something,maxItr=10000))
    print(f'\nUsing gradient descent on problem 9, with the the inital X_0=\n{X_something}')
    print('X_sol:\n',gradientDescent.gradientDescent(objectiveFunctions.P9.getVal,objectiveFunctions.P9.getGrad,X_something,maxSteps=10000))
    print(f'--------------------------------------------------')

def main():
    P5()
    P9()

main()