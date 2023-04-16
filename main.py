import numpy as np
import BFGS
import objectiveFunctions

def main():
    X_0 = np.zeros(4*3)
    print(BFGS.BFGS(objectiveFunctions.P5,X_0))

main()