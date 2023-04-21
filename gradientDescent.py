import numpy as np

def gradientDescent(f,gradf,X_0,alpha0=1,c=1e-2,rho=0.1,tol=1e-6,maxSteps=100,maxBacktracking=20):
    k = 0
    X_k = X_0
    val_k, grad_k = f(X_k), gradf(X_k)

    gradNorm_k = np.linalg.norm(grad_k)

    while(k<maxSteps and gradNorm_k>tol):
        alpha_k = alpha0
        X_k1 = X_k - alpha_k*grad_k
        val_k1 = f(X_k1)

        backtrackingSteps = 0
        while (backtrackingSteps < maxBacktracking and val_k1 > val_k - c*alpha_k*gradNorm_k**2):
            alpha_k *= rho
            X_k1 = X_k - alpha_k*grad_k
            val_k1 = f(X_k1)
            backtrackingSteps += 1
        
        X_k = X_k1
        val_k = val_k1
        grad_k = gradf(X_k)

        gradNorm_k = np.linalg.norm(grad_k)

        k += 1
    
    if (k == maxSteps):
        print(f"The algorithm did not converge after {k} iterations.")
    else:
        print(f"The algorithm converged after {k} iterations")
    return X_k

