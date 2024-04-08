import numpy as np

def solve_analytical(X,Y):
    Xt=np.transpose(X)
    XtX = Xt@X
    return np.linalg.pinv(XtX)@Xt@Y

def solve_analytical_l2(X,Y,ld):
    Xt=np.transpose(X)
    XtX=Xt@X
    Id=np.eye(XtX.shape[0])
    Id[0][0]=0
    return np.linalg.pinv(ld*Id+XtX)@Xt@Y
