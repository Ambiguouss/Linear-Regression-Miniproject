import numpy as np

def least_sqares(X,Y,theta):
    return (np.transpose((X@theta)-Y)@((X@theta)-Y))/X.shape[0]

def absolute(X,Y,theta):
    return(np.sum(np.abs((X@theta)-Y)))/X.shape[0]