import numpy as np


def parametrs_standard(data):
    column_means = np.mean(data, axis=0)
    column_deviation = np.std(data, axis=0)
    return (column_means,column_deviation)

def scale(data,mean,deviation):
    return (add_ones((data-mean)/deviation))


def min_max(data):
    minimum = np.min(data,axis=0)
    maximum = np.max(data,axis=0)
    return (minimum,maximum)

def scale_min(data,min_val,max_val):
    return add_ones((data-min_val)/(max_val-min_val))

def add_ones(data):
    return np.concatenate((np.ones((data.shape[0],1)),data),axis=1)