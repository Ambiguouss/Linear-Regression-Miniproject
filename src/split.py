import numpy as np


def split3(data):
    np.random.shuffle(data)
    n=data.shape[0]
    training = data[:n//2]
    validation = data[n//2:3*n//4]
    test = data[3*n//4:]
    return (training,validation,test)
def split2(data):
    np.random.shuffle(data)
    n=data.shape[0]
    training = data[:4*n//5]
    test = data[4*n//5:]
    return (training,test)