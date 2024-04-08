import numpy as np

def id(x):
    return x

def pow(func,deg):
    def new_func(x):
        x=func(x)
        return x**deg
    return new_func

def poly(func,deg):
    def new_func(x):
        x=func(x)
        res=x
        for i in range(2,deg+1):
            y=x**i
            res = np.concatenate((res, y), axis=1)
        return res
    return new_func

def gauss(func,par):
    def new_func(x):
        x=func(x)
        mean=np.mean(x,axis=0)
        return np.exp(-(x-mean)/par**2)
    return new_func

def reduce(func,column):
    def new_func(x):
        x=func(x)
        return np.delete(x,column,axis=1)
    return new_func

def add_product(func,columns):
    def new_func(x):
        x=func(x)
        y=np.prod(x[:, columns], axis=1)
        return np.column_stack((x,y))
    return new_func
def product_all(func):
    def new_func(x):
        x=func(x)
        a=x.shape[1]
        for i in range(0,a):
            for j in range(0,a):
                y=np.prod(x[:, [i,j]], axis=1)
                x=np.column_stack((x,y))
        return x
    return new_func

