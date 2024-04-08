import numpy as np



def gradient_descent(x,y,step,iterations=1000,l1=0,l2=0):
    n=x.shape[1]
    theta = np.zeros((n, 1))
    for i in range(0,iterations):
        a=(x@theta-y)
        gradient = (np.transpose(x)@a)

        l1reg=l1*np.sign(theta)
        l1reg[0]=0
        l2reg = l2*theta
        l2reg[0]=0
        
        gradient+=l1reg+l2reg
        theta -= step*gradient
    return theta


def coordinate_descent(x,y,iterations=1000,l1=0):
    n=x.shape[1]
    theta = np.zeros((n, 1))
    theta[0]=np.mean(y)
    for i in range(0,iterations):
        k = np.random.randint(1,n)
        xi=x[:,k]
        tb=np.delete(theta,k,axis=0)
        xb=np.delete(x,k,axis=1)
        cj=2*xi@(y-xb@tb)
        a=2*np.sum(xi**2)
        if cj<-l1:
            theta[k]=(cj+l1)/a
        elif cj>l1:
            theta[k]=(cj-l1)/a
        else:
            theta[k]=0
    return theta