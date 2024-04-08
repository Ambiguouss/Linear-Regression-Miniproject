import numpy as np
from split import split2, split3

data = np.loadtxt('data/dane.data')
training2,test2=split2(data)
np.savetxt("data/training2.data",training2)
np.savetxt("data/test2.data",test2)
training3,validation3,test3=split3(data)
np.savetxt("data/training3.data",training3)
np.savetxt("data/validation3.data",validation3)
np.savetxt("data/test3.data",test3)
