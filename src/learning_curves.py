import numpy as np
import matplotlib.pyplot as plt
from scale import parametrs_standard,scale,min_max,scale_min
from solve_analytical import solve_analytical,solve_analytical_l2
from loss import least_sqares
from gradient import gradient_descent,coordinate_descent
from basis_functions import id,pow,gauss,poly,reduce,add_product,product_all


unscaled_training = np.loadtxt("data/training2.data")
unscaled_test=np.loadtxt("data/test2.data")
min_value,max_value = min_max(unscaled_training)
training = scale_min(unscaled_training,min_value,max_value)
test = scale_min(unscaled_test,min_value,max_value)
trainingX = training[:,:-1]
trainingY = unscaled_training[:,-1:]
testX = test[:,:-1]
testY=unscaled_test[:,-1:]

#func = id
#func = poly(id,5)
#func = poly(id,10)
#func = product_all(id)
func = poly(product_all(id),10)
#func = product_all(poly(id,5))
#func = id
based_training=func(trainingX)
based_test=func(testX)


fracs = [0.01,0.02,0.03,0.125,0.625,1]
res=np.empty((20,6))

for iter in range(0,20):
    indices = np.arange(based_training.shape[0])
    np.random.shuffle(indices)
    based_training=based_training[indices]
    trainingY=trainingY[indices]
    minires=np.empty((1,6))
    for idx,i in enumerate(fracs):
        first_n_rowsX = based_training[:int(based_training.shape[0] * i)]
        first_n_rowsY = trainingY[:int(trainingY.shape[0] * i)] 
        theta=solve_analytical(first_n_rowsX,first_n_rowsY)
        test_loss = least_sqares(based_test,testY,theta)
        minires[0,idx] = test_loss[0][0]
    res[iter] =minires
res=np.mean(res,axis=0)


print(res)


plt.figure(figsize=(8, 6))
plt.plot(fracs,res, marker='o', linestyle='-')
plt.title('Średnia strata')
plt.xlabel('Część zbioru treningowego')
plt.ylabel('Strata')
plt.grid(True)
plt.show()
