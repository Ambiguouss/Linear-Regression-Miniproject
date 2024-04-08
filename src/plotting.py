import numpy as np
import matplotlib.pyplot as plt
from scale import parametrs_standard,scale,min_max,scale_min
from solve_analytical import solve_analytical,solve_analytical_l2
from loss import least_sqares
from gradient import gradient_descent,coordinate_descent
from basis_functions import id,pow,gauss,poly,reduce,add_product,product_all


unscaled_training = np.loadtxt("data/training3.data")
unscaled_validation = np.loadtxt("data/validation3.data")
unscaled_test=np.loadtxt("data/test3.data")
min_value,max_value = min_max(unscaled_training)
training = scale_min(unscaled_training,min_value,max_value)
validation = scale_min(unscaled_validation,min_value,max_value)
test = scale_min(unscaled_test,min_value,max_value)
trainingX = training[:,:-1]
trainingY = unscaled_training[:,-1:]
testX = test[:,:-1]
testY=unscaled_test[:,-1:]
validX = validation[:,:-1]
validY = unscaled_validation[:,-1:]

#func = id
#func = poly(id,5)
#func = pow(id,10)
#func = product_all(id)
#func = poly(product_all(id),10)
#func = product_all(poly(id,5))
func = id
based_training=func(trainingX)
based_valid = func(validX)

iterations = np.arange(0, 300, 30)

theta_history=[]
loss_history=[]
for i in iterations:
    thetareg = coordinate_descent(based_training,trainingY,10000,100*i)
    theta_history.append(thetareg)
    valid_loss = least_sqares(based_valid,validY,thetareg)
    loss_history.append(valid_loss)

loss_history=np.array(loss_history)
print(np.argmin(loss_history))
print(np.min(loss_history))
print(theta_history[np.argmin(loss_history)])
theta_history = np.array(theta_history)
plt.figure(figsize=(10, 6))
for i in range(1,theta_history.shape[1]):
    plt.plot(iterations, theta_history[:, i], label=f"Theta_{i}")
plt.axvline(x=30*np.argmin(loss_history), color='red', linestyle='--', linewidth=2)
plt.xlabel("Lambda")
plt.ylabel("Wartość parametru")
plt.legend()
plt.grid(True)
plt.show()


training_loss=least_sqares(based_training,trainingY,thetareg)
valid_loss = least_sqares(based_valid,validY,thetareg)
print(training_loss,valid_loss)

