import numpy as np
from scale import *
from solve_analytical import *
from loss import *
from gradient import *
from basis_functions import *


unscaled_training = np.loadtxt("data/training2.data")
#unscaled_validation = np.loadtxt("data/validation3.data")
unscaled_test=np.loadtxt("data/test2.data")
min_value,max_value = min_max(unscaled_training)
training = scale_min(unscaled_training,min_value,max_value)
#validation = scale_min(unscaled_validation,min_value,max_value)
test = scale_min(unscaled_test,min_value,max_value)
trainingX = training[:,:-1]
trainingY = unscaled_training[:,-1:]
testX = test[:,:-1]
testY=unscaled_test[:,-1:]
#validX = validation[:,:-1]
#validY = unscaled_validation[:,-1:]

#func = id
#func = poly(id,5)
#func = poly(id,10)
#func = product_all(id)
#func = poly(product_all(id),10)
func = product_all(poly(id,5))
#func = gauss(id,1)
based_training=func(trainingX)
#based_valid = func(validX)
based_test=func(testX)

theta=solve_analytical(based_training,trainingY)
#theta = gradient_descent(based_training,trainingY,step=0.00001,iterations=10000)
training_loss=least_sqares(based_training,trainingY,theta)
test_loss = least_sqares(based_test,testY,theta)
print(training_loss,test_loss)

