execfile("extended_kalman_filter.py")
from numpy import random

mean = matrix([[1],[2]])
covar = matrix([[2,0],[0,1]])
state_update = matrix(identity(2))
control_update = matrix([[0,0],[0,0]])
covar_update_noise = matrix([[0,0],[0,0]])
measurement_predict = matrix([[1,0],[0,1]])
covar_measurement_noise = matrix([[.4,0],[0,.5]])

control = matrix([[1],[-1]])
measurement = matrix([[2.3],[1.3]])

k = discreteKalman(mean, covar, state_update, control_update, covar_update_noise, measurement_predict, covar_measurement_noise)
print k.mean
#print k.covar
for i in range(1000):
    measurement = matrix([[random.normal(2.3,.4)],[random.normal(1.3,.5)]])
    k.step(control,measurement)
    print k.mean
    #print k.covar
