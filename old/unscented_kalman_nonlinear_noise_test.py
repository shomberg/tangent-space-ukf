execfile("unscented_kalman_filter_nonlinear_noise.py")
from numpy import random, identity, zeros, array
from math import pow

"""mean = matrix([[1],[2]])
covar = matrix([[2,0],[0,1]])
#update_function = lambda x: matrix([[5],[-3]]) + .5*(x-matrix([[5],[-3]]))
def update_function(x, v):
    ret = x + matrix([[2],[-1]]) + v
    return ret
def update_function(x,v):
    gamma = 1
    beta = 2
    n = 9.65
    ret = concatenate(((1-gamma)*x[0]+beta*(x[-1,0]/(1+x[-1,0]**n)),x[0:x.shape[0]-1,0])) + concatenate((v[0,0],zeros((v.shape[0]-1,1))))
    return matrix(ret)
covar_update_noise = matrix([[.1,0],[0,.1]])
def measurement_predict_function(x, n):
    ret = (x + n)*2
    return ret
covar_measurement_noise = matrix([[.4,0],[0,.5]])"""

mean = matrix([[0.9697],[0.9699],[0.9794],[1.0003],[1.0319],[1.0703],[1.1076],[1.1352],[1.1485],[1.1482],[1.1383],[1.1234],[1.1072],[1.0928],[1.0820],[1.0756],[1.0739],[1.0759]])
covar = identity(18)/50.
def update_function(x,v):
    gamma = 1
    beta = 1
    n = 9.1
    ret = concatenate((array((1-gamma)*x[0]+beta*(x[-1,0]/(1+pow(x[-1,0],n)))),x[0:x.shape[0]-1,0])) + concatenate((matrix([[v[0,0]]]),zeros((v.shape[0]-1,1))))
    return matrix(ret)
covar_update_noise = identity(18)/10000.
def measurement_predict_function(x, n):
    ret = matrix([[(x + n)[0,0]]])
    return ret
covar_measurement_noise = matrix([[.3]])

k = unscentedKalman(mean, covar, update_function, measurement_predict_function, covar_update_noise, covar_measurement_noise)
print k.mean
#print k.covar
state = mean.copy()
for i in range(100):
    state = update_function(state, matrix(random.multivariate_normal([0]*covar_update_noise.shape[0], covar_update_noise)).getT())
    measurement = measurement_predict_function(state, matrix(random.multivariate_normal([0]*covar_measurement_noise.shape[0], covar_measurement_noise)).getT())
    print "measurement"
    print measurement
    k.step(measurement, .001)
    print "state"
    print state
    print "mean"
    print k.mean
    #print k.covar
