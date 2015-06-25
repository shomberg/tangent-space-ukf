execfile("unscented_kalman_filter.py")
from numpy import random

mean = matrix([[1],[2]])
covar = matrix([[2,0],[0,1]])
#update_function = lambda x: matrix([[5],[-3]]) + .5*(x-matrix([[5],[-3]]))
def update_function(x):
    return x + matrix([[2],[-1]])
covar_update_noise = matrix([[.1,0],[0,.1]])
measurement_predict_function = lambda x: x
covar_measurement_noise = matrix([[.4,0],[0,.5]])

measurement = matrix([[2.3],[1.3]])

k = unscentedKalman(mean, covar, update_function, measurement_predict_function, covar_update_noise, covar_measurement_noise)
print k.mean
#print k.covar
state = matrix([[1],[2]])
for i in range(10):
    state = matrix(random.multivariate_normal(update_function(state).getT().tolist()[0], covar_update_noise)).getT()
    measurement = matrix(random.multivariate_normal(measurement_predict_function(state).getT().tolist()[0],covar_measurement_noise)).getT()
    k.step(measurement, .001)
    print "state"
    print state
    print "mean"
    print k.mean
    #print k.covar
