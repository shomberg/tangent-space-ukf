from math import sin, cos, pi
from numpy import matrix, random, identity
from kalman_util import append_matrices, add_noise

from quaternion import Quaternion
from angle import Angle
from angle3d import Angle3D
from vector import Vector
from unscented_kalman_filter_objects import *

import pdb

objects = [Vector(matrix([[1],[0]]))]

covars = [.1*matrix(identity(2))]

def update_function(command, objects, noise):
    #pdb.set_trace()
    return [Vector(objects[0].v+matrix([[(command[0]+noise[0,0])],[(command[0]+noise[0,0])]]))]

def measurement_predict_function(objects, noise):
    #pdb.set_trace()
    return [Vector(objects[0].v+noise[0:2])]

covar_process_noise = matrix([[.005**2]])

covar_measurement_noise = (.005**2)*matrix(identity(2))

k = UnscentedKalmanFilter(objects, covars, update_function, measurement_predict_function, [None], covar_process_noise, covar_measurement_noise,True)

realPos = [Vector(matrix([[1],[0]]))]
for i in xrange(50):
    realPos = update_function([pi/12], realPos, matrix(random.multivariate_normal([0],covar_process_noise)).getT())
    measurement = measurement_predict_function(realPos, matrix(random.multivariate_normal([0,0],covar_measurement_noise)).getT())
    #k.step([[measurement[0],measurement[2]],[measurement[1],measurement[3]]], [[0,2],[1,3]], [pi/12,pi/12])
    k.stateUpdate([pi/12])
    k.measurementUpdate([measurement[0]], [0])
    print "-------------"
    print "real position:"
    for i in realPos:
        print i
    print "perceived position:"
    for i in k.mean:
        print i
    print "delta:"
    for i in range(len(realPos)):
        print k.mean[i].relative(realPos[i])
    pdb.set_trace()
