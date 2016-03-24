from math import sin, cos, pi, sqrt
from numpy import matrix, random, identity
from kalman_util import append_matrices, add_noise, magnitude

from quaternion import Quaternion
from angle import Angle
from angle3d import Angle3D
from vector import Vector
from unscented_kalman_filter_objects import *

import pdb

objects = [Angle3D(Quaternion(.5,0,0,.5).normalize())]
covars = [pi/6*matrix(identity(3))]
def update_function(command, objects, noise):
    return [Angle3D.fromRotationAxis(pi/4+objects[0].getRotation()+noise,objects[0].getAxis())]

def measurement_predict_function(objects, noise):
    pdb.set_trace()
    if magnitude(noise[1:]) == 0:
        axis = noise[1:]
    else:
        axis = noise[1:]/magnitude(noise[1:])
    theta = noise[0,0]
    return [Angle3D(Quaternion(sin(theta/2)*axis[0,0],sin(theta/2)*axis[1,0],sin(theta/2)*axis[2,0],cos(theta/2))*objects[0].r)]

covar_process_noise = matrix([[.001]])
covar_measurement_noise = .005*matrix(identity(4))

k = unscentedKalmanFilter(objects, covars, update_function, measurement_predict_function, [None], covar_process_noise, covar_measurement_noise)

realPos = [Angle3D.fromRotationAxis(pi/2,matrix([[1/sqrt(3)],[1/sqrt(3)],[1/sqrt(3)]]))]
for i in xrange(100):
    realPos = update_function(0,realPos,random.normal(0,covar_process_noise[0,0]))
    measurement = measurement_predict_function(realPos,matrix(random.multivariate_normal([0,0,0,0],covar_measurement_noise)).getT())
    print measurement[0].r
    k.step(measurement, [0])
    ret = k.mean
    print "filter, real"
    print ret[0].getRotation(), realPos[0].getRotation()
    print ret[0].getAxis()
    print realPos[0].getAxis()
    pdb.set_trace()
