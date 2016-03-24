from math import sin, cos, pi
from numpy import matrix, random, identity
from kalman_util import append_matrices, add_noise

from quaternion import Quaternion
from angle import Angle
from angle3d import Angle3D
from vector import Vector
from unscented_kalman_filter_objects import *

import pdb

objects = [Vector(matrix([[1],[0]])),Angle.fromRadians(pi/2)]

covars = [.1*matrix(identity(2)),matrix([[(pi/1006)**2]])]

def update_function_vec(command, objs, noise, obj_indices, noise_indices):
    #pdb.set_trace()
    return Vector(objects[obj_indices[0]].v+matrix([[(command[0]+noise[noise_indices[0],0])*cos(objects[obj_indices[1]].toRadians())],[(command[0]+noise[noise_indices[0],0])*sin(objects[obj_indices[1]].toRadians())]]))

def update_function_angle(command, objs, noise, obj_indices, noise_indices):
    return Angle.fromRadians(pi/12+objects[obj_indices[0]].toRadians()+noise[noise_indices[0],0])

def measurement_predict_function_vec(objects, noise):
    #pdb.set_trace()
    return Vector(objects[0].v+noise[0:2])

def measurement_predict_function_angle(objects, noise):
    return Angle.fromRadians(objects[1].toRadians()+noise[2,0])

covar_process_noise = append_matrices((matrix([[.005**2]]),matrix([[(pi/36)**2]])))

covar_measurement_noise = append_matrices(((.005**2)*matrix(identity(2)),matrix([[(pi/36)**2]])))

k = UnscentedKalmanFilter([], [], [], [measurement_predict_function_vec, measurement_predict_function_angle], [], matrix([[]]).reshape((0,0)), covar_measurement_noise,True)

k.addObject(objects[1], covars[1], None, update_function_angle, 

k.addObject(objects[0], covars[0], None, update_function_vec, [0,1], covar

realPos = [Vector(matrix([[1],[0]])),Angle.fromRadians(pi/2)]
for i in xrange(50):
    realPos = update_function([pi/12], realPos, matrix(random.multivariate_normal([0,0],covar_process_noise)).getT())
    measurement = measurement_predict_function(realPos, matrix(random.multivariate_normal([0,0,0],covar_measurement_noise)).getT())
    #k.step([[measurement[0],measurement[2]],[measurement[1],measurement[3]]], [[0,2],[1,3]], [pi/12,pi/12])
    k.stateUpdate([pi/12])
    k.measurementUpdate([measurement[0],measurement[1]], [0,1])
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
