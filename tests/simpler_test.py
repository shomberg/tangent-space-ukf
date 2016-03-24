from math import sin, cos, pi
from numpy import matrix, random, identity
from kalman_util import append_matrices, add_noise

from quaternion import Quaternion
from angle import Angle
from angle3d import Angle3D
from vector import Vector
from unscented_kalman_filter_objects import *
from filter_system import *

import pdb

objects = [Vector(matrix([[1],[0]])),Angle.fromRadians(pi/2)]

covars = [.1*matrix(identity(2)),matrix([[(pi/1006)**2]])]

def update_function_vec(command, objs, noise, obj_indices, noise_indices):
    #pdb.set_trace()
    return Vector(objs[obj_indices[0]].v+matrix([[(command[0]+noise[noise_indices[0],0])*cos(objs[obj_indices[1]].toRadians())],[(command[0]+noise[noise_indices[0],0])*sin(objs[obj_indices[1]].toRadians())]]))

def update_function_angle(command, objs, noise, obj_indices, noise_indices):
    return Angle.fromRadians(pi/12+objs[obj_indices[0]].toRadians()+noise[noise_indices[0],0])

def measurement_predict_function_vec(objs, noise, obj_indices, noise_indices):
    #pdb.set_trace()
    return Vector(objs[obj_indices[0]].v+noise[noise_indices[0]:noise_indices[0]+2])

def measurement_predict_function_angle(objs, noise, obj_indices, noise_indices):
    return Angle.fromRadians(objs[obj_indices[0]].toRadians()+noise[noise_indices[0],0])

covar_process_noise = append_matrices((matrix([[.005**2]]),matrix([[(pi/36)**2]])))

covar_measurement_noise = append_matrices(((.005**2)*matrix(identity(2)),matrix([[(pi/36)**2]])))

fs = FilterSystem()
fs.addObject('ObjA')
fs.addContinuousDist('ObjA', 'heading', objects[1], covars[1], update_function_angle, [], covar_process_noise[1:2,1:2])
fs.addContinuousDist('ObjA', 'position', objects[0], covars[0], update_function_vec, [('ObjA', 'heading')], covar_process_noise[0:1,0:1])
fs.addObservation('apos', measurement_predict_function_vec, None, [('ObjA', 'position')], covar_measurement_noise[0:2,0:2])
fs.addObservation('ahead', measurement_predict_function_angle, None, [('ObjA', 'heading')], covar_measurement_noise[2:3,2:3])


realPos = [Vector(matrix([[1],[0]])),Angle.fromRadians(pi/2)]
for i in xrange(50):
    realPos[0] = update_function_vec([pi/12], realPos, matrix(random.multivariate_normal([0,0],covar_process_noise)).getT(), [0,1], [0])
    realPos[1] = update_function_angle([pi/12], realPos, matrix(random.multivariate_normal([0,0],covar_process_noise)).getT(), [1], [1])
    m_noise = matrix(random.multivariate_normal([0,0,0],covar_measurement_noise)).getT()
    measurement_vec = measurement_predict_function_vec(realPos, m_noise, [0], [0])
    measurement_angle = measurement_predict_function_angle(realPos, m_noise, [1], [2])
    measurement = [measurement_vec, measurement_angle]
    fs.stateUpdate([pi/12])
    fs.measurementUpdate([measurement[0],measurement[1]], ['apos','ahead'])
    print "-------------"
    print "real position:"
    for i in realPos:
        print i
    print "perceived position:"
    for i in fs.k.mean:
        print i
    print "delta:"
    for i in range(len(realPos)):
        print fs.k.mean[1-i].relative(realPos[i])
    pdb.set_trace()
