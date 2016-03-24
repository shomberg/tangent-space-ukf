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

objects = [Angle.fromRadians(pi/2)]

covars = [matrix([[(pi/1006)**2]])]

def update_function(command, objs, noise, obj_indices, noise_indices):
    #pdb.set_trace()
    return Angle.fromRadians(pi/12+objs[obj_indices[0]].toRadians()+noise[noise_indices[0],0])

def measurement_predict_function(objects, noise, obj_indices, noise_indices):
    #pdb.set_trace()
    return Angle.fromRadians(objects[obj_indices[0]].toRadians()+noise[noise_indices[0],0])

covar_process_noise = matrix([[(pi/36)**2]])

covar_measurement_noise = matrix([[(pi/36)**2]])

fs = FilterSystem()
fs.addObject('ObjA')
fs.addContinuousDist('ObjA', 'heading', objects[0], covars[0], update_function, [], covar_process_noise)
fs.addObservation('ahead', measurement_predict_function, None, [('ObjA', 'heading')], covar_measurement_noise)

realPos = Angle.fromRadians(pi/2)
for i in xrange(50):
    realPos = update_function([pi/12], [realPos], matrix(random.multivariate_normal([0],covar_process_noise)).getT(), [0], [0])
    measurement = measurement_predict_function([realPos], matrix(random.multivariate_normal([0],covar_measurement_noise)).getT(), [0], [0])
    fs.stateUpdate([pi/12])
    fs.measurementUpdate([measurement], ['ahead'])
    print "-------------"
    print "real position:"
    print realPos
    print "perceived position:"
    for i in fs.k.mean:
        print i
    print "delta:"
    print fs.k.mean[0].relative(realPos).toRadians()
    pdb.set_trace()
