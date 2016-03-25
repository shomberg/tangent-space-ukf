from numpy import matrix, random, identity, linalg
from math import pi, tan, sin, cos, sqrt, atan
from time import clock

from quaternion import Quaternion
from angle import Angle
from angle3d import Angle3D
from vector import Vector
from unscented_kalman_filter_objects import *
from kalman_util import add_noise
from filter_system import *

from testPr2 import *
import pdb

objects = [Vector(matrix([[1.1],[-.4],[tZ]])),Angle.fromRadians(0),Vector(matrix([[1.1],[0],[tZ]])),Angle.fromRadians(0),Vector(matrix([[1.3],[0],[0]])),Angle.fromRadians(pi/2),Vector(matrix([[0.],[0.],[0.]])),Angle.fromRadians(0)]

covars = [matrix([[.1**2,0,0],[0,.1**2,0],[0,0,1e-10]]),(.3**2)*matrix(identity(1)),matrix([[.1**2,0,0],[0,.1**2,0],[0,0,1e-10]]),(.3**2)*matrix(identity(1)),matrix([[.07**2,0,0],[0,.03**2,0],[0,0,1e-10]]),(.2**2)*matrix(identity(1)),(.00001**2)*matrix(identity(3)),(.00001**2)*matrix(identity(1))]

covar_process_noise = append_matrices([(.00001**2)*matrix(identity(12)),(.005**2)*matrix(identity(2)),.000000001*matrix(identity(1)),((pi/36)**2)*matrix(identity(1))])

## Mess with this to increase observation variance
#typicalErrProbs.obsVar = 10*typicalErrProbs.obsVar

covar_measurement_noise = append_matrices([typicalErrProbs.obsVar[0:4,0:4],typicalErrProbs.obsVar[0:4,0:4],typicalErrProbs.obsVar[0:4,0:4]])

def state_update_none_vec(command, objs, noise, obj_indices, noise_indices):
    return add_noise([objs[obj_indices[0]]],noise[noise_indices[0]:noise_indices[0]+3])[0]

def state_update_none_ang(command, objs, noise, obj_indices, noise_indices):
    return add_noise([objs[obj_indices[0]]],noise[noise_indices[0]:noise_indices[0]+1])[0]

def state_update_move_vec(command, objs, noise, obj_indices, noise_indices):
    return add_noise([objs[obj_indices[0]]],noise[noise_indices[0]:noise_indices[0]+3]+matrix([[command*cos(objs[obj_indices[1]].toRadians()+pi/2)],[command*sin(objs[obj_indices[1]].toRadians()+pi/2)],[0]]))[0]

def state_update_move_ang(command, objs, noise, obj_indices, noise_indices):
    return add_noise([Angle.fromRadians(objs[obj_indices[0]].toRadians()-pi/6*command)],noise[noise_indices[0]])[0]

def measure_predict_vec(objs, noise, obj_indices, noise_indices):
    #pdb.set_trace()
    return add_noise([objs[obj_indices[0]].relative(objs[obj_indices[1]],[0,1,2]).relative(objs[obj_indices[2]], [0,1])], noise[noise_indices[0]:noise_indices[0]+3])[0]

def measure_predict_ang(objs, noise, obj_indices, noise_indices):
    return add_noise([objs[obj_indices[0]].relative(objs[obj_indices[1]])], noise[noise_indices[0]:noise_indices[0]+1])[0]

symmetries = [None, 4, None, 4, None, 2, None, None]

fs = FilterSystem()
fs.addObject('robot')
fs.addObject('ObjA')
fs.addObject('ObjB')
fs.addObject('table')

fs.addContinuousDist('robot', 'heading', objects[7], covars[7], state_update_move_ang, [], covar_process_noise[15:16,15:16])
fs.addContinuousDist('robot', 'position', objects[6], covars[6], state_update_move_vec, [('robot', 'heading')], covar_process_noise[12:15,12:15])
fs.addContinuousDist('ObjA', 'heading', objects[1], covars[1], state_update_none_ang, [], covar_process_noise[3:4,3:4])
fs.addContinuousDist('ObjA', 'position', objects[0], covars[0], state_update_none_vec, [], covar_process_noise[0:3,0:3])
fs.addContinuousDist('ObjB', 'heading', objects[3], covars[3], state_update_none_ang, [], covar_process_noise[7:8,7:8])
fs.addContinuousDist('ObjB', 'position', objects[2], covars[2], state_update_none_vec, [], covar_process_noise[4:7,4:7])
fs.addContinuousDist('table', 'heading', objects[5], covars[5], state_update_none_ang, [], covar_process_noise[11:12,11:12])
fs.addContinuousDist('table', 'position', objects[4], covars[4], state_update_none_vec, [], covar_process_noise[8:11,8:11])

fs.addObservation('apos', measure_predict_vec, symmetries[0], [('ObjA', 'position'),('robot', 'position'), ('robot', 'heading')], covar_measurement_noise[0:3,0:3])
fs.addObservation('ahead', measure_predict_ang, symmetries[1], [('ObjA', 'heading'),('robot', 'heading')], covar_measurement_noise[3:4,3:4])
fs.addObservation('bpos', measure_predict_vec, symmetries[2], [('ObjB', 'position'),('robot', 'position'), ('robot', 'heading')], covar_measurement_noise[4:7,4:7])
fs.addObservation('bhead', measure_predict_ang, symmetries[3], [('ObjB', 'heading'),('robot', 'heading')], covar_measurement_noise[7:8,7:8])
fs.addObservation('tpos', measure_predict_vec, symmetries[4], [('table', 'position'),('robot', 'position'), ('robot', 'heading')], covar_measurement_noise[8:11,8:11])
fs.addObservation('thead', measure_predict_ang, symmetries[5], [('table', 'heading'),('robot', 'heading')], covar_measurement_noise[11:12,11:12])

sim = testSim()
sim.bs.pbs.conf=sim.bs.pbs.conf.set("pr2Head", [0,.8])

pos = matrix([[0.0],[0.0],[0.0]])
theta = 0
command = -.25
for i in xrange(-5,27):
    if i < 0:
        command = 0
    elif i >= 15:
        command = .25
    else:
        command = -.25
    add = random.multivariate_normal(mean=[0,0,0,0],cov=covar_process_noise[12:16,12:16])
    pos[0] += command*cos(theta+pi/2)+add[0]
    pos[1] += command*sin(theta+pi/2)+add[1]
    theta += -pi/6*command+add[3]
    print "------------------"
    print "Actual Robot Position"
    print pos
    print theta/pi
    sim.bs.pbs.conf = sim.bs.pbs.conf.set('pr2Base',[pos[0],pos[1],theta])
    sim.bs.pbs.conf.draw('W', 'cyan')
    obs = sim.realWorld.doLook(sim.bs.pbs.conf)
    measurement = []
    obs_names = []
    obs_dic = {'table':['tpos'], 'soda':['apos', 'bpos'], 'sodat':['ahead', 'bhead']}
    dic = {'table':[], 'soda':[], 'sodat':[]}
    for o in obs:
        dic[o[0]].append(Vector(matrix([[o[2].x-pos[0,0]],[o[2].y-pos[1,0]],[o[2].z-pos[2,0]]])).relative(Angle.fromRadians(theta),[0,1]))
        if o[0] == 'table':
            measurement.append(Angle.fromRadians(o[2].theta).relative(Angle.fromRadians(theta)))
            obs_names.append('thead')
        if o[0] == 'soda':
            dic['sodat'].append(Angle.fromRadians(o[2].theta).relative(Angle.fromRadians(theta)))
    for i in dic.keys():
        if dic[i]:
            measurement.append(dic[i])
            obs_names.append(obs_dic[i])
    print "measurement:"
    for i in measurement:
        if isinstance(i, list):
            for j in i:
                print j
        else:
            print i

    start = clock()
    fs.stateUpdate(command)
    fs.measurementUpdate(measurement,obs_names)
    end = clock()
    print "time to execute step method:", (end-start)

    ret = fs.k.mean
    print ret[0].toRadians()/pi
    print ret[1].v
    print ret[2].toRadians()/pi
    print ret[3].v
    print ret[4].toRadians()/pi
    print ret[5].v
    print ret[6].toRadians()/pi
    print ret[7].v
    sim.bs.pbs.fixObjBs['table1'].poseD.mu = util.Pose(ret[7].v[0],ret[7].v[1],ret[7].v[2],ret[6].toRadians())
    sim.bs.pbs.fixObjBs['table1'].poseD.muTuple = sim.bs.pbs.fixObjBs['table1'].poseD.mu.xyztTuple()
    sim.bs.pbs.fixObjBs['table1'].poseD.var = (fs.k.covar[13,13],fs.k.covar[14,14],fs.k.covar[15,15],fs.k.covar[12,12])
    sim.bs.pbs.moveObjBs['objA'].poseD.mu = util.Pose(ret[3].v[0],ret[3].v[1],ret[3].v[2],ret[2].toRadians())
    sim.bs.pbs.moveObjBs['objA'].poseD.muTuple = sim.bs.pbs.moveObjBs['objA'].poseD.mu.xyztTuple()
    sim.bs.pbs.moveObjBs['objA'].poseD.var = (fs.k.covar[5,5],fs.k.covar[6,6],fs.k.covar[7,7],fs.k.covar[4,4])
    sim.bs.pbs.moveObjBs['objB'].poseD.mu = util.Pose(ret[5].v[0],ret[5].v[1],ret[5].v[2],ret[4].toRadians())
    sim.bs.pbs.moveObjBs['objB'].poseD.muTuple = sim.bs.pbs.moveObjBs['objB'].poseD.mu.xyztTuple()
    sim.bs.pbs.moveObjBs['objB'].poseD.var = (fs.k.covar[9,9],fs.k.covar[10,10],fs.k.covar[11,11],fs.k.covar[8,8])
    sim.bs.pbs.conf = sim.bs.pbs.conf.set('pr2Base',[ret[1].v[0,0],ret[1].v[1,0],ret[0].toRadians()])
    sim.bs.pbs.shadowWorld = None
    sim.bs.pbs.draw(win='Belief')
    pdb.set_trace()


