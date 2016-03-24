from numpy import matrix, random, identity, linalg
from math import pi, tan, sin, cos, sqrt, atan
from time import clock

from quaternion import Quaternion
from angle import Angle
from angle3d import Angle3D
from vector import Vector
from unscented_kalman_filter_objects import *
from kalman_util import add_noise

from testPr2 import *
import pdb

objects = [Vector(matrix([[1.1],[-.4],[tZ]])),Angle.fromRadians(0),Vector(matrix([[1.1],[0],[tZ]])),Angle.fromRadians(0),Vector(matrix([[1.3],[0],[0]])),Angle.fromRadians(pi/2),Vector(matrix([[0.],[0.],[0.]])),Angle.fromRadians(0)]

covars = [matrix([[.1**2,0,0],[0,.1**2,0],[0,0,1e-10]]),(.3**2)*matrix(identity(1)),matrix([[.1**2,0,0],[0,.1**2,0],[0,0,1e-10]]),(.3**2)*matrix(identity(1)),matrix([[.07**2,0,0],[0,.03**2,0],[0,0,1e-10]]),(.2**2)*matrix(identity(1)),(.00001**2)*matrix(identity(3)),(.00001**2)*matrix(identity(1))]

covar_process_noise = append_matrices([(.00001**2)*matrix(identity(12)),(.005**2)*matrix(identity(2)),.000000001*matrix(identity(1)),((pi/36)**2)*matrix(identity(1))])

## Mess with this to increase observation variance
#typicalErrProbs.obsVar = 10*typicalErrProbs.obsVar

covar_measurement_noise = append_matrices([typicalErrProbs.obsVar[0:4,0:4],typicalErrProbs.obsVar[0:4,0:4],typicalErrProbs.obsVar[0:4,0:4]])

def state_update(command, objects, noise):
#    return objects
    return add_noise(objects[0:6],noise[0:12])+add_noise([objects[6]],noise[12:15]+matrix([[command*cos(objects[7].toRadians()+pi/2)],[command*sin(objects[7].toRadians()+pi/2)],[0]]))+add_noise([Angle.fromRadians(objects[7].toRadians()-pi/6*command)],noise[15])

def measurement_predict(objects, noise):
    return add_noise([objects[0].relative(objects[6],[0,1,2]).relative(objects[7], [0,1]),objects[1].relative(objects[7]),objects[2].relative(objects[6],[0,1,2]).relative(objects[7], [0,1]),objects[3].relative(objects[7]),objects[4].relative(objects[6],[0,1,2]).relative(objects[7],[0,1]),objects[5].relative(objects[7])],noise[0:12])

symmetries = [None, 4, None, 4, None, 2, None, None]

k = UnscentedKalmanFilter(objects, covars, state_update, measurement_predict, symmetries, covar_process_noise, covar_measurement_noise,True)

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
    indices = []
    indexDic = {'table':[4], 'soda':[0,2], 'sodat':[1,3]}
    dic = {'table':[], 'soda':[], 'sodat':[]}
    for o in obs:
        dic[o[0]].append(Vector(matrix([[o[2].x-pos[0,0]],[o[2].y-pos[1,0]],[o[2].z-pos[2,0]]])).relative(Angle.fromRadians(theta),[0,1]))
        if o[0] == 'table':
            measurement.append(Angle.fromRadians(o[2].theta).relative(Angle.fromRadians(theta)))
            indices.append(5)
        if o[0] == 'soda':
            dic['sodat'].append(Angle.fromRadians(o[2].theta).relative(Angle.fromRadians(theta)))
    for i in dic.keys():
        if dic[i]:
            measurement.append(dic[i])
            indices.append(indexDic[i])
    print "measurement:"
    for i in measurement:
        if isinstance(i, list):
            for j in i:
                print j
        else:
            print i

    start = clock()
    if k.version==1:
        k.step(measurement,indices,command,scoreThresh=[.05])
    else:
        k.stateUpdate(command)
        k.measurementUpdate(measurement,indices,scoreThresh=[.05])
    end = clock()
    print "time to execute step method:", (end-start)

    if k.version == 0:
        ret = k.getMarginalDistribution()[1]
    else:
        ret = k.mean
    print ret[0].v
    print ret[1].toRadians()/pi
    print ret[2].v
    print ret[3].toRadians()/pi
    print ret[4].v
    print ret[5].toRadians()/pi
    print ret[6].v
    print ret[7].toRadians()/pi
    sim.bs.pbs.fixObjBs['table1'].poseD.mu = util.Pose(ret[4].v[0],ret[4].v[1],ret[4].v[2],ret[5].toRadians())
    sim.bs.pbs.fixObjBs['table1'].poseD.muTuple = sim.bs.pbs.fixObjBs['table1'].poseD.mu.xyztTuple()
    sim.bs.pbs.fixObjBs['table1'].poseD.var = (k.covar[8,8],k.covar[9,9],k.covar[10,10],k.covar[11,11])
    sim.bs.pbs.moveObjBs['objA'].poseD.mu = util.Pose(ret[0].v[0],ret[0].v[1],ret[0].v[2],ret[1].toRadians())
    sim.bs.pbs.moveObjBs['objA'].poseD.muTuple = sim.bs.pbs.moveObjBs['objA'].poseD.mu.xyztTuple()
    sim.bs.pbs.moveObjBs['objA'].poseD.var = (k.covar[0,0],k.covar[1,1],k.covar[2,2],k.covar[3,3])
    sim.bs.pbs.moveObjBs['objB'].poseD.mu = util.Pose(ret[2].v[0],ret[2].v[1],ret[2].v[2],ret[3].toRadians())
    sim.bs.pbs.moveObjBs['objB'].poseD.muTuple = sim.bs.pbs.moveObjBs['objB'].poseD.mu.xyztTuple()
    sim.bs.pbs.moveObjBs['objB'].poseD.var = (k.covar[4,4],k.covar[5,5],k.covar[6,6],k.covar[7,7])
    sim.bs.pbs.conf = sim.bs.pbs.conf.set('pr2Base',[ret[6].v[0,0],ret[6].v[1,0],ret[7].toRadians()])
    sim.bs.pbs.shadowWorld = None
    sim.bs.pbs.draw(win='Belief')
    pdb.set_trace()


