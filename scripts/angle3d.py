from numpy import matrix, concatenate
from math import sin, cos, acos, sqrt
from scipy.optimize import minimize
from kalman_util import magnitude
import pdb

from quaternion import Quaternion

class Angle3D:
    def __init__(self, r):
        if not isinstance(r, (Quaternion)):
            raise TypeError('Argument must be a quaternion')
        if not abs(r.mag()-1) < .000001:
            raise ValueError('Argument must be unit')
        self.r = r.normalize()
        self.dim = 3

    def exp(self, delta):
        if(magnitude(delta)==0):
            other = Angle3D(Quaternion(0,0,0,1))
        else:
            other = Angle3D.fromRotationAxis(magnitude(delta),delta/magnitude(delta))
        return Angle3D(other.r*self.r)

    def toVector(self):
        return self.getRotation()*self.getAxis()

    @classmethod
    def calculateMean(cls, weights, angles):
        # constrained optimization problem of sum_i d(x_i,mu)^2 over mu in unit quaternion space
        def sumDist(r):  # function to minimize
            ret = 0
            for i in range(len(angles)):
                add = weights[i]*pow(magnitude(Angle3D(Quaternion(*r)/magnitude(r)).log(angles[i])),2)
                ret += add
            return ret

        r_mean = minimize(sumDist, [angles[0].r.asVector()],constraints={'type': 'eq', 'fun': lambda r: magnitude(r)-1}).x
        return Angle3D(Quaternion(*r_mean).normalize())

    def getRotation(self):
        return 2*acos(self.r.real)

    def getAxis(self):
        if self.r.i==0 and self.r.j==0 and self.r.k==0:
            return matrix([[0],[0],[0]])
        return self.r.asVector()[0:3,0]/sqrt(self.r.i**2+self.r.j**2+self.r.k**2)

    @classmethod
    def fromRotationAxis(cls, theta, axis):
        if not abs(sqrt(axis[0,0]**2+axis[1,0]**2+axis[2,0]**2)-1)<.000001:
            print theta, axis
            raise ValueError('Axis must be unit')
        return Angle3D(Quaternion(*(sin(theta/2)*axis).getT().tolist()[0]+[cos(theta/2)]))

    def log(self, other, symmetry=None):
        if symmetry and symmetry[0]:
            given = other.relative(self)
            base = given.getRotation()
            axis = given.getAxis()
            bestRot = None
            bestProb = -1
            for i in range(0,symmetry[0]):
                rotation = Angle3D.fromRotationAxis(base+i*2*pi/symmetry[0],axis).toVector()
                probability = multivariate_normal.pdf(rotation,mean=symmetry[1],cov=symmetry[2])
                if probability > bestProb:
                    bestRot = rotation
                    bestProb = probability
            return bestRot
        else:
            return other.relative(self).toVector()

    def relative(self, reference):
        if not isinstance(reference, Angle3D):
            raise TypeError('Argument must be a 3d angle')
        return Angle3D(self.r*reference.r.conj())

    def scoreEquals(self, other, mean, covar):
        if isinstance(other, Angle3D):
            return multivariate_normal.pdf(self.log(other).getT().tolist()[0],mean.getT().tolist()[0],covar)
        else:
            raise TypeError('Argument must be a 3d angle')

    def __str__(self):
        return "Angle3D("+str(self.r)+")"
