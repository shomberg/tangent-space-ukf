from numpy import matrix, concatenate
from math import sin, cos, acos, sqrt
from scipy.optimize import minimize


from quaternion import Quaternion

class Angle3D:
    def __init__(self, r):
        if not isinstance(r, (int, float, long, Quaternion)):
            raise TypeError('Argument must be a quaternion')
        if not abs(r.mag()-1) < .000001:
            raise ValueError('Argument must be unit')
        self.r = r
        self.dim = 3

    def exp(self, delta):
        return Angle3D(Quaternion(self.r.i+delta[0],self.r.j+delta[1],self.r.k+delta[2],self.r.real).normalize())

    def toVector(self):
        return self.getRotation()*self.getAxis()

    @classmethod
    def calculateMean(cls, weights, angles):
        r_mean = 0
        for i in range(len(angles)):
            r_mean += weights[i]*angles[i].r
        return Angle3D(r_mean.normalize())

    def getRotation(self):
        return 2*acos(self.r.real)

    def getAxis(self):
        if self.r.i==0 and self.r.j==0 and self.r.k==0:
            return matrix([[0],[0],[0]])
        return self.r.asVector()[0:3,0]/sqrt(self.r.i**2+self.r.j**2+self.r.k**2)

    @classmethod
    def fromRotationAxis(cls, theta, axis):
        if not abs(sqrt(axis[0,0]**2+axis[1,0]**2+axis[2,0]**2)-1)<.000001:
            raise ValueError('Axis must be nonzero')
        return Angle3D(Quaternion(*(sin(theta/2)*axis).getT().tolist()[0]+[cos(theta/2)]))

    def log(self, other, symmetry=None):
        rUse = self.r
        k = 1/rUse.real
        return (k*rUse).asVector()[0:3]

    def relative(self, reference):
        if not isinstance(reference, Angle3D):
            raise TypeError('Argument must be a 3d angle')
        return Angle3D(self.r*reference.r.conj())

    def scoreEquals(self, other):
        if isinstance(other, Angle3D):
            r = self.r*other.r.conj()
            return 1-r.real
        else:
            raise TypeError('Argument must be a 3d angle')

    def __str__(self):
        return "Angle3D("+str(self.r)+")"
