from numpy import matrix, concatenate
from math import sin, cos, acos, sqrt


from quaternion import Quaternion

class Angle3D:
    def __init__(self, r):
        if not isinstance(r, (int, float, long, Quaternion)):
            raise TypeError('Argument must be a quaternion')
        if not abs(r.mag()-1) < .000001:
            raise ValueError('Argument must be unit')
        self.r = r

    @classmethod
    def exp(cls, v_t, r, basis):
        return Angle3D((r+Quaternion(*((basis*v_t).getT().tolist()[0]))).normalize())

    @classmethod
    def generateBasis(cls, r):
        q1 = r*Quaternion(1,0,0,0)
        q2 = r*Quaternion(0,1,0,0)
        q3 = r*Quaternion(0,0,1,0)
        return concatenate((q1.asVector(),q2.asVector(),q3.asVector()),axis=1)

    def getCoordinates(self):
        return matrix([[0],[0],[0]])

    def getOrigin(self):
        return self.r

    @classmethod
    def normalizeOrigin(cls, r):
        return r.normalize()

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

    def log(self, rSpace, basis):
        k = rSpace.dot(rSpace)/(self.r.dot(rSpace))
        return basis.getI()*((k*self.r-rSpace).asVector())

    def __str__(self):
        return "Angle3D("+str(self.r)+")"
