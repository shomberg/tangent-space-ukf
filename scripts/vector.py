from numpy import matrix, identity
from angle import Angle
from math import cos, sin, sqrt

class Vector:
    def __init__(self, v):
        if not isinstance(v, matrix) or not v.shape[1] == 1:
            raise TypeError('Argument must be a vector')
        self.v = v
        self.dim = len(v)

    @classmethod
    def exp(cls, v, o):
        return Vector(v+o)

    @classmethod
    def generateBasis(cls, o):
        return matrix(identity(o.shape[0]))

    def getCoordinates(self):
        return matrix([[0]*self.v.shape[0]]).getT()

    def getOrigin(self):
        return self.v

    @classmethod
    def normalizeOrigin(cls, v):
        return v

    def log(self, o, symmetry=None):
        return self.v-o

    def relative(self, reference, indices):
        if isinstance(reference, Vector):
            ret = self.v.copy()
            for i in xrange(len(indices)):
                ret[indices[i]] -= reference.v[i]
            return Vector(ret)
        elif isinstance(reference, Angle):
            ret = self.v.copy()
            theta = reference.toRadians()
            ret[indices[1]] = self.v[indices[1]]*cos(theta)-self.v[indices[0]]*sin(theta)
            ret[indices[0]] = self.v[indices[0]]*cos(theta)+self.v[indices[1]]*sin(theta)
            return Vector(ret)
        else:
            raise TypeError('Argument must be a vector')

    def scoreEquals(self, other):
        if isinstance(other, Vector):
            s = 0
            v = self.v-other.v
            for i in v:
                s += i*i
            return sqrt(s)
        else:
            raise TypeError('Argument must be a vector')

    def __str__(self):
        return str(self.v)
