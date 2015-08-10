from numpy import matrix, identity
from angle import Angle
from math import cos, sin

class Vector:
    def __init__(self, v):
        if not isinstance(v, matrix) or not v.shape[1] == 1:
            raise TypeError('Argument must be a vector')
        self.v = v

    @classmethod
    def exp(cls, v, o, basis):
        return Vector(basis*v+o)

    @classmethod
    def generateBasis(cls, o):
        return matrix(identity(o.shape[0]))

    def getCoordinates(self):
        return self.v

    def getOrigin(self):
        return matrix([[0]*self.v.shape[0]]).getT()

    @classmethod
    def normalizeOrigin(cls, o):
        return o

    def log(self, o, basis, symmetry=None):
        return basis.getI()*(self.v-o)

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

    def __str__(self):
        return str(self.v)
