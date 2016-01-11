from numpy import matrix, identity
from angle import Angle
from math import cos, sin, sqrt
from scipy.stats import multivariate_normal

class Vector:
    def __init__(self, v):
        if not isinstance(v, matrix) or not v.shape[1] == 1:
            raise TypeError('Argument must be a vector')
        self.v = v
        self.dim = len(v)

    def exp(self, delta):
        return Vector(self.v+delta)

    @classmethod
    def calculateMean(cls, weights, vectors):
        v_mean = 0
        for i in range(len(vectors)):
            v_mean += weights[i]*vectors[i].v
        return Vector(v_mean)

    def log(self, other, symmetry=None):
        return other.v-self.v

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
            raise TypeError('Argument must be a vector or angle')

    def scoreEquals(self, other, mean, covar):
        if isinstance(other, Vector):
            return multivariate_normal.pdf(self.log(other).getT().tolist()[0],mean.getT().tolist()[0],covar)
        else:
            raise TypeError('Argument must be a vector')

    def __str__(self):
        return str(self.v)
