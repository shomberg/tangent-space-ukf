from numpy import matrix, conjugate
from math import atan2, sin, cos, pi, atan
from scipy.optimize import minimize
from kalman_util import magnitude
from scipy.stats import multivariate_normal
import pdb

class Angle:
    def __init__(self, r):
        if not isinstance(r, (int, float, long, complex)):
            raise TypeError('Argument must be a complex number')
        if not abs(abs(r)-1) < .000001:
            raise ValueError('Argument must be unit')
        self.r = r
        self.dim = 1

    def exp(self, delta):
        return Angle.fromRadians(self.toRadians()+delta)

    @classmethod
    def fromRadians(cls, theta):
        return Angle(complex(cos(theta),sin(theta)))

    @classmethod
    def calculateMean(cls, weights, angles):
        # constrained optimization problem of sum_i d(x_i,mu)^2 over mu in unit complex space
        def sumDist(r):  # function to minimize
            ret = 0
            for i in range(len(angles)):
                add = weights[i]*pow(magnitude(Angle(complex(*r)/magnitude(r)).log(angles[i])),2)
                ret += add
            return ret
        
        r_mean = minimize(sumDist, [angles[0].r.real,angles[0].r.imag],constraints={'type': 'eq', 'fun': lambda r: magnitude(r)-1}).x
        return Angle(complex(*r_mean)/magnitude(r_mean))

    def toRadians(self):
        return atan2(self.r.imag, self.r.real)

    def log(self, other, symmetry=None):
        if symmetry and symmetry[0]:
            base = other.toRadians()-self.toRadians()
            bestRot = None
            bestProb = -1
            for i in range(0,symmetry[0]):
                rotation = matrix([[base + (i*2*pi/symmetry[0])]])
                probability = multivariate_normal.pdf(rotation,mean=symmetry[1],cov=symmetry[2])
                if probability > bestProb:
                    bestRot = rotation
                    bestProb = probability
            return bestRot
        else:
            return matrix([[other.toRadians()-self.toRadians()]])
        
    def relative(self, reference):
        if not isinstance(reference, Angle):
            raise TypeError('Argument must be an angle')
        return Angle(self.r/reference.r)

    def scoreEquals(self, other):
        if isinstance(other, Angle):
            r = self.r/other.r
            return 1-r.real
        else:
            raise TypeError('Argument must be an angle')

    def __str__(self):
        return "Angle("+str(self.r)+")"
