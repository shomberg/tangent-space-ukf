from numpy import matrix, conjugate
from math import atan2, sin, cos, pi, atan
from scipy.optimize import minimize

class Angle:
    def __init__(self, r):
        if not isinstance(r, (int, float, long, complex)):
            raise TypeError('Argument must be a complex number')
        if not abs(abs(r)-1) < .000001:
            raise ValueError('Argument must be unit')
        self.r = r
        self.dim = 1

    def exp(self, delta):
        theta = atan(delta)
        c = complex(cos(theta),sin(theta))*self.r
        return Angle(c)

    @classmethod
    def fromRadians(cls, theta):
        return Angle(complex(cos(theta),sin(theta)))

    @classmethod
    def calculateMean(cls, weights, angles):
        r_mean = 0
        for i in range(len(angles)):
            r_mean += weights[i]*angles[i].r
        return Angle(r_mean/abs(r_mean))

    def toRadians(self):
        return atan2(self.r.imag, self.r.real)

    def log(self, other, symmetry=None):
        rUse = other.r/self.r
        if symmetry:
            otherT = other.toRadians()
            plus = (otherT-self.toRadians())%(2*pi/symmetry)
            minus = plus-(2*pi/symmetry)
            if plus + minus > 0:
                rUse = Angle.fromRadians(minus).r
            else:
                rUse = Angle.fromRadians(plus).r
        k = 1/rUse.real
        coord = k*rUse
        return matrix([[coord.imag]])

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
