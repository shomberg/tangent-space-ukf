from numpy import matrix, conjugate
from math import atan2, sin, cos, pi, atan

class Angle:
    def __init__(self, r):
        if not isinstance(r, (int, float, long, complex)):
            raise TypeError('Argument must be a complex number')
        if not abs(abs(r)-1) < .000001:
            raise ValueError('Argument must be unit')
        self.r = r
        self.dim = 1

    @classmethod
    def exp(cls, v_t, r):
        theta = atan(v_t)
        c = complex(cos(theta),sin(theta))*r
        return Angle(c)

    @classmethod
    def fromRadians(cls, theta):
        return Angle(complex(cos(theta),sin(theta)))
    
    @classmethod
    def generateBasis(cls, r):
        c = complex(0,1)*r
        return matrix([[c.real],[c.imag]])

    def getOrigin(self):
        return self.r

    @classmethod
    def normalizeOrigin(cls, r):
        return r/abs(r)

    def toRadians(self):
        return atan2(self.r.imag, self.r.real)

    def log(self, rSpace, symmetry=None):
        rUse = self.r/rSpace
        if symmetry:
            otherT = Angle(rSpace).toRadians()
            plus = (self.toRadians()-otherT)%(2*pi/symmetry)
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
