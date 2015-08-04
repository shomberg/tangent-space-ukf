from numpy import matrix, conjugate
from math import atan2, sin, cos, pi

class Angle:
    def __init__(self, r):
        if not isinstance(r, (int, float, long, complex)):
            raise TypeError('Argument must be a complex number')
        if not abs(abs(r)-1) < .000001:
            raise ValueError('Argument must be unit')
        self.r = r

    @classmethod
    def exp(cls, v_t, r, basis):
        c = r+complex(*(basis*v_t).getT().tolist()[0])
        return Angle(c/abs(c))

    @classmethod
    def fromRadians(cls, theta):
        return Angle(complex(cos(theta),sin(theta)))
    
    @classmethod
    def generateBasis(cls, r):
        c = complex(0,1)*r
        return matrix([[c.real],[c.imag]])

    def getCoordinates(self):
        return matrix([[0]])

    def getOrigin(self):
        return self.r

    @classmethod
    def normalizeOrigin(cls, r):
        return r/abs(r)

    def toRadians(self):
        return atan2(self.r.imag, self.r.real)

    def log(self, rSpace, basis, symmetry=None):
        rUse = self.r
        if symmetry:
            otherT = Angle(rSpace).toRadians()
            plus = (self.toRadians()-otherT)%(2*pi/symmetry)
            minus = plus-(2*pi/symmetry)
            if plus + minus > 0:
                rUse = Angle.fromRadians(otherT+minus).r
            else:
                rUse = Angle.fromRadians(otherT+plus).r
        k = (rSpace*conjugate(rSpace)).real/(rUse*conjugate(rSpace)).real
        coord = k*rUse-rSpace
        return basis.getI()*(matrix([[coord.real],[coord.imag]]))

    def relative(self, reference):
        if not isinstance(reference, Angle):
            raise TypeError('Argument must be an angle')
        return Angle(self.r/reference.r)

    def __str__(self):
        return "Angle("+str(self.r)+")"
