from numpy import matrix, identity

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

    def relative(self, reference):
        if not isinstance(reference, Vector):
            raise TypeError('Argument must be a vector')
        return Vector(self.v-reference.v)

    def __str__(self):
        return str(self.v)
