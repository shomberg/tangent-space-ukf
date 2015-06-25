from numpy import matrix
from math import sqrt

class Quaternion:
    def __init__(self, i, j, k, r):
        for arg in (r,i,j,k):
            if not isinstance(arg, (int, float, long)):
                raise TypeError('All arguments must be real numbers')
        self.real = float(r)
        self.i = float(i)
        self.j = float(j)
        self.k = float(k)

    def asVector(self):
        return matrix([[self.i],[self.j],[self.k],[self.real]])

    def dot(self, q):
        return self.real*q.real + self.i*q.i + self.j*q.j + self.k*q.k

    def normalize(self):
        return self/(self.mag())

    def conj(self):
        return Quaternion(-self.i, -self.j, -self.k, self.real)
    
    def mag(self):
        return sqrt((self*self.conj()).real)

    def __add__(self, q):
        return Quaternion(self.i+q.i, self.j+q.j, self.k+q.k, self.real+q.real)

    def __radd__(self, r):
        if isinstance(r, (int, float, long)):
            return Quaternion(self.i, self.j, self.k, self.real+r)
        else:
            raise TypeError('incompatible types')
    
    def __sub__(self, q):
        return self + (-q)

    def __mul__(self, q):
        if isinstance(q, Quaternion):
            return Quaternion(
                self.real*q.i + self.i*q.real - self.j*q.k + self.k*q.j,
                self.real*q.j + self.i*q.k + self.j*q.real - self.k*q.i,
                self.real*q.k - self.i*q.j + self.j*q.i + self.k*q.real,
                self.real*q.real - self.i*q.i - self.j*q.j - self.k*q.k)
        elif isinstance(q, (int, float, long)):
            return Quaternion(self.i*q, self.j*q, self.k*q, self.real*q)
        else:
            raise TypeError('incompatible types')

    def __rmul__(self, r):
        if isinstance(r, (int, float, long)):
            return Quaternion(self.i*r, self.j*r, self.k*r, self.real*r)
        else:
            raise TypeError('incompatible types')
    
    def __div__(self, r):
        if isinstance(r, (int, float, long)):
            return Quaternion(self.i/r, self.j/r, self.k/r, self.real/r)
        else:
            raise TypeError('incompatible types')

    def __neg__(self):
        return Quaternion(-self.i, -self.j, -self.k, -self.real)

    def __str__(self):
        return "Quaternion(" + str(self.i) + "," + str(self.j) + "," + str(self.k) + "," + str(self.real) + ")"


class DualQuaternion:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def conj(self):
        return DualQuaternion(self.a.conj(), self.b.conj())

    def __add__(self, dq):
        if isinstance(dq, DualQuaternion):
            return DualQuaternion(self.a+dq.a, self.b+dq.b)
        else:
            raise TypeError('incompatible types')

    def __mul__(self, dq):
        if isinstance(dq, DualQuaternion):
            return DualQuaternion(self.a*dq.a, self.a*dq.b+self.b*dq.a)
        else:
            raise TypeError('incompatible types')

    def __str__(self):
        return str(self.a) + "+e*" + str(self.b)
