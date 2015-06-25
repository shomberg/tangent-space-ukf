from numpy import matrix, concatenate, conjugate
from scipy import integrate, stats
from math import sqrt, pi, sin, cos
from time import clock

execfile('Quaternions.py')

class PG:
    def __init__(self, meanT, covarT, rotPart):
        self.meanT = meanT
        self.covarT = covarT
        self.rpg = rotPart

    def pi(self, r, t):
        return 1



class RPG:
    def __init__(self, r0, basis, mean, covar):
        self.r0 = r0
        self.basis = basis
        self.mean = mean
        self.covar = covar
        self.C = 1
        self.C = integrate.tplquad(lambda psi, phi, theta: sin(psi)**2*sin(phi)*self.pdfSphere(Quaternion(cos(psi), sin(psi)*cos(phi), sin(psi)*sin(phi)*cos(theta), sin(psi)*sin(phi)*sin(theta))), 0, 2*pi, lambda theta: 0, lambda theta: pi, lambda phi, theta: 0, lambda phi,theta: pi)[0]
    
    def pdfTangent(self, t):
        return stats.multivariate_normal.pdf(t.getT().tolist()[0], mean=self.mean.getT().tolist()[0], cov=self.covar)

    def pdfSphere(self, q):
        if q.dot(self.r0) == 0:
            return 0
        return self.pdfTangent(self.piInv(q))/self.C

    def pi(self, t):
        return Quaternion(*((basis*t).getT().tolist()[0])).normalize()

    def piInv(self, r):
        k = self.r0.dot(self.r0)/(r.dot(self.r0))
        return basis.getI()*((k*r-self.r0).asVector())


class Pose:
    def __init__(self, dq):
        if isinstance(dq, DualQuaternion):
            self.dq = dq
        else:
            raise TypeError('Pass a DualQuaternion or use fromComponents')

    @classmethod
    def fromComponents(cls, qr, qt):
        if abs(qr.mag()-1) > .0000000001:
            raise ValueError('Rotation quaternion must have magnitude 1')
        if not qt.real == 0:
            raise ValueError('Translation quaternion must be pure imaginary')
        return cls(DualQuaternion(qr, qt))

    @classmethod
    def fromArgs(cls, args, r):
        return Pose.fromComponents(r,Quaternion(0,args[3,0],args[4,0],args[5,0]))
    
    def rotation(self):
        return self.dq.a

    def translation(self):
        return self.dq.b

    def asVector(self):
        return matrix(concatenate((self.rotation().asVector(),self.translation().asVector())))

    def __str__(self):
        return "Rot:" + str(self.rotation()) + "Trans:" + str(self.translation())

class Transform:
    def __init__(self, dq):
        if isinstance(dq, DualQuaternion):
            self.dq = dq
        else:
            raise TypeError('Pass a DualQuaternion or use fromComponents')

    @classmethod
    def fromComponents(cls, qr, qt):
        if abs(qr.mag()-1) > .0000000001:
            raise ValueError('Rotation quaternion must have magnitude 1')
        if not qt.real == 0:
            raise ValueError('Translation quaternion must be pure imaginary')
        return cls(DualQuaternion(qr, .5*qt*qr))

    def of(self, pose):
        if isinstance(pose, Pose):
            ret = Pose(self.dq*pose.dq*self.dq.conj())
            ret.translation().real = 0
            return ret
        else:
            raise TypeError('must apply transform to a pose')

    def compose(self, other):
        return Transform(self.dq*other.dq)

    def rotation(self):
        return self.dq.a

    def translation(self):
        return 2*self.dq.b*self.dq.a.conj()/(self.dq.a*self.dq.a.conj()).real

    def asVector(self):
        return matrix(concatenate((self.rotation().asVector(),self.translation().asVector())))

    def __str__(self):
        return "Rot:" + str(self.rotation()) + "Trans:" + str(self.translation())

#r = Quaternion(0,1,0,0)
#basis = matrix([[1,0,0],[0,0,0],[0,1,0],[0,0,1]])
#m = matrix([[1],[0],[0]])
#covar = matrix([[.3,.2,0],[.2,.5,.6],[0,.6,1]])
#start = clock()
#y = RPG(r,basis,m,covar)
#end = clock()
#print (end-start)

#x = Pose.fromComponents(Quaternion(1,2,3,4).normalize(), Quaternion(0,1,2,3))
#t = Transform.fromComponents(Quaternion(-1,-2,-3,-4).normalize(),Quaternion(0,6,7,8))
#p = Pose.fromComponents(Quaternion(1,0,0,0),Quaternion(0,9,8,7))
#t1 = Transform.fromComponents(Quaternion(1,1,1,1).normalize(),Quaternion(0,1,2,3))
#p2 = Pose.fromComponents(Quaternion(1,0,0,0),Quaternion(0,0,0,0))
