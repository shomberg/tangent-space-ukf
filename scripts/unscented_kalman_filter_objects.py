"""
General (augmented state) Unscented Kalman Filter implementation for compositions of Euclidean positions, angles, and poses


"""

from numpy import matrix, concatenate, zeros, ones, logical_not, random, around, real
from scipy import linalg
import math
from kalman_util import append_matrices

class unscentedKalmanFilter:
    def __init__(self, objects, covars, bases, f, h, symmetries, Q, R):
        #Unpack mean vector from objects
        self.mean = matrix([[]]).reshape((0,1))
        self.dims = []
        self.origins = []
        self.types = []
        for o in objects:
            self.mean = matrix(concatenate((self.mean, o.getCoordinates())))
            self.dims.append(o.getCoordinates().shape[0])
            self.origins.append(o.getOrigin())
            self.types.append(o.__class__)
        self.stateDim = self.mean.shape[0]
        self.bases = bases
        
        #Augmented state vector
        self.mean = matrix(concatenate((self.mean,zeros((Q.shape[0],1)),zeros((R.shape[0],1)))))
        
        #Dimension of each portion of the augmentation
        self.processNoiseDim = Q.shape[0]
        self.measurementNoiseDim = R.shape[0]
        
        #Unpack covariances for diagonal from objects
        self.covar = append_matrices(covars)
        
        #Augmented covariance matrix
        self.covar = append_matrices([self.covar,Q,R])
        self.update_function = f
        self.measurement_predict_function = h
        self.syms = symmetries


    def step(self, measurement, indices, command=None, W_0=.001):
        #Sigma point selection
        points = []
        weights = [W_0] + [(1-W_0)/(2*self.mean.shape[0])]*(2*self.mean.shape[0])
        points.append(self.mean)
        root = matrix(real(linalg.sqrtm(self.mean.shape[0]/(1-W_0)*self.covar)))
        for i in xrange(self.mean.shape[0]):
            points.append(self.mean+root[i].getT())
        for i in xrange(self.mean.shape[0]):
            points.append(self.mean-root[i].getT())

        #Time update
        #Repack sigma points into objects
        objectPoints = []
        for p in points:
            counter = 0
            l = []
            for i in xrange(len(self.dims)):
                l.append(self.types[i].exp(p[counter:counter+self.dims[i],0],self.origins[i],self.bases[i]))
                counter += self.dims[i]
            objectPoints.append(l)

        #Apply unscented transformation
        state_forecast = []
        for i in xrange(len(objectPoints)):
            state_forecast.append(self.update_function(command, objectPoints[i],points[i][self.stateDim:self.stateDim+self.processNoiseDim,0]))

        #Calculate appropriate(mean) space for reprojection
        self.origins = [0]*len(state_forecast[0])
        for i in xrange(len(state_forecast)):
            for j in xrange(len(state_forecast[i])):
                self.origins[j] += weights[i]*state_forecast[i][j].getOrigin()/len(state_forecast)
        #Normalize mean
        for j in xrange(len(self.origins)):
            self.origins[j] = self.types[j].normalizeOrigin(self.origins[j])
            self.bases[j] = self.types[j].generateBasis(self.origins[j])
        
        #Reproject
        state_forecast_projected = []
        for x in state_forecast:
            add = matrix([[]]).reshape((0,1))
            for i in xrange(len(x)):
                add = matrix(concatenate((add,x[i].log(self.origins[i],self.bases[i]))))
            state_forecast_projected.append(add)

        #Update mean with forecast from sigma points
        self.mean = 0
        for i in xrange(len(state_forecast_projected)):
            self.mean += state_forecast_projected[i]*weights[i]

        covar_state = 0
        for i in xrange(len(state_forecast_projected)):
            deviation = state_forecast_projected[i]-self.mean
            covar_state += weights[i]*(deviation*deviation.getT())

        #Measurement update
        if len(measurement) > 0:
            #Predict measurements at forecast sigma points
            measurement_forecast = []
            for i in xrange(len(state_forecast)):
                t = self.measurement_predict_function(state_forecast[i], points[i][self.stateDim+self.processNoiseDim:,0])
                measurement_forecast.append([t[j] for j in indices])

            measurement_types = []
            for m in measurement_forecast[0]:
                measurement_types.append(m.__class__)

            #Calculate appropriate(mean) space for reprojection
            measurement_origins = [0]*len(measurement_forecast[0])
            measurement_bases = [0]*len(measurement_forecast[0])
            for i in xrange(len(measurement_forecast)):
                for j in xrange(len(measurement_forecast[i])):
                    measurement_origins[j] += weights[i]*measurement_forecast[i][j].getOrigin()/len(measurement_forecast)
            for j in xrange(len(measurement_origins)):
                measurement_origins[j] = measurement_types[j].normalizeOrigin(measurement_origins[j])
                measurement_bases[j] = measurement_types[j].generateBasis(measurement_origins[j])
        
            #Reproject
            measurement_forecast_projected = []
            for x in measurement_forecast:
                add = matrix([[]]).reshape((0,1))
                for i in xrange(len(x)):
                    add = matrix(concatenate((add,x[i].log(measurement_origins[i],measurement_bases[i]))))
                measurement_forecast_projected.append(add)

            #Update measurement forecast mean and covariance
            mean_z = 0
            for i in xrange(len(measurement_forecast_projected)):
                mean_z += weights[i]*measurement_forecast_projected[i]
            covar_z = 0
            for i in xrange(len(measurement_forecast)):
                deviation = measurement_forecast_projected[i]-mean_z
                covar_z += weights[i]*(deviation*deviation.getT())

            #Calculate Kalman gain
            cross_covar = 0
            for i in xrange(len(measurement_forecast)):
                cross_covar += weights[i]*((state_forecast_projected[i]-self.mean)*(measurement_forecast_projected[i]-mean_z).getT())
            gain = cross_covar * covar_z.getI()
        
            #Reproject Measurement
            measurement_projected = matrix([[]]).reshape((0,1))
            for i in xrange(len(measurement)):
                measurement_projected = matrix(concatenate((measurement_projected,measurement[i].log(measurement_origins[i],measurement_bases[i],self.syms[i]))))

            self.mean = self.mean + gain*(measurement_projected-mean_z)
            
            #Update covariance
            covar_state = covar_state - gain*covar_z*gain.getT()

        #Measurement-corrected updates to mean and covariance
        self.mean = concatenate((self.mean, zeros((self.processNoiseDim+self.measurementNoiseDim,1))))

        self.covar[0:covar_state.shape[0],0:covar_state.shape[1]] = covar_state
        
    def getMarginalDistribution(self):
        #Pack in order to return
        counter = 0
        mRet = []
        for i in xrange(len(self.dims)):
            mRet.append(self.types[i].exp(self.mean[counter:counter+self.dims[i],0],self.origins[i],self.bases[i]))
            counter += self.dims[i]
        
        #Returns updated mean as list of objects
        return (range(len(self.dims)), mRet, self.getSigmaPoints(self.mean[0:self.stateDim], self.covar[0:self.stateDim,0:self.stateDim], range(len(self.dims))))

    def sampleMarginalDistribution(self):
        sample = matrix(random.multivariate_normal(mean=self.mean[0:self.stateDim].getT().tolist()[0], cov=self.covar[0:self.stateDim,0:self.stateDim])).getT()
        #Pack in order to return
        counter = 0
        ret = []
        for i in xrange(len(self.dims)):
            ret.append(self.types[i].exp(*[sample[counter:counter+self.dims[i]]]+[self.origins[i]]+[self.bases[i]]))
            counter += self.dims[i]
        return ret

    def getConditionalDistribution(self, indices, values):
        #Calculate cumulative indices in the mean vector
        counter = 0
        cumIndices = []
        for i in range(len(self.dims)):
            if i in indices:
                for j in range(counter,counter+self.dims[i]):
                    cumIndices.append(j)
            counter += self.dims[i]

        #Mask with newly calculated indices
        stateMask = ones(self.stateDim,dtype=bool)
        stateMask[cumIndices] = False
        conditionMask = logical_not(stateMask)
        s11 = self.covar[stateMask][:,stateMask]
        s12 = self.covar[stateMask][:,conditionMask]
        s21 = self.covar[conditionMask][:,stateMask]
        s22 = self.covar[conditionMask][:,conditionMask]
        m1 = self.mean[stateMask]
        m2 = self.mean[conditionMask]

        #Project conditioned values
        value_projected = matrix([[]]).reshape((0,1))
        for i in xrange(len(values)):
            value_projected = matrix(concatenate((value_projected,values[i].log(self.origins[indices[i]],self.bases[indices[i]]))))

        #Calculate new mean
        mPrime = m1 + s12*s22.getI()*(value_projected-m2)
        sPrime = s11 - s12*s22.getI()*s21
        mRet = []
        counter = 0
        rest = list(set(range(len(self.dims)))-set(indices))
        for i in range(len(rest)):
            mRet.append(self.types[rest[i]].exp(*[mPrime[counter:counter+self.dims[rest[i]],0]]+[self.origins[rest[i]]]+[self.bases[rest[i]]]))
            counter += self.dims[rest[i]]
        return (rest, mRet, self.getSigmaPoints(mPrime, sPrime, rest))
        

    def sampleConditionalDistribution(self, indices, values):
        #Calculate cumulative indices in the mean vector
        counter = 0
        cumIndices = []
        for i in range(len(self.dims)):
            if i in indices:
                for j in range(counter,counter+self.dims[i]):
                    cumIndices.append(j)
            counter += self.dims[i]

        #Mask with newly calculated indices
        stateMask = ones(self.stateDim,dtype=bool)
        stateMask[cumIndices] = False
        conditionMask = logical_not(stateMask)
        s11 = self.covar[stateMask][:,stateMask]
        s12 = self.covar[stateMask][:,conditionMask]
        s21 = self.covar[conditionMask][:,stateMask]
        s22 = self.covar[conditionMask][:,conditionMask]
        m1 = self.mean[stateMask]
        m2 = self.mean[conditionMask]

        #Project conditioned values
        value_projected = matrix([[]]).reshape((0,1))
        for i in xrange(len(values)):
            value_projected = matrix(concatenate((value_projected,values[i].log(self.origins[indices[i]],self.bases[indices[i]]))))

        #Calculate new mean
        mPrime = m1 + s12*s22.getI()*(value_projected-m2)
        sPrime = s11 - s12*s22.getI()*s21

        sample = matrix(random.multivariate_normal(mean=mPrime.getT().tolist()[0], cov=sPrime[0:self.stateDim,0:self.stateDim])).getT()
        #Pack in order to return
        counter = 0
        ret = []
        rest = list(set(range(len(self.dims)))-set(indices))
        for i in rest:
            ret.append(self.types[i].exp(*[sample[counter:counter+self.dims[i]]]+[self.origins[i]]+[self.bases[i]]))
            counter += self.dims[i]
        return ret

    def getSigmaPoints(self, mean, covar, indices, W_0=.001):
        points = []
        root = matrix(real(linalg.sqrtm(mean.shape[0]/(1-W_0)*covar)))
        for i in xrange(mean.shape[0]):
            points.append(mean+root[i].getT())
        for i in xrange(mean.shape[0]):
            points.append(mean-root[i].getT())

        objectPoints = []
        for p in points:
            counter = 0
            l = []
            for i in indices:
                l.append(self.types[i].exp(*[p[counter:counter+self.dims[i],0]]+[self.origins[i]]+[self.bases[i]]))
                counter += self.dims[i]
            objectPoints.append(l)
        return objectPoints
