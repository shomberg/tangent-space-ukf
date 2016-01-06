"""
General (augmented state) Unscented Kalman Filter implementation for compositions of Euclidean positions, angles, and poses


"""

from numpy import array, matrix, concatenate, zeros, ones, logical_not, random, around, real, transpose
from scipy import linalg
import math
from kalman_util import append_matrices
import pdb

class unscentedKalmanFilter:
    def __init__(self, mean, covars, f, h, symmetries, Q, R):
        self.version = 1
        #Unpack mean vector from objects
        self.mean = matrix([[]]).reshape((0,1))
        self.dims = []
        self.origins = []
        self.types = []
        self.mean = mean
        self.stateDim = 0
        for o in mean:
            self.dims.append(o.dim)
            self.stateDim += o.dim
            self.types.append(o.__class__)

        #Dimension of each portion of the augmentation
        self.processNoiseDim = Q.shape[0]
        self.measurementNoiseDim = R.shape[0]
        self.dim = self.stateDim + self.processNoiseDim + self.measurementNoiseDim
        
        #Unpack covariances for diagonal from objects
        self.covar = append_matrices(covars)
        
        #Augmented covariance matrix
        self.covar = append_matrices([self.covar,Q,R])
        self.update_function = f
        self.measurement_predict_function = h
        self.syms = symmetries


    def step(self, measurement, indices, command=None, W_0=.001, scoreThresh=[None]):
        #Select sigma points
        weights = [W_0] + [(1-W_0)/(2*self.dim)]*(2*self.dim)
        deltas = selectSigmaPoints(self.covar, W_0)
        objectPoints = packPoints(self.mean, deltas, range(len(self.dims)), self.types, self.dims)
        
        #Apply unscented transformation
        state_forecast = []
        for i in xrange(len(objectPoints)):
            state_forecast.append(self.update_function(command, objectPoints[i],deltas[i][self.stateDim:self.stateDim+self.processNoiseDim,0]))

        #Calculate appropriate(mean) space for reprojection
        forecast_mean = []
        for i in range(len(self.types)):
            forecast_mean.append(self.types[i].calculateMean(weights, [state_forecast[j][i] for j in range(len(state_forecast))]))

        #Reproject
        state_forecast_projected = []
        for x in state_forecast:
            add = matrix([[]]).reshape((0,1))
            for i in xrange(len(x)):
                add = matrix(concatenate((add,forecast_mean[i].log(x[i]))))
            state_forecast_projected.append(add)

        #Augmented state vector
        tangent_space_mean = matrix(zeros((self.stateDim,1)))

        #Update mean with forecast from sigma points
        for i in xrange(len(state_forecast_projected)):
            tangent_space_mean += state_forecast_projected[i]*weights[i]

        covar_state = 0
        for i in xrange(len(state_forecast_projected)):
            deviation = state_forecast_projected[i]-tangent_space_mean
            covar_state += weights[i]*(deviation*deviation.getT())

        #Measurement update
        if len(measurement) > 0:
            allIndices = flatten_list(indices)

            #Predict measurements at forecast sigma points
            measurement_forecast = []
            for i in xrange(len(state_forecast)):
                t = self.measurement_predict_function(state_forecast[i], deltas[i][self.stateDim+self.processNoiseDim:,0])
                measurement_forecast.append(t)

            measurement_types = []
            for m in measurement_forecast[0]:
                measurement_types.append(m.__class__)

            #Calculate appropriate(mean) space for reprojection
            measurement_means = []
            for i in range(len(measurement_types)):
                measurement_means.append(measurement_types[i].calculateMean(weights, [measurement_forecast[j][i] for j in range(len(measurement_forecast))]))

            #Find measurement dimensions
            measurement_dims = []
            measurement_cum_dims = []
            current = 0
            first = True
            for o in measurement_forecast[0]:
                measurement_dims.append(o.dim)
                measurement_cum_dims.append(current)
                current += o.dim

            measurement_cum_dims.append(current)

            #Reproject
            measurement_forecast_projected = []
            for x in measurement_forecast:
                add = matrix([[]]).reshape((0,1))
                for i in xrange(len(x)):
                    add = matrix(concatenate((add,measurement_means[i].log(x[i]))))
                measurement_forecast_projected.append(add)

            #Update measurement forecast mean and covariance
            mean_z = 0
            for i in xrange(len(measurement_forecast_projected)):
                mean_z += weights[i]*measurement_forecast_projected[i]

            covar_z = 0
            for i in xrange(len(measurement_forecast)):
                deviation = measurement_forecast_projected[i]-mean_z
                covar_z += weights[i]*(deviation*deviation.getT())

            #Repack mean z
            mean_z_objects = packPoints(measurement_means,[mean_z],range(len(measurement_forecast[0])),measurement_types,measurement_dims)[0]

            #Associate data
            measurement_associated = []
            indices_associated = []

            for i in xrange(len(measurement)):
                if isinstance(measurement[i], list):
                    ai = associateData(measurement[i], [mean_z_objects[j] for j in indices[i]], indices[i], scoreThresh[i%len(scoreThresh)])
                    if ai != None:
                        measurement_associated.extend(measurement[i])
                        indices_associated.extend(ai)
                    else:
                        #association error
                        raise Exception("Error associating measurements")
                else:
                    measurement_associated.append(measurement[i])
                    indices_associated.append(indices[i])

            sliceIndices = []
            for i in indices_associated:
                sliceIndices.extend(range(measurement_cum_dims[i],measurement_cum_dims[i+1]))

            mean_z = mean_z[array(sliceIndices)]
            covar_z = covar_z[array(sliceIndices).reshape((len(sliceIndices),1)), array(sliceIndices)]
                
            #Calculate Kalman gain
            cross_covar = 0
            for i in xrange(len(measurement_forecast)):
                cross_covar += weights[i]*((state_forecast_projected[i]-tangent_space_mean)*(measurement_forecast_projected[i][array(sliceIndices)]-mean_z).getT())

            gain = cross_covar * covar_z.getI()

            #Reproject Measurement
            measurement_projected = matrix([[]]).reshape((0,1))
            for i in xrange(len(measurement_associated)):
                measurement_projected = matrix(concatenate((measurement_projected,measurement_means[indices_associated[i]].log(measurement_associated[i],self.syms[indices_associated[i]]))))

            tangent_space_mean = tangent_space_mean + gain*(measurement_projected-mean_z)
            
            #Update covariance
            covar_state = covar_state - gain*covar_z*gain.getT()

        #Measurement-corrected updates to mean and covariance

        self.mean = packPoints(forecast_mean, [tangent_space_mean], range(len(forecast_mean)), self.types, self.dims)[0]

        self.covar[0:covar_state.shape[0],0:covar_state.shape[1]] = covar_state
        
    def getMarginalDistribution(self):
        #Pack in order to return
        counter = 0
        mRet = []
        for i in xrange(len(self.dims)):
            mRet.append(self.types[i].exp(self.mean[counter:counter+self.dims[i],0],self.origins[i]))
            counter += self.dims[i]
        
        #Returns updated mean as list of objects
        return (range(len(self.dims)), mRet, getPackedSigmaPoints(self.mean[0:self.stateDim], self.covar[0:self.stateDim,0:self.stateDim], range(len(self.dims)), self.types, self.dims, self.origins))

    def sampleMarginalDistribution(self):
        sample = matrix(random.multivariate_normal(mean=self.mean[0:self.stateDim].getT().tolist()[0], cov=self.covar[0:self.stateDim,0:self.stateDim])).getT()
        #Pack in order to return
        counter = 0
        ret = []
        for i in xrange(len(self.dims)):
            ret.append(self.types[i].exp(sample[counter:counter+self.dims[i]],self.origins[i]))
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
            value_projected = matrix(concatenate((value_projected,values[i].log(self.origins[indices[i]]))))

        #Calculate new mean
        mPrime = m1 + s12*s22.getI()*(value_projected-m2)
        sPrime = s11 - s12*s22.getI()*s21
        mRet = []
        counter = 0
        rest = list(set(range(len(self.dims)))-set(indices))
        for i in range(len(rest)):
            mRet.append(self.types[rest[i]].exp(mPrime[counter:counter+self.dims[rest[i]],0],self.origins[rest[i]]))
            counter += self.dims[rest[i]]
        return (rest, mRet, getPackedSigmaPoints(mPrime, sPrime, rest, self.types, self.dims, self.origins))
        

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
            value_projected = matrix(concatenate((value_projected,values[i].log(self.origins[indices[i]]))))

        #Calculate new mean
        mPrime = m1 + s12*s22.getI()*(value_projected-m2)
        sPrime = s11 - s12*s22.getI()*s21

        sample = matrix(random.multivariate_normal(mean=mPrime.getT().tolist()[0], cov=sPrime[0:self.stateDim,0:self.stateDim])).getT()
        #Pack in order to return
        counter = 0
        ret = []
        rest = list(set(range(len(self.dims)))-set(indices))
        for i in rest:
            ret.append(self.types[i].exp(sample[counter:counter+self.dims[i]],self.origins[i]))
            counter += self.dims[i]
        return ret

def selectSigmaPoints(covar, W_0=.001):
    deltas = [matrix(zeros((covar.shape[0],1)))]
    root = matrix(real(linalg.sqrtm(covar.shape[0]/(1-W_0)*covar)))
    for i in xrange(covar.shape[0]):
        deltas.append(root[i].getT())
    for i in xrange(covar.shape[0]):
        deltas.append(-root[i].getT())
    return deltas

def packPoints(mean, deltas, indices, types, dims):
    objectPoints = []
    for d in deltas:
        counter = 0
        l = []
        for i in indices:
            l.append(mean[i].exp(d[counter:counter+dims[i]]))
            counter += dims[i]
        objectPoints.append(l)
    return objectPoints

def getPackedSigmaPoints(mean, covar, indices, types, dims, W_0=.001):
    return packPoints(mean, selectSigmaPoints(covar, W_0), indices, types, dims)

#returns list of indices in order of associated measurements
def associateData(measurement, measurement_forecast, indices, scoreThreshold):
    ret = []
    used = []
    for i in xrange(len(measurement)):
        bestScore = float("inf")
        bestIndex = -1
        for j in xrange(len(measurement_forecast)):
            if indices[j] in used:
                continue
            score = measurement[i].scoreEquals(measurement_forecast[j])
            if score < bestScore:
                bestScore = score
                bestIndex = indices[j]
        if not scoreThreshold or bestScore < scoreThreshold:
            ret.append(bestIndex)
            used.append(bestIndex)
        else:
            return None
    return ret

def flatten_list(l):
    ret = []
    for i in l:
        if isinstance(i, (list,tuple)):
            ret.extend(flatten_list(i))
        else:
            ret.append(i)
    return ret
