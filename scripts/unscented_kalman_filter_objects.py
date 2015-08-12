"""
General (augmented state) Unscented Kalman Filter implementation for compositions of Euclidean positions, angles, and poses


"""

from numpy import array, matrix, concatenate, zeros, ones, logical_not, random, around, real, transpose
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


    def step(self, measurement, indices, command=None, W_0=.001, scoreThresh=[None]):
        #Select sigma points
        weights = [W_0] + [(1-W_0)/(2*self.mean.shape[0])]*(2*self.mean.shape[0])
        points = [self.mean]+self.selectSigmaPoints(self.mean, self.covar, W_0)
        objectPoints = self.packPoints(points, range(len(self.dims)), self.origins, self.bases)
        
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

        #print "pre measurement update:"
        #print self.mean[10:14,0]
        #print self.origins[5]

        #Measurement update
        if len(measurement) > 0:
            allIndices = []
            for i in indices:
                if isinstance(i, list):
                    allIndices.extend(i)
                else:
                    allIndices.append(i)

            #Predict measurements at forecast sigma points
            measurement_forecast = []
            for i in xrange(len(state_forecast)):
                t = self.measurement_predict_function(state_forecast[i], points[i][self.stateDim+self.processNoiseDim:,0])
                measurement_forecast.append(t)#[t[j] for j in allIndices])

            #measurement_forecast = measurement_forecast_all


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
            
            #print "origins"
            #print measurement_origins
            #for j in measurement:
            #    print j.getOrigin()
            #print "---"
            #for j in measurement_origins:
            #    print j

            #Find measurement dimensions
            measurementCumDims = []
            current = 0
            first = True
            for o in measurement_forecast[0]:
                measurementCumDims.append(current)
                current += o.dim
            measurementCumDims.append(current)

            #print "MeasurementCumDims", measurementCumDims

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

            #print "mean z all"
            #print mean_z
            #print "covar z all"
            #print covar_z

            #Repack mean z
            mean_z_objects = self.packPoints([mean_z],range(len(measurement_forecast[0])),measurement_origins,measurement_bases)[0]

            #Associate data
            measurement_associated = []
            indices_associated = []

            #print "measurement"
            #print measurement
            #print "indices"
            #print indices
            #print "mean"
            #print mean_z_objects

            for i in xrange(len(measurement)):
                if isinstance(measurement[i], list):
                    ai = self.associateData(measurement[i], [mean_z_objects[j] for j in indices[i]], indices[i], scoreThresh[i%len(scoreThresh)])
                    if ai != None:
                        measurement_associated.extend(measurement[i])
                        indices_associated.extend(ai)
                    else:
                        #association error
                        raise Exception("Error associating measurements")
                else:
                    measurement_associated.append(measurement[i])
                    indices_associated.append(indices[i])
            #print "indices_associated"
            #print indices_associated

            sliceIndices = []
            for i in indices_associated:
                sliceIndices.extend(range(measurementCumDims[i],measurementCumDims[i+1]))
            #print "sliceIndices"
            #print sliceIndices
            mean_z = mean_z[array(sliceIndices)]
            covar_z = covar_z[array(sliceIndices).reshape((len(sliceIndices),1)), array(sliceIndices)]

            #print "mean z"
            #print mean_z
            #print "covar z"
            #print covar_z
                
            #Calculate Kalman gain
            cross_covar = 0
            for i in xrange(len(measurement_forecast)):
                cross_covar += weights[i]*((state_forecast_projected[i]-self.mean)*(measurement_forecast_projected[i][array(sliceIndices)]-mean_z).getT())
            gain = cross_covar * covar_z.getI()
        
            #Reproject Measurement
            measurement_projected = matrix([[]]).reshape((0,1))
            for i in xrange(len(measurement_associated)):
                measurement_projected = matrix(concatenate((measurement_projected,measurement_associated[i].log(measurement_origins[indices_associated[i]],measurement_bases[indices_associated[i]],self.syms[indices_associated[i]]))))
            #print "diff and gain:"
            #print measurement_projected
            #print (measurement_projected-mean_z)
            #print gain
            #print "adding to mean:"
            #print gain*(measurement_projected-mean_z)
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
        return (range(len(self.dims)), mRet, self.getPackedSigmaPoints(self.mean[0:self.stateDim], self.covar[0:self.stateDim,0:self.stateDim], range(len(self.dims))))

    def sampleMarginalDistribution(self):
        sample = matrix(random.multivariate_normal(mean=self.mean[0:self.stateDim].getT().tolist()[0], cov=self.covar[0:self.stateDim,0:self.stateDim])).getT()
        #Pack in order to return
        counter = 0
        ret = []
        for i in xrange(len(self.dims)):
            ret.append(self.types[i].exp(sample[counter:counter+self.dims[i]],self.origins[i],self.bases[i]))
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
            mRet.append(self.types[rest[i]].exp(mPrime[counter:counter+self.dims[rest[i]],0],self.origins[rest[i]],self.bases[rest[i]]))
            counter += self.dims[rest[i]]
        return (rest, mRet, self.getPackedSigmaPoints(mPrime, sPrime, rest))
        

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
            ret.append(self.types[i].exp(sample[counter:counter+self.dims[i]],self.origins[i],self.bases[i]))
            counter += self.dims[i]
        return ret

    def selectSigmaPoints(self, mean, covar, W_0=.001):
        points = []
        root = matrix(real(linalg.sqrtm(mean.shape[0]/(1-W_0)*covar)))
        for i in xrange(mean.shape[0]):
            points.append(mean+root[i].getT())
        for i in xrange(mean.shape[0]):
            points.append(mean-root[i].getT())
        return points

    def packPoints(self, points, indices, origins, bases):
        objectPoints = []
        for p in points:
            counter = 0
            l = []
            for i in indices:
                l.append(self.types[i].exp(p[counter:counter+self.dims[i],0],origins[i],bases[i]))
                counter += self.dims[i]
            objectPoints.append(l)
        return objectPoints

    def getPackedSigmaPoints(self, mean, covar, indices, W_0=.001):
        return self.packPoints(self.selectSigmaPoints(mean, covar, W_0), indices, self.origins, self.bases)

    #returns list of indices in order of associated measurements
    def associateData(self, measurement, measurement_forecast, indices, scoreThreshold):
        #print "associating data"
        #for i in measurement:
        #    print i
        #for i in measurement_forecast:
        #    print i
        #print indices
        ret = []
        used = []
        for i in xrange(len(measurement)):
            bestScore = float("inf")
            bestIndex = -1
            for j in xrange(len(measurement_forecast)):
                if indices[j] in used:
                    continue
                score = measurement[i].scoreEquals(measurement_forecast[j])
                #print "measurement", i, "index", j, "score", score
                if score < bestScore:
                    bestScore = score
                    bestIndex = indices[j]
            if not scoreThreshold or bestScore < scoreThreshold:
                ret.append(bestIndex)
                used.append(bestIndex)
                #print "assigned measurement", i, "to index", bestIndex
            else:
                return None
        return ret
