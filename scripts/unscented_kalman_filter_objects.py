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
        self.state_dim = 0
        for o in mean:
            self.dims.append(o.dim)
            self.state_dim += o.dim
            self.types.append(o.__class__)

        #Dimension of each portion of the augmentation
        self.process_noise_dim = Q.shape[0]
        self.measurement_noise_dim = R.shape[0]
        self.dim = self.state_dim + self.process_noise_dim + self.measurement_noise_dim
        
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
        deltas = selectSigmaPoints(matrix(zeros((self.covar.shape[0],1))),self.covar, W_0)
        object_points = packPoints(self.mean, deltas, range(len(self.dims)), self.types, self.dims)
        
        #Apply unscented transformation
        state_forecast = []
        for i in xrange(len(object_points)):
            state_forecast.append(self.update_function(command, object_points[i],deltas[i][self.state_dim:self.state_dim+self.process_noise_dim,0]))

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
        tangent_space_mean = matrix(zeros((self.state_dim,1)))

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
                t = self.measurement_predict_function(state_forecast[i], deltas[i][self.state_dim+self.process_noise_dim:,0])
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
                    ai = associateData(measurement[i], [(measurement_means[j],mean_z[measurement_cum_dims[j]:measurement_cum_dims[j+1]],covar_z[measurement_cum_dims[j]:measurement_cum_dims[j+1],measurement_cum_dims[j]:measurement_cum_dims[j+1]]) for j in indices[i]], indices[i], scoreThresh[i%len(scoreThresh)])
                    if ai != None:
                        measurement_associated.extend(measurement[i])
                        indices_associated.extend(ai)
                    else:
                        #association error
                        raise Exception("Error associating measurements")
                else:
                    measurement_associated.append(measurement[i])
                    indices_associated.append(indices[i])

            slice_indices = []
            associated_cum_dims = [0]
            current = 0
            for i in indices_associated:
                slice_indices.extend(range(measurement_cum_dims[i],measurement_cum_dims[i+1]))
                current += measurement_cum_dims[i+1]-measurement_cum_dims[i]
                associated_cum_dims.append(current)

            mean_z = mean_z[array(slice_indices)]
            covar_z = covar_z[array(slice_indices).reshape((len(slice_indices),1)), array(slice_indices)]
                
            #Calculate Kalman gain
            cross_covar = 0
            for i in xrange(len(measurement_forecast)):
                cross_covar += weights[i]*((state_forecast_projected[i]-tangent_space_mean)*(measurement_forecast_projected[i][array(slice_indices)]-mean_z).getT())

            gain = cross_covar * covar_z.getI()

            #Reproject Measurement
            measurement_projected = matrix([[]]).reshape((0,1))
            for i in xrange(len(measurement_associated)):
                measurement_projected = matrix(concatenate((measurement_projected,measurement_means[indices_associated[i]].log(measurement_associated[i],(self.syms[indices_associated[i]],mean_z[associated_cum_dims[i]:associated_cum_dims[i+1]],covar_z[associated_cum_dims[i]:associated_cum_dims[i+1],associated_cum_dims[i]:associated_cum_dims[i+1]])))))

            tangent_space_mean = tangent_space_mean + gain*(measurement_projected-mean_z)
            
            #Update covariance
            covar_state = covar_state - gain*covar_z*gain.getT()

        #Measurement-corrected updates to mean and covariance

        self.mean = packPoints(forecast_mean, [tangent_space_mean], range(len(forecast_mean)), self.types, self.dims)[0]

        self.covar[0:covar_state.shape[0],0:covar_state.shape[1]] = covar_state
        
    def getMarginalDistribution(self):
        #Pack in order to return
        return (range(len(self.dims)), getPackedSigmaPoints(self.mean, matrix(zeros((self.state_dim,1))), self.covar[0:self.state_dim,0:self.state_dim], range(len(self.dims)), self.types, self.dims))

    def sampleMarginalDistribution(self):
        sample = matrix(random.multivariate_normal(zeros((self.state_dim)), cov=self.covar[0:self.state_dim,0:self.state_dim])).getT()
        
        #Pack in order to return
        return packPoints(self.mean, [sample], range(len(self.dims)), self.types, self.dims)[0]

    def getConditionalDistribution(self, indices, values):
        #Calculate cumulative indices in the mean vector
        counter = 0
        cum_indices = []
        for i in range(len(self.dims)):
            if i in indices:
                cum_indices.extend(range(counter,counter+self.dims[i]))
            counter += self.dims[i]

        #Mask with newly calculated indices
        state_mask = ones(self.state_dim,dtype=bool)
        state_mask[cum_indices] = False
        condition_mask = logical_not(state_mask)
        s11 = self.covar[state_mask][:,state_mask]
        s12 = self.covar[state_mask][:,condition_mask]
        s21 = self.covar[condition_mask][:,state_mask]
        s22 = self.covar[condition_mask][:,condition_mask]
        m1 = zeros((sum(state_mask),1))
        m2 = zeros((sum(condition_mask),1))

        #Project conditioned values
        value_projected = matrix([[]]).reshape((0,1))
        for i in xrange(len(values)):
            value_projected = matrix(concatenate((value_projected,self.mean[indices[i]].log(values[i]))))

        #Calculate new mean
        m_prime = m1 + s12*s22.getI()*(value_projected-m2)
        s_prime = s11 - s12*s22.getI()*s21
        rest = list(set(range(len(self.dims)))-set(indices))
        return (rest, getPackedSigmaPoints(self.mean, m_prime, s_prime, rest, self.types, self.dims))
        

    def sampleConditionalDistribution(self, indices, values):
        #Calculate cumulative indices in the mean vector
        counter = 0
        cum_indices = []
        for i in range(len(self.dims)):
            if i in indices:
                for j in range(counter,counter+self.dims[i]):
                    cum_indices.append(j)
            counter += self.dims[i]

        #Mask with newly calculated indices
        state_mask = ones(self.state_dim,dtype=bool)
        state_mask[cum_indices] = False
        condition_mask = logical_not(state_mask)
        s11 = self.covar[state_mask][:,state_mask]
        s12 = self.covar[state_mask][:,condition_mask]
        s21 = self.covar[condition_mask][:,state_mask]
        s22 = self.covar[condition_mask][:,condition_mask]
        m1 = zeros((sum(state_mask),1))
        m2 = zeros((sum(condition_mask),1))

        #Project conditioned values
        value_projected = matrix([[]]).reshape((0,1))
        for i in xrange(len(values)):
            value_projected = matrix(concatenate((value_projected,self.mean[indices[i]].log(values[i]))))

        #Calculate new mean
        m_prime = m1 + s12*s22.getI()*(value_projected-m2)
        s_prime = s11 - s12*s22.getI()*s21
        sample = matrix(random.multivariate_normal(mean=m_prime.getT().tolist()[0], cov=s_prime)).getT()

        #Pack in order to return
        rest = list(set(range(len(self.dims)))-set(indices))
        return packPoints(self.mean, [sample], rest, self.types, self.dims)[0]

def selectSigmaPoints(delta, covar, W_0=.001):
    deltas = [delta]
    root = matrix(real(linalg.sqrtm(covar.shape[0]/(1-W_0)*covar)))
    for i in xrange(covar.shape[0]):
        deltas.append(delta+root[i].getT())
    for i in xrange(covar.shape[0]):
        deltas.append(delta-root[i].getT())
    return deltas

def packPoints(mean, deltas, indices, types, dims):
    object_points = []
    for d in deltas:
        counter = 0
        l = []
        for i in indices:
            l.append(mean[i].exp(d[counter:counter+dims[i]]))
            counter += dims[i]
        object_points.append(l)
    return object_points

def getPackedSigmaPoints(mean, delta, covar, indices, types, dims, W_0=.001):
    return packPoints(mean, selectSigmaPoints(delta, covar, W_0), indices, types, dims)

#returns list of indices in order of associated measurements
def associateData(measurement, measurement_forecast, indices, scoreThreshold):
    ret = []
    used = []
    for i in xrange(len(measurement)):
        bestScore = float("-inf")
        bestIndex = -1
        for j in xrange(len(measurement_forecast)):
            if indices[j] in used:
                continue
            score = measurement[i].scoreEquals(*measurement_forecast[j])
            if score > bestScore:
                bestScore = score
                bestIndex = indices[j]
        if not scoreThreshold or bestScore > scoreThreshold:
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
