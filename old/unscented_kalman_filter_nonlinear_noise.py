"""
General (augmented state) Unscented Kalman Filter implementation


"""

from numpy import matrix, column_stack, concatenate, zeros
from scipy import linalg

class unscentedKalman:
    def __init__(self, x_0, P_0, f, h, Q, R):
        #Augmented state vector
        self.mean = concatenate((x_0.copy(),zeros((Q.shape[0],1)),zeros((R.shape[0],1))))
        #Dimension of each portion of the augmentation
        self.stateDim = x_0.shape[0]
        self.processNoiseDim = Q.shape[0]
        self.measurementNoiseDim = R.shape[0]
        #Augmented covariance matrix
        self.covar = concatenate((column_stack((P_0.copy(),zeros((P_0.shape[0],Q.shape[1]+R.shape[1])))), column_stack((zeros((Q.shape[1],P_0.shape[1])),Q,zeros((Q.shape[0],R.shape[1])))), column_stack((zeros((R.shape[1],P_0.shape[1]+Q.shape[1])),R))))
        self.update_function = f
        self.measurement_predict_function = h

    def step(self, measurement, W_0):
        #Sigma point selection
        points = []
        weights = []
        points.append(self.mean)
        weights.append(W_0)
        for i in xrange(self.mean.shape[0]):
            points.append(self.mean+matrix(linalg.sqrtm(self.mean.shape[0]/(1-W_0)*self.covar))[i].getT())
            weights.append((1-W_0)/(2*self.mean.shape[0]))
        for i in xrange(self.mean.shape[0]):
            points.append(self.mean-matrix(linalg.sqrtm(self.mean.shape[0]/(1-W_0)*self.covar))[i].getT())
            weights.append((1-W_0)/(2*self.mean.shape[0]))


        #Time update
        #Apply unscented transformation
        state_forecast = []
        for x in points:
            state_forecast.append(self.update_function(x[0:self.stateDim,0].copy(),x[self.stateDim:self.stateDim+self.processNoiseDim,0].copy()))

        #Update mean and covariance with forecast from sigma points
        self.mean = 0
        for i in xrange(len(state_forecast)):
            self.mean += state_forecast[i]*weights[i]

        covar_state = 0
        for i in xrange(len(state_forecast)):
            deviation = state_forecast[i]-self.mean
            covar_state += weights[i]*(deviation*deviation.getT())

        #Measurement update
        #Predict measurements at forecast sigma points
        measurement_forecast = []
        for i in range(len(state_forecast)):
            measurement_forecast.append(self.measurement_predict_function(state_forecast[i], points[i][self.stateDim+self.processNoiseDim:,0]))

        #Update measurement mean and covariance
        mean_z = 0
        for i in xrange(len(measurement_forecast)):
            mean_z += weights[i]*measurement_forecast[i]
        covar_z = 0
        for i in xrange(len(measurement_forecast)):
            deviation = measurement_forecast[i]-mean_z
            covar_z += weights[i]*(deviation*deviation.getT())

        #Calculate Kalman gain
        cross_covar = 0
        for i in xrange(len(measurement_forecast)):
            cross_covar += weights[i]*((state_forecast[i]-self.mean)*(measurement_forecast[i]-mean_z).getT())
        gain = cross_covar * covar_z.getI()
        #Measurement-corrected updates to mean and covariance
        self.mean = concatenate((self.mean + gain*(measurement-mean_z), zeros((self.processNoiseDim+self.measurementNoiseDim,1))))
        covar_state = covar_state - gain*covar_z*gain.getT()
        self.covar[0:covar_state.shape[0],0:covar_state.shape[1]] = covar_state
