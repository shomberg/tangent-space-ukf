from numpy import matrix, identity, column_stack, concatenate
from scipy import linalg

class unscentedKalman:
    def __init__(self, x_0, P_0, f, h, Q, R):
        self.mean = x_0.copy()
        self.covar = P_0.copy()
        self.update_function = f
        self.measurement_predict_function = h
        self.covar_update_noise = Q.copy()
        self.covar_measurement_noise = R.copy()

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


        #time update
        #apply unscented transformation
        state_forecast = []
        for x in points:
            state_forecast.append(self.update_function(x))

        #update
        self.mean = 0
        for i in xrange(len(state_forecast)):
            self.mean += state_forecast[i]*weights[i]

        self.covar = self.covar_update_noise.copy()
        for i in xrange(0,len(state_forecast)):
            deviation = state_forecast[i]-self.mean
            self.covar += weights[i]*(deviation*deviation.getT())

        #measurement update
        #apply unscented transformation
        measurement_forecast = []
        for x in state_forecast:
            measurement_forecast.append(self.measurement_predict_function(x))
        mean_z = 0
        for i in xrange(len(measurement_forecast)):
            mean_z += weights[i]*measurement_forecast[i]
        covar_z = self.covar_measurement_noise.copy()
        for i in xrange(len(measurement_forecast)):
            deviation = measurement_forecast[i]-mean_z
            covar_z += weights[i]*(deviation*deviation.getT())
        #calculate gain
        cross_covar = 0
        for i in xrange(len(measurement_forecast)):
            cross_covar += weights[i]*((state_forecast[i]-self.mean)*(measurement_forecast[i]-mean_z).getT())
        gain = cross_covar * covar_z.getI()
        #update
        self.mean = self.mean + gain*(measurement-mean_z)
        self.covar = self.covar - gain*covar_z*gain.getT()

