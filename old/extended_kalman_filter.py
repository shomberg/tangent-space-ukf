from numpy import matrix, identity

class discreteKalman:
    def __init__(self, x_0, P_0, A, B, Q, H, R):
        self.mean = x_0
        self.covar = P_0
        self.state_update = A
        self.control_update = B
        self.covar_update_noise = Q
        self.measurement_predict = H
        self.covar_measurement_noise = R

    def step(self, control, measurement):
        #time update
        self.mean = self.state_update*self.mean + self.control_update*control
        self.covar = self.state_update*self.covar*self.state_update.getT() + self.covar_update_noise
        #measurement update
        gain = self.covar*self.measurement_predict.getT()*(self.measurement_predict*self.covar*self.measurement_predict.getT() + self.covar_measurement_noise).getI()
        self.mean = self.mean + gain*(measurement-self.measurement_predict*self.mean)
        self.covar = (matrix(identity(self.mean.shape[0]))-gain*self.measurement_predict)*self.covar

