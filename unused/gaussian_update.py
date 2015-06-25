from numpy import mean, std, matrix, append
from scipy import stats
from scipy.stats import norm
from math import sqrt

class beliefScalar:
    def __init__(self, mu_0, sigma2_0, sigma2, data = []):
        self.mean_prior = float(mu_0)
        self.var_prior = float(sigma2_0)
        self.mean_post = self.mean_prior
        self.var_post = self.var_prior
        self.var_measurement = float(sigma2)
        self.data = []
        self.update(data)

    def update(self, data):
        if not isinstance(data, collections.Sequence):
            data = [data]
        if len(data)==0:
            return
        self.data.extend(data)
        #print self.data
        self.var_post = 1./(len(self.data)/self.var_measurement + 1/self.var_prior)
        #print self.var_post
        self.mean_post = self.var_post*(self.mean_prior/self.var_prior + len(self.data)*mean(data)/self.var_measurement)
        #print self.mean_post

    def predictCDF(self, val):
        return norm.cdf(val, self.mean_post, sqrt(self.var_post+self.var_measurement))

    def predictPPF(self, percent):
        norm.ppf(percent, self.mean_post, sqrt(self.var_post+self.var_measurement))

class beliefMultivariate:
    def __init__(self, dim, mu_0, sigma_0, sigma):
        self.mean_prior = mu_0
        self.covar_prior = sigma_0
        self.mean_post = self.mean_prior
        self.covar_post = self.covar_prior
        self.covar_measurement = sigma
        self.data = matrix([]).reshape((dim,0))

    def addData(self, data):
        self.data = append(self.data.getT(),data.getT(),0).getT()

    def update(self, data):
        if len(data.getT())==0 or len(data)==0:
            return
        self.addData(data)
        self.covar_post = (self.covar_prior.getI()+self.data.shape[1]*self.covar_measurement.getI()).getI()
        self.mean_post = self.covar_post*(self.data.shape[1]*self.covar_measurement.getI()*self.data.mean(1)+self.covar_prior.getI()*self.mean_prior)
