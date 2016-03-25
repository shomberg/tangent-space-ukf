from unscented_kalman_filter_objects import UnscentedKalmanFilter
import numpy as np
import pdb

class FilterSystem():
    def __init__(self):
        self.objects = {}
        self.observations = {}
        self.k = UnscentedKalmanFilter([], [], [], [], [], np.matrix([[]]).reshape((0,0)), np.matrix([[]]).reshape((0,0)),True)

    def addObject(self, name):
        self.objects[name] = FilterObject()

    def addContinuousDist(self, obj_name, att_name, mean, covar, update_func, of_objs, noise_covar):
        of_indices = []
        for (obj, att) in of_objs:
            of_indices.append(self.getObject(obj).getAttribute(att).index)
        index = len(self.k.mean)
        self.k.addObject(mean, covar, update_func, of_indices, noise_covar)
        self.getObject(obj_name).addAttribute(ContinuousDist(index), att_name)

    def addDiscreteDist(self, obj_name, att_name, dist):
        self.getObject(obj_name).addAttribute(DiscreteDist(dist), att_name)

    def addValue(self, obj_name, att_name, val):
        self.getObject(obj_name).addAtribute(Value(val), att_name)
        
    def getObject(self, name):
        return self.objects[name]

    def addObservation(self, name, predict_func, sym, of_objs, noise_covar):
        of_indices = []
        for (obj, att) in of_objs:
            of_indices.append(self.getObject(obj).getAttribute(att).index)
        index = len(self.k.measurement_predict_functions)
        self.k.addObservation(predict_func, sym, of_indices, noise_covar)
        self.observations[name] = FilterObservation(index)

    def getObservation(self, name):
        return self.observations[name]

    def stateUpdate(self, command=None):
        self.k.stateUpdate(command)

    def measurementUpdate(self, measurement, obs_list):
        indices = []
        for i in obs_list:
            if isinstance(i, list):
                add = []
                for obs in i:
                    add.append(self.getObservation(obs).index)
                indices.append(add)
            else:
                indices.append(self.getObservation(i).index)
        self.k.measurementUpdate(measurement, indices)

    def getMarginalSigmas(self, of_objs):
        of_indices = []
        for (obj, att) in of_objs:
            of_indices.append(self.getObject(obj).getAttribute(att).index)
        full = self.k.getMarginalDistribution()
        return [[point[i] for i in of_indices] for point in full[1]]

    def getMarginalTangentSpace(self, of_objs):
        of_indices = []
        for (obj, att) in of_objs:
            of_indices.append(self.getObject(obj).getAttribute(att).index)
        full = self.k.getMarginalDistributionTangent()
        cum_dims = [0]+np.cumsum(full[3])
        cov_index = []
        for i in of_indices:
            cov_index.extend(range(cum_dims[i],cum_dims[i+1]))
        return ([full[1][i] for i in of_indices], full[2][np.array(cov_index)][:,np.array(cov_index)])

    def getConditionalSigmas(self, of_objs, cond_objs, cond_vals):
        of_indices = []
        for (obj, att) in of_objs:
            of_indices.append(self.getObject(obj).getAttribute(att).index)
        cond_indices = []
        for (obj, att) in cond_objs:
            cond_indices.append(self.getObject(obj).getAttribute(att).index)
        full = self.k.getConditionalDistribution(cond_indices,cond_vals)
        return [[point[full[0].index(i)] for i in of_indices] for point in full[1]]
        
    def getConditionalTangentSpace(self, of_objs, cond_objs, cond_vals):
        pdb.set_trace()
        of_indices = []
        for (obj, att) in of_objs:
            of_indices.append(self.getObject(obj).getAttribute(att).index)
        cond_indices = []
        for (obj, att) in cond_objs:
            cond_indices.append(self.getObject(obj).getAttribute(att).index)
        full = self.k.getConditionalDistributionTangent(cond_indices,cond_vals)
        cum_dims = np.append([0],np.cumsum(full[3]))
        cov_index = []
        for i in of_indices:
            cov_index.extend(range(cum_dims[full[0].index(i)],cum_dims[full[0].index(i)+1]))
        return ([full[1][full[0].index(i)] for i in of_indices], full[2][np.array(cov_index)][:,np.array(cov_index)])

        

class FilterObservation():
    def __init__(self, index):
        self.index = index

class FilterObject():
    def __init__(self):
        self.attributes = {}

    def addAttribute(self, att, name):
        self.attributes[name] = att

    def getAttribute(self, name):
        return self.attributes[name]

class Observation():
    def __init__(self, atts, func):
        self.predict_function = func
        self.of_attributes = atts

class Attribute():
    def __init__(self):
        pass

    def getValue(self):
        pass

class Value(Attribute):
    def __init__(self, val):
        self.value = val

class DiscreteDist(Attribute):
    def __init__(self, dist):
        self.dist = dist

class ContinuousDist(Attribute):
    def __init__(self, index):
        self.index = index
