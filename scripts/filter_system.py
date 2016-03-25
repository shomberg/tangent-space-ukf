from unscented_kalman_filter_objects import UnscentedKalmanFilter
from numpy import matrix

class FilterSystem():
    def __init__(self):
        self.objects = {}
        self.observations = {}
        self.k = UnscentedKalmanFilter([], [], [], [], [], matrix([[]]).reshape((0,0)), matrix([[]]).reshape((0,0)),True)

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
