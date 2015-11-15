from numpy import concatenate, matrix, zeros


def add_noise(objects, noise):
    counter = 0
    ret = []
    for o in objects:
        dim = o.dim
        ret.append(o.__class__.exp(noise[counter:counter+dim,0], o.getOrigin()))
        counter += dim
    return ret

def append_matrices(mats):
    dim = 0
    for m in mats:
        dim += m.shape[1]
    ret = matrix([[]]).reshape((0,dim))
    counter = 0
    for m in mats:
        ret = matrix(concatenate((ret,concatenate((zeros((m.shape[0],counter)),m,zeros((m.shape[0],dim-counter-m.shape[1]))), axis=1))))
        counter += m.shape[1]
    return ret
