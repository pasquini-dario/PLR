import pickle

def dump(filename, data, **kargs):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load(filename, python2=False, **kargs):
    if python2:
        kargs.update(encoding='latin1')
    with open(filename, 'rb') as f:
        data = pickle.load(f, **kargs)
    return data
