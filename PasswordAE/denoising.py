import tensorflow as tf
import numpy as np
import gin
import sys, os
import myPickle, os
import math
import h5py


def makeIterNoise(home, epochs, batch_size, MAX_LEN, buffer_size, include_end_char=False ,chunk_size=2**13, test=False):
    
    CMPATH = os.path.join(home, 'charmap.pickle')
    char_map = myPickle.load(CMPATH)
    char_num = len(char_map)
    
    print("include_end_char", include_end_char)
    
    XPATH = os.path.join(home, 'X.h5df') 
    
    if test:
        key = 'test' 
    else:
        key = 'train' 
    
    with h5py.File(XPATH, 'r') as f:
        f = f[key]
        N = len(f)
    
    def G(*args):
        with h5py.File(XPATH, 'r') as f:
            f = f[key]
            bn = math.ceil(N / chunk_size)
            for i in range(bn):
                s = i * chunk_size
                e = (i+1) * chunk_size
                Xchunk = f[s:e]
                for x in Xchunk:
                    x_ = x.copy()
                    x_ = applyNoise(x_, include_end_char=include_end_char)
                    yield x_, x
    
    def batch():
        dataset = tf.data.Dataset.from_generator(G, (tf.int32, tf.int32), (MAX_LEN,MAX_LEN))
        if not test:
            dataset = dataset.repeat(epochs)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        iterator = dataset.make_one_shot_iterator()
        x_, x = iterator.get_next()
        return {'x':x_}, x
    
    return batch, char_num, N


def applyNoise(x, include_end_char, CHAR=-1):#char_num+1):
    l = (x >= 1).sum(0)
    if include_end_char and l != x.shape[-1]:
        s = np.random.randint(0, l+1)
    else:
        s = np.random.randint(0, l)
    x[s] = CHAR
    return x


#def applyNoise(x, p, char_num):
#    n = len(x)
#    m = np.random.binomial(1, p, n).astype(np.bool)
#    x[m] = char_num + 1
#    return x


###############################################

def applyMask(x, msize, NCHAR=-1):#char_num+1):
    l = (x >= 1).sum(0)
    l = l - msize     
    s = np.random.randint(0, l+1)
    x[s:s+msize] = NCHAR
    return x

def makeIterMask(home, mask_size, epochs, batch_size, MAX_LEN, buffer_size, chunk_size=2**13, test=False):
    
    CMPATH = os.path.join(home, 'charmap.pickle')
    char_map = myPickle.load(CMPATH)
    char_num = len(char_map)
    
    XPATH = os.path.join(home, 'X.h5df') 
    
    if test:
        key = 'test' 
    else:
        key = 'train' 
    
    with h5py.File(XPATH, 'r') as f:
        f = f[key]
        N = len(f)
    
    def G(*args):
        with h5py.File(XPATH, 'r') as f:
            f = f[key]
            bn = math.ceil(N / chunk_size)
            for i in range(bn):
                s = i * chunk_size
                e = (i+1) * chunk_size
                Xchunk = f[s:e]
                for x in Xchunk:
                    x_ = x.copy()
                    x_ = applyMask(x_, mask_size)
                    yield x_, x
    
    def batch():
        dataset = tf.data.Dataset.from_generator(G, (tf.int32, tf.int32), (MAX_LEN,MAX_LEN))
        if not test:
            dataset = dataset.repeat(epochs)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        iterator = dataset.make_one_shot_iterator()
        x_, x = iterator.get_next()
        return {'x':x_}, x
    
    return batch, char_num, N

###############################################


def makeIterMNoise(home, holes_number, epochs, batch_size, MAX_LEN, buffer_size, chunk_size=2**13, test=False):
    
    CMPATH = os.path.join(home, 'charmap.pickle')
    char_map = myPickle.load(CMPATH)
    char_num = len(char_map)
    
    XPATH = os.path.join(home, 'X.h5df') 
    
    if test:
        key = 'test' 
    else:
        key = 'train' 
    
    with h5py.File(XPATH, 'r') as f:
        f = f[key]
        N = len(f)
    
    def G(*args):
        with h5py.File(XPATH, 'r') as f:
            f = f[key]
            bn = math.ceil(N / chunk_size)
            for i in range(bn):
                s = i * chunk_size
                e = (i+1) * chunk_size
                Xchunk = f[s:e]
                for x in Xchunk:
                    x_ = x.copy()
                    x_ = applyMNoise(x_, holes_number)
                    yield x_, x
    
    def batch():
        dataset = tf.data.Dataset.from_generator(G, (tf.int32, tf.int32), (MAX_LEN,MAX_LEN))
        if not test:
            dataset = dataset.repeat(epochs)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        iterator = dataset.make_one_shot_iterator()
        x_, x = iterator.get_next()
        return {'x':x_}, x
    
    return batch, char_num, N


def applyMNoise(x, holes_number, CHAR=-1):
    l = (x >= 1).sum(0)
    n = x.shape[-1]
    p = holes_number / l
    m = np.random.binomial(1, p, n).astype(np.bool)
    m[l:] = 0
    x[m] = CHAR
    return x
