EMPTY = '\t'
MAX_LEN = 16

import numpy as np
import myPickle, os
import tensorflow as tf
import tensorflow_hub as hub


def load_charmap(path):
    cm = myPickle.load(path)
    cm_ = [x[0] for x in sorted(cm.items(), key=lambda x: x[1])]
    return cm, cm_

def parseString(C, cm):
    I = np.zeros(MAX_LEN, np.int32)
    for i in range(len(C)):
        c = C[i]
        if c == '\t':
            I[i] = -1
        else:
            I[i] = cm[C[i]]
    return I[None, :]


def filterout_invalid(t, x, ERR_LIM=0):
    if len(t) != len(x):
        return False
    n = min(len(t), len(x))
    x = list(x)
    if len(t) > len(x):
        x += [''] * (len(t) - len(x))
    err = 0
    for i in range(len(t)):
        if t[i] != EMPTY:
            if x[i] != t[i]:
                err += 1
            if err > ERR_LIM:
                return False
            x[i] = t[i]
    return ''.join(x)

def curate_sample(X, s):
    X = [filterout_invalid(s, x) for x in X]
    # get uniques and sort
    c = {}
    for x in X:
        if not x in c:
            c[x] = 0
        c[x] += 1
    c = list(c.items())
    c = sorted(c, key=lambda x:-x[1])
    return [v[0] for v in c if v[0]]


def infer(s, xph, prediction, cm, cm_):
    i = parseString(s, cm)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        prediction_ = sess.run( prediction, {xph:i})
        toS = lambda P: ''.join([cm_[p] for p in P if p > 0]) 
        prediction_ = [toS(p) for p in prediction_]
    return prediction_

def setup_model(path, n, stddev):
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    module = hub.Module(path)
    xph = tf.placeholder(tf.int32, shape=(None, MAX_LEN))
    
    inputs = {
    'x' : xph,
    'stddev' : (stddev,),
    'n' : (n,)
    }
    out = module(inputs, as_dict=True, signature='sample_from_latent')
    prediction = out['prediction']
    return xph, prediction