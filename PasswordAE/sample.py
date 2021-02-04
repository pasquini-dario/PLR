import math
import tqdm
import os
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import myPickle
from itertools import count
import re


EMPTY = '\t'

def getFromTxt(X, r):
    r = re.compile(r)
    O = set()
    for x in X:
        if r.fullmatch(x):
            O.add(x)
    return O


class Sample:
    
    
    def parseString(self, C):
        I = np.zeros(self.x_len, np.int32)
        for i in range(len(C)):
            c = C[i]
            if c == EMPTY:
                I[i] = -1
            else:
                if not c in self.cm:
                    I[i] = 0
                else:
                    I[i] = self.cm[c]
        return I[None, :]
    
    def toS(self, P):
        return ''.join([self.cm_[p] for p in P if p > 0]) 
    
    def samplingLatent(self):
        return tf.random_normal(shape=(self.batch_size, self.ls))
            
    
    def __init__(self, mpath, dhome, ls, batch_size=4096, usegpu=-1):
        self.mpath = mpath
        self.dhome = dhome
        self.batch_size = batch_size
        self.ls = ls
        
        cm_ = os.path.join(dhome, 'char_map.pickle')
        self.cm = myPickle.load(cm_)
        self.cm_ = [x[0] for x in sorted(self.cm.items(), key=lambda x: x[1])]
        
        tf.logging.set_verbosity(tf.logging.ERROR)
        module = hub.Module(mpath)
        self.module = module
        
        if True:
            # pure sampling from latent
            z = self.samplingLatent()
            o = module(z, signature='latent', as_dict=True)
            self.latent2data = o['prediction_string']
            p = o['prediction']
            self.x_len = p.shape.as_list()[-1]
        else:
            self.x_len = 16
            print("NO LATENT")
        
        try:
            # proximity sampling \eg for SSPG
            self.x4PP = tf.placeholder(tf.int32, shape=(None, self.x_len))
            self.n4PP = tf.placeholder(tf.int32, shape=1)
            self.stddev4PP = tf.placeholder(tf.float32, shape=(1,))
            inputs = {'x':self.x4PP, 'n':self.n4PP, 'stddev':self.stddev4PP}
            self.latent2data4PP = module(inputs, as_dict=True, signature='sample_from_latent')['prediction_string']
        except:
            ...
        
        # simple inference
        out = module(self.x4PP, as_dict=True)
        self.infp = out['p']
        self.infx = out['x']
        self.infprediction_string = out['prediction_string']
        
        ###
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        
        if usegpu != -1:
            print("USE_GPU:", str(usegpu))
            self.config.gpu_options.visible_device_list = str(usegpu)
   
        
        
    def sample(self, n):
        nb = math.ceil(n / self.batch_size)
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            I = 0
            for i in tqdm.trange(nb):
                xi = sess.run(self.latent2data)
                
                for j, x in enumerate(xi):
                    I += 1
                    
                    if I > n:
                        break
                    x = x.decode().split('\n')[0]
                    yield x

                    
    def proximitySample(self, xps, n, stddev):
        
        xp = self.parseString(xps)
        
        nb = math.ceil(n / self.batch_size)
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            I = 0
            for i in tqdm.trange(nb):
                xi = sess.run(self.latent2data4PP, {self.x4PP:xp, self.n4PP:(self.batch_size,), self.stddev4PP:(stddev,)})
                
                for j, x in enumerate(xi):
                    I += 1
                    if I > n:
                        break
                    x = x.decode().split('\n')[0]
                    yield x

    @staticmethod
    def sub(t, x, ERR_LIM):
        if len(x) != len(t):
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
    
    MAX_ITER = 5*(10**9)
                    
    def proximitySampleClean(self, xps, n, stddev, errc, MAX_ITER=MAX_ITER):
        UN = set()
        #for x in self.proximitySample(xps, n, stddev):
        #    xc = self.sub(xps, x, errc)
        #    if xc and not xc in UN:
        #        UN.add(xc)
        #        yield xc
        xp = self.parseString(xps)
        nb = math.ceil(n / self.batch_size)
        
        I = 0
        END = False
        i = 0
        
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            while not END:
                xi = sess.run(self.latent2data4PP, {self.x4PP:xp, self.n4PP:(self.batch_size,), self.stddev4PP:(stddev,)})
                
                for j, x in enumerate(xi):
                    i += 1
                    x = x.decode().split('\n')[0]
                    xc = self.sub(xps, x, errc)
                    if xc and not xc in UN:
                        UN.add(xc)
                        I += 1
                        if I > n or i > MAX_ITER:
                            END = True
                            if i > MAX_ITER:
                                print("\n\nBAD!\n\n")
                            break
                        #print(j, i, xc)
                        yield xc, i
