import math
import tqdm
import os
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import myPickle
from itertools import count
import re
from .sample import *

SCORE_TYP = 0

def makeHolesString(s_):
    S = []
    for i in range(len(s_)):
        s = list(s_)
        s[i] = EMPTY
        S.append(''.join(s) )
    return S

def makeHoles(xi, l):
    for i in range(l):
        xi[i, i] = -1
    return xi

getnettemps =  lambda p, t: np.where(p.argsort()[::-1]==t)

class SampleMeter(Sample):
    
    def score(self, p, s):
        l = len(s)
        c = np.zeros(l)
        A = []
        for j in range(l):
            if not s[j] in self.cm:
                t = 0
            else:
                t = self.cm[s[j]]

            nattempt = getnettemps(p[j][j], t)
            A.append(nattempt)
            
            c[j] = (p[j][j][t])

            #if SCORE_TYP == 0:
                #c[j] = (p[j][j][t])
            #elif SCORE_TYP == 1:
            #    c[j] = (p[j][j][t]) / p[j][j].max()
                
        A = np.concatenate(A).ravel() + 1
        return c, A

    def meterSingle(self, s, sess=None):
        l = len(s)
        
        S = makeHolesString(s)
        
        xi = np.concatenate([self.parseString(s) for s in S])

        if sess is None:
            sess = tf.Session(config=self.config)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
        x_, p_, prediction_string_ = sess.run([self.infx, self.infp, self.infprediction_string], {self.x4PP:xi})
            
        c = self.score(p_, s)
            
        prediction_string_ = [''.join(ss.decode().split('\n')) for ss in prediction_string_]
        
        return prediction_string_, c, p_, sess
    
    
    def _meterBatch(self, H, X, sess=None):
        n = len(X)
        
        do = lambda sess, H: sess.run([self.infx, self.infp], {self.x4PP:H})
        
        if sess:
            _, p_ = do(sess, H)
        else:
            print("Creating tf.Session")
            with tf.Session(config=self.config) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                _, p_ = do(sess, H)
            
        SCOREs = [None] * n
        tot = 0
        for i in range(n):
            l = len(X[i])
            pi = p_[tot:tot+l]
            SCOREs[i] = self.score(pi, X[i])
            tot += l
        return SCOREs
        
        
    def meterBatched(self, _X):
        H = []
        X = []
        SCORE = []
        
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
        
            for i, x_ in tqdm.tqdm( list( enumerate(_X) ) ):
                #if len(x_) < self.x_len:
                #    x_ = x_ + '\n'
                l = len(x_)
                X.append(x_)
                xi = self.parseString(x_)
                xi = np.tile(xi, (l,1))
                xi = makeHoles(xi, l)
                H.append(xi)

                if i and i % self.batch_size == 0:
                    H = np.concatenate(H)
                    #print(f"{i}-BATCH-{len(H)}")
                    SCORE += self._meterBatch(H, X, sess)
                    H = []
                    X = []
                    
            if H:
                H = np.concatenate(H)
                SCORE += self._meterBatch(H, X, sess)

        return SCORE
        
        


