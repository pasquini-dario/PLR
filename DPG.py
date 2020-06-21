import sys
import random
import numpy as np
from peloton_bloomfilters import BloomFilter
from numpy_ringbuffer import RingBuffer

class DPG:
    memory_bank_size = 10 ** 6
    def __init__(self, latent_size, stddv_p, stddv, hot_start, N,  batch_size, accuracy_bloomf=0.01, static=False):
        self.i = 0
        self.I = 0
        self.BI = 1
        
        #self.guessed_z = []
        
        self.DYNAMIC = False
        self.hot_start = hot_start
        self.latent_size = latent_size
        self.stddv_p = stddv_p
        self.stddv = stddv
        self.LOG = False
        self.STATIC = static
        self.init_att_size = 0
        self.N = N
        
        self.batch_size = batch_size
        self.accuracy_bloomf = accuracy_bloomf
        
        self.matched_i = 0
        
        self.P = []
        
        if not self.STATIC:
            self.guessed_z = RingBuffer(capacity=self.memory_bank_size, dtype=(np.float32, self.latent_size))
        

    def enable_logging(self):
        print("LOGGING!")
        self.LOG = True
        self.log_unique = []
        self.log_guessed = []
        
    def __call__(self, z, x, attacked_set):
        new = None
        self.I += 1
        
        if not self.init_att_size:
            self.guesses = BloomFilter(self.N, self.accuracy_bloomf)
            self.init_att_size = len(attacked_set)
        
        #if True:
        if not x in self.guesses:
            self.guesses.add(x)

            self.i += 1
            self.FLAG = True
            new = x
            
            if self.LOG:
                if not self.I % self.log_fq:
                    self.log_unique += [(self.I, self.i, self.DYNAMIC)]
            
            # Matched
            if x in attacked_set:
                self.matched_i += 1
                attacked_set.remove(x)
                if not self.STATIC:
                    self.guessed_z.append(z)
                self.P += [self.BI]
                
                if self.LOG:
                    m = self.matched_i / self.init_att_size
                    self.log_guessed += [(self.I, self.i, x, m, self.DYNAMIC)]

                if  not self.STATIC and self.matched_i / self.init_att_size > self.hot_start and not self.DYNAMIC:
                    print("DYNAMIC starts now ....")
                    self.DYNAMIC = True
                    
        return new
    
    
    def guess(self, z_ph, G, sess):
        if self.DYNAMIC and len(self.guessed_z):
            idxs = np.random.randint(0, len(self.guessed_z), self.batch_size, np.int32)
            gi = self.guessed_z[idxs] 
            z = np.random.normal(gi, self.stddv)
        else:
            z = np.random.normal(0, scale=self.stddv_p, size=(self.batch_size, self.latent_size))
        x = sess.run(G, {z_ph:z})
        return z, x