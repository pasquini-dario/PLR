"""
Proof of concept for Dynamic Password Guessing (DPG).
"""

import sys
import gin
import pickle
import random
import os
import numpy as np
import itertools
import math
from tqdm import tqdm
import tensorflow as tf # only 1.14.0 tested
import tensorflow_hub as hub

import util

def generate(Z, X, batch_size, latent_size, zma, DYNAMIC, stddv):
    if DYNAMIC and len(zma):
        gi = random.choices(zma, k=batch_size)
        z = np.random.normal(gi, stddv)
    else:
        z = np.random.normal(0, scale=1.0, size=(batch_size, latent_size))
    x = sess.run(X, {Z:z})
    return z, x

def setupG4DPG(module_path, batch_size, latent_size, signature='latent_to_data'):
    G = hub.Module(module_path)
    z = tf.placeholder(shape=(batch_size, latent_size), dtype=tf.float32)
    x = G(z, signature=signature)
    return z, x
    

if __name__ == '__main__':
    
    try:
        gin_conf = sys.argv[1]
        test_path = sys.argv[2]
        n = int(sys.argv[3])   
        output_path = sys.argv[4]
    except:
        print("USAGE: CONF TEST-SET #GUESSES OUTPUTFILE")
        sys.exit(1)

    @gin.configurable
    def setup(**conf):
        return conf

    gin.parse_config_file(gin_conf)
    conf = setup()
    module_path = conf['G_TFHUB']
    latent_size = conf['LATENT_SIZE']
    batch_size = conf['BATCHSIZE']
    hot_start = conf['HOT_START']
    stddv = conf['STDDV']
    ENCODING = conf['ENCODING']
    MAX_LEN = conf['MAX_LEN']
    # END SETUP
    
    print("READING TEST ...")
    tc = util.readP(test_path, encoding=ENCODING, n=MAX_LEN)
    tcn = len(tc)
    
    hot_start = int( tcn * hot_start )
    print('HOT_START->', hot_start, '/',  tcn)

    print("LOADING GENERATOR ...")
    Z, X = setupG4DPG(module_path, batch_size, latent_size, signature='latent_to_data')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    
    print("PASSWORDS GENERATION ...")
    i = 0
    UN = set()
    zma = []
    tcmn = 0

    FLAG = False
    DYNAMIC_ = False
    EXIT = False
    
    fout = open(output_path, 'w', encoding=ENCODING)
    
    pbar = tqdm(total=n)
    while not EXIT:

        zbatch, batch = generate(Z, X, batch_size, latent_size, zma, DYNAMIC_, stddv)
        
        batch = util.clean(batch)

        for z, p in zip(zbatch, batch):

            if not p in UN:
                print(p, file=fout)
                UN.add(p)
                i += 1
                FLAG = True
                pbar.update(1)

            if p in tc:
                tcmn += 1
                tc.remove(p)
                zma.append(z)

                if len(zma) > hot_start and not DYNAMIC_:
                    print("DYNAMIC starts now ....")
                    DYNAMIC_ = True
                    
            if i >= n:
                EXIT = True
                break
    
    score = tcmn / tcn
    print("GUESSED PASSWORDS: %f" % score)
        
    pbar.close()
    fout.close()
    sess.close()