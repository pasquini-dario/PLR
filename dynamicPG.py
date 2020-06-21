"""
Proof of concept for Dynamic Password Guessing (DPG).
"""

LOG_FQ = 10 ** 6

import sys
import gin
import os
import numpy as np
import math
from tqdm import tqdm
import tensorflow as tf # only 1.14.0 tested
import tensorflow_hub as hub
import util

from DPG import *


def setupG4DPG(module_path, batch_size, latent_size, signature='latent_to_data'):
    G = hub.Module(module_path)
    z = tf.placeholder(shape=(batch_size, latent_size), dtype=tf.float32)
    x = G(z, signature=signature)
    return z, x


@gin.configurable
def setup(**conf):
    return conf
    
if __name__ == '__main__':
    
    try:
        gin_conf = sys.argv[1]
        test_path = sys.argv[2]
        n = int(sys.argv[3])
        output_file = sys.argv[4]
    except:
        print("USAGE: CONF TEST-SET #GUESSES OUTPUTFILE")
        sys.exit(1)

    gin.parse_config_file(gin_conf)
    conf = setup()
    module_path = conf['G_TFHUB']
    latent_size = conf['LATENT_SIZE']
    batch_size = conf['BATCHSIZE']
    ENCODING = conf['ENCODING']
    MAX_LEN = conf['MAX_LEN']
    stddev_p = conf['STDDV_P']
    hot_start = conf['HOT_START']
    stddv = conf['STDDV']
    STATIC = conf['STATIC']
    accuracy_bloomf =  conf['accuracy_bloomf']
    # END SETUP
    
    if STATIC:
        print("STATIC attack")
    else:
        print("DYNAMIC attack")
        
    
    state = DPG(latent_size, stddev_p,stddv, hot_start, n, batch_size, accuracy_bloomf, static=STATIC)
   
    outfile = open(output_file, 'w', encoding=ENCODING)

    print("READING TEST ...")
    attacked_set = set(util.readP_skip_encoding(test_path, encoding=ENCODING, MAX_LEN=MAX_LEN))
    attacked_set_n = len(attacked_set)
    
    hot_start = int( attacked_set_n * hot_start )
    print('HOT_START->', hot_start, '/',  attacked_set_n)

    print("LOADING GENERATOR ...")
    tf.logging.set_verbosity(tf.logging.ERROR)
    z_ph, G = setupG4DPG(module_path, batch_size, latent_size, signature='latent_to_data')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    
    print("PASSWORDS GENERATION ...")
    
    pbar = tqdm(total=n)
    while state.i < n and len(attacked_set):

        zbatch, batch = state.guess(z_ph, G, sess)
        batch = util.clean(batch)

        for z, x in zip(zbatch, batch):
            new = state(z, x, attacked_set)
            
            if not new is None:
                logged = False
                pbar.update(1)
                print(x, file=outfile)      
                
            if state.i % LOG_FQ == 0 and not logged:
                logged = True
                m = state.matched_i / state.init_att_size
                print("Dynamic: %s, Matched: %s" % (state.DYNAMIC, m))
                                
    
    pbar.close()
    sess.close()
    outfile.close()
