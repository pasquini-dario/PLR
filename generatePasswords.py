"""
USAGE: NBATCH BATCHSIZE OUTPUTFILE
It prints on OUTPUTFILE NBATCH*BATCHSIZE passwords sampled from the prior latent distribution (i.e., N(0,I))
It requires: tensorflow, tensorflow_hub, numpy, tqdm
"""

import sys
import tensorflow as tf # only 1.14.0 tested
import tensorflow_hub as hub
import numpy as np
from util import *
from tqdm import trange

LATENT_SIZE = 64
MHUB_SIGNATURE = 'latent_to_data'
MODEL_PATH = './DATA/TFHUB_models/BNK_RESNET_PassGAN_1/'
CHARMAP_PATH = './DATA/MISC/char_map.pickle'
ENCODING = 'UTF-8'

def loadPassGAN():
    passgan = hub.Module(MODEL_PATH)
    return passgan

if __name__ == '__main__':
    try:
        NBATCH = int(sys.argv[1])
        BATCHSIZE = int(sys.argv[2])
        OUTPUTFILE = sys.argv[3]
    except:
        print("USAGE: NBATCH BATCHSIZE OUTPUTFILE")
        sys.exit(1)
        
        
    passgan = loadPassGAN()
    z = tf.random_normal(shape=(BATCHSIZE, LATENT_SIZE))
    x = passgan(z, signature=MHUB_SIGNATURE)
    
    print("Let's crack...")
    
    with open(OUTPUTFILE, 'w',  encoding=ENCODING) as f:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            
            for i in trange(NBATCH):
                _pwd = sess.run(x)
                pwd = clean(_pwd)
                
                print(*pwd, file=f, sep='\n')
        
        