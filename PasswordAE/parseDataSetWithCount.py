import numpy as np
import sys
import string
from tqdm import tqdm, trange
import myPickle
import random
import os
import h5py

def printP(path, X, encoding='iso-8859-1'):
    with open(path, 'w', encoding=encoding) as f:
        for x in X:
            print(x, file=f)

def readPC_skip_encoding(path, encoding='utf-8', MIN_LEN=0, MAX_LEN=100):
    with open(path, encoding=encoding, errors='ignore') as f:
        raw = [x.lstrip()[:-1].split(' ') for x in f]
        raw = [[int(x[0]), ' '.join(x[1:])] for x in raw ]
        raw = [x for x in raw if x[1] and len(x[1]) <= MAX_LEN and len(x[1]) >= MIN_LEN]
        F = np.array( [x[0] for x in raw] )
        X = [x[1] for x in raw]

    return X, F

def rankp(f):
    f_ = {x:i for i, x in enumerate(sorted( set(f), reverse=True))}
    rank = [f_[x] for x in f]
    return rank

def string2index(s, MAX_LEN, char_map):
    idx = np.zeros(MAX_LEN, np.uint8)
    for i, c in enumerate(s):
        idx[i] = char_map[c]
    return idx[None,:]

if __name__ == '__main__':
    try:
        Xpath = sys.argv[1]
        MAX_LEN = int(sys.argv[2])
        MIN_LEN = int(sys.argv[3])
        ENCODING = sys.argv[4]
        HOME = sys.argv[5]
        TESTSIZE = float(sys.argv[6])
        MERGE_SPECIAL = 0
    except:
        print("USAGE: DATASET MAX_LEN MIN_LEN ENCODING OUTPUT_HOME %TESTSIZE")
        sys.exit(1)
        
    if MERGE_SPECIAL:
        print("\n\n\t MERGE SPECIAL!!!!!\n")
        
        
    # READ FROM WITH-COUNT FILE AND WRITE TXT ON FILE
    X, F = readPC_skip_encoding(Xpath, encoding=ENCODING, MAX_LEN=MAX_LEN, MIN_LEN=MIN_LEN)
    rank = rankp(F)
    TEST_OUT = os.path.join(HOME, 'X.txt')
    printP(TEST_OUT, X)
    #####################################
    
    print("NUMBER OF UNIQUE PASSWORDS ", len(X))
    
    # CREATE CHARS MAP
    chars = dict()
    for p in tqdm(X):
        for c in p:
            if not c in chars:
                chars[c] = 0
            chars[c] += 1
    #####################################
    
    print("NUMBER OF CHARS ", len(chars))
    
    # MERGE STRANGE CHARACTERS
    if MERGE_SPECIAL:
        NORMAL = list( string.ascii_letters + string.digits + string.punctuation ) + [' ']
        SPECIAL = chars.copy()

        for c in NORMAL:
            if c in chars:
                SPECIAL.pop(c)
        SPECIAL = list(SPECIAL.keys())

        TOT = { c:i for i, c in enumerate(['\n'] + NORMAL) }
        TOT.update( {c:len(TOT) for c in SPECIAL} )
    else:
        TOT = { c:i for i, c in enumerate(['\n'] + list(chars.keys())) }
    #####################################
    
    # WRITE CHARS MAP ON FILE 
    CM_OUT = os.path.join(HOME, 'charmap.pickle')
    myPickle.dump(CM_OUT, TOT)
    #####################################
    
    
    # CONVERT PASSWORDS IN INDEX
    X = [string2index(x, MAX_LEN, TOT) for x in X]
    #####################################
    
    # CREATE WRITE TEST IN FILE
    TEST_OUT = os.path.join(HOME, 'rfX')
    Xtest = np.array(X)
    Xtest = np.squeeze(Xtest)
    rank = np.array(rank)[:, None]
    rfX = np.concatenate((rank, F[:, None], Xtest), 1) 
    np.save(TEST_OUT, rfX)
    #####################################
    
    # EXPLOD AND WRITE ON FILE (h5df) PASSWORDS BASED ON FEQUENCY
    Xe = []
    n = len(X)
    for i in trange(n):
        Xe += [X[i]] * F[i]
    
    print("SHUFFLING....")
    random.shuffle(Xe)
    
    Xe = np.array(Xe)
    Xe = np.squeeze(Xe)
    
    n = len(Xe)
    train_size = int(n * (1-TESTSIZE))
    
    Xe_train = Xe[:train_size]
    Xe_test = Xe[train_size:]
    
    print(Xe_train.shape, ' ', Xe_test.shape)
    
    X_OUT = os.path.join(HOME, 'X.h5df')
    
    with h5py.File(X_OUT, 'w') as f:
        f.create_dataset('train', data=Xe_train)
        f.create_dataset('test', data=Xe_test)
    #####################################
