from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import sys
import re

TT_SPLIT = .2

def readPC(path, encoding='iso-8859-1', MIN_LEN=0, MAX_LEN=100):
    print(f'MIN_LEN: {MIN_LEN}; MAX_LEN: {MAX_LEN}')
    with open(path, encoding=encoding) as f:
        raw = [x.lstrip()[:-1].split(' ') for x in f]
        raw = [[int(x[0]), ' '.join(x[1:])] for x in raw ]
        raw = [x for x in raw if x[1] and len(x[1]) <= MAX_LEN and len(x[1]) >= MIN_LEN]
        F = np.array( [x[0] for x in raw] )
        X = [x[1] for x in raw]

    return X, F

try:
    path = sys.argv[1]
    MAX_LEN = int(sys.argv[2])
    out_path = sys.argv[3]
except:
    print("USAGE: PASSWD MAX_LEN OUT")
    sys.exit(1)

raw, F = readPC(path, MAX_LEN=MAX_LEN)
n = len(raw)

# map chars
chars = set()
for p in tqdm(raw):
    for c in p:
        chars.add(c)
        
print('CHAR_MAP size: %d' % len(chars))

index2char = ['\n'] + list(chars)
char2index = {c : i for i, c in enumerate(index2char)}

def getTextfromIndex(index):
    return ''.join(index2char[i] for i in index)

# convert password
_index = np.zeros((n, MAX_LEN), np.uint8)
for i, p in tqdm(list(enumerate(raw))):
    pad_p = p + ''.join(['\n'] * (MAX_LEN - len(p)))
    _index[i] = [char2index[c] for c in pad_p]
    
index = []
assert len(_index) == len(F)
for i in range(len(_index)):
    index += [_index[i]] * F[i]
index = np.array(index)
print( index.shape )
assert len(index) == int(F.sum())
    
# split train/test
train, test = train_test_split(index, test_size=TT_SPLIT)

train_byte = set(p.tobytes() for p in train)
test_byte = set(p.tobytes() for p in test)

test_clean_byte = test_byte - train_byte

out = index2char, (train, test_byte, test_clean_byte)
with open(out_path, 'wb') as f:
    pickle.dump(out, f)

