from __future__ import print_function

import tempfile
import time
import peloton_bloomfilters

try:
    range = xrange
except NameError:
    pass

NS = 10**9
for _p in range(1,3):
    p = 10 ** _p
    for e in range(9):
        with tempfile.NamedTemporaryFile() as f:
            X = int(1000 * 10 ** (e / 2.0))
            print(X, p, end='')
            name = f.name
            bloomfilter = peloton_bloomfilters.SharedMemoryBloomFilter(name,  X+ 1, 1.0 / p)
            t = time.time()

            for x in range(X):
                bloomfilter.add(x)
            print((time.time() - t) / X * NS, end='')
            t = time.time()
            for x in range(X):
                x in bloomfilter
            print((time.time() - t) / X * NS, end='')
            t = time.time()
            for x in range(X, 2*X):
                x in bloomfilter
            print((time.time() - t ) / X * NS)
