# Bloomin fast Bloomfilters from Peloton 

`peloton_bloomfilter.SharedMemoryBloomfilter` is the easiest to use,
fastest bloomfilter implementation for cPython.


##  Usage

Bloomfilters are probabilistic data structures supporting basic object
membership testing.  Objects are added to bloomfilters with the `add`
method and tested for membership with the `in` operator.  A
bloomfilter reporting `False` to an `in` query is guaranteed to not
have the object; a bloomfilter reporting `True` likely has the tested
object subject to a tunable false positive rate.

`peloton_bloomfilters` implements three bloom-filter classes; `BloomFilter` is a plain old bloomfilterfor single threads or gevent apps; `ThreadSafeBloomFilter` releases the GIL and uses `__atomic_or_fetch` to prevent lost bits during writes and `SharedMemoryBloomfilter` supports the creation of bloomfilters that are shared between processes in real time using files and `mmap`

To create a bloomfilter object you merely import the module and call
it with two or three parameters: the file name to hold the shared memory mmap
object, the capacity of the bloomfilter and its false positive rate.


```
>>> from peloton_bloomfilter import *
>>> bf = BloomFilter(1000, 0.001)
>>> tsbf = ThreadSafeBloomFilter(1000, 0.001)
>>> smbf = SharedMemoryBloomfilter("/tmp/filter", 1000, 0.001)
```

Adding and testing membership against a bloomfilter works exactly like
a set, except a bloomfilter cannot be enumerated.

```
>>> smbf.add(1)
False
>>> 1 in smbf
True
>>> 2 in smbf
False
```

Note that `add` returns False.  `SharedMemoryBloomfilter` has a
limited capacity; before each add the remaining capacity is tested an
if its insufficient the bloom-filer will be cleared prior to
performing the add and `True` is returned

`len()` reports the number of items stored in the bloomfilter since
created or last cleared.

```
>>> len(smbf)
1
```

bloomfilters may be explicitly cleared. 

```
>>> smbf.clear()
>>> 1 in smbf
False
>>> len(smbf)
0
```


## Performance

`peloton_bloomfilter.SharedMemoryBloomfilter` is the fastest cPython
bloomfilter implementation known to its authors.  How fast?  Here we
benchmark peloton_bloomfilter against pybloomfiltermmap-0.3.14 in
their ability to add and test membership for member and non-member
objects against a 1,000,000 capacity bloomfilter for varying false
error rates.

Both libraries were compiled with gcc-4.8.5, CFLAGS="-mtune=native
-march=native" on Unbuntu 14.04 running on a Dell XPS 13 with 16Gb Ram
and a dual core Intel(R) Core(TM) i7-6560U CPU @ 2.20GHz and `cpupower
frequency-set -g performance`.  Times are in nanoseconds.


1,000,000 adds

```
1/p     peloton   pybloommap
10        139       276
100       184       394 
10000     431       459
1000000   693       757
```

1,000,000 membership tests, existing items

```
1/p     peloton   pybloommap
10        87        224
100       114       352
10000     307       424
1000000   523       671
```

1,000,000 membership tests, absent items

```
1/p     peloton   pybloommap
10        81        209
100       82        222
10000     102       159
1000000   119       179 
```





