import tempfile
from unittest import TestCase

import peloton_bloomfilters


class TestDivideByMultiple(TestCase):

    def assert_divides(self, D):
        multiplier, pre_shift, post_shift, increment = peloton_bloomfilters._compute_unsigned_magic_info(D, 64)
        n = 1
        while n < D**3:
            n *= 1.41
            N = int(n)
            N += increment
            N = N >> pre_shift
            if multiplier != 1:
                N *= multiplier
                N = N >> 64
                N = N % (2 ** 64)
            N = N >> post_shift
            self.assertEquals(N, n // D)

    def test(self):
        for x in range(1, 1000):
            print(x)
            self.assert_divides(x)


class BloomFilterCase(object):
    def test_add(self):
        self.assertEqual(0, len(self.bloomfilter))
        self.assertNotIn("5", self.bloomfilter)
        self.assertFalse(self.bloomfilter.add("5"))
        self.assertEqual(1, len(self.bloomfilter))
        self.assertIn("5", self.bloomfilter)

    def test_capacity(self):
        for i in range(50):
            self.assertFalse(self.bloomfilter.add(i))
        for i in range(50):
            self.assertIn(i, self.bloomfilter)
        self.assertTrue(self.bloomfilter.add(50))
        for i in range(50):
            self.assertNotIn(i, self.bloomfilter)
        self.assertIn(50, self.bloomfilter)


class TestBloomFilter(TestCase, BloomFilterCase):
    def setUp(self):
        self.bloomfilter = peloton_bloomfilters.BloomFilter(50, 0.001)

class TestThreadSafeBloomFilter(TestCase, BloomFilterCase):
    def setUp(self):
        self.bloomfilter = peloton_bloomfilters.ThreadSafeBloomFilter(50, 0.001)


class TestSharedMemoryBloomFilter(TestCase, BloomFilterCase):
    def setUp(self):
        self.fd = tempfile.NamedTemporaryFile()
        self.bloomfilter = peloton_bloomfilters.SharedMemoryBloomFilter(self.fd.name, 50, 0.001)

    def tearDown(self):
        self.fd.close()

    def test_sharing(self):
        print("Test started")
        bf1 = self.bloomfilter
        bf2 = peloton_bloomfilters.SharedMemoryBloomFilter(self.fd.name, 50, 0.001)
        self.assertEquals(len(bf2), 0)
        self.assertNotIn(1, bf1)
        self.assertNotIn(1, bf2)

        bf1.add(1)

        self.assertIn(1, bf1)
        self.assertIn(1, bf2)

        bf2.add(2)
        self.assertIn(2, bf1)
        self.assertIn(2, bf2)


    def test_capacity_in_sync(self):
        bf1 = self.bloomfilter
        bf2 = peloton_bloomfilters.SharedMemoryBloomFilter(self.fd.name, 50, 0.001)
        bfs = [bf1, bf2]
        for i in range(50):
            bfs[i % 2].add(i)
        for i in range(50):
            self.assertIn(i, bf1)
            self.assertIn(i, bf2)
        self.assertTrue(bf2.add(50))
        for i in range(50):
            self.assertNotIn(i, bf1)
            self.assertNotIn(i, bf2)

        self.assertIn(50, bf1)
        self.assertIn(50, bf2)

