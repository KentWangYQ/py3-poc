import unittest
from aiostream import stream


class AioStreamTest(unittest.TestCase):
    async def _source(self, n):
        for i in range(n):
            yield i

    def test_enumerate(self):
        for i in stream.list(self._source(10)):
            for j in i:
                print(j)
