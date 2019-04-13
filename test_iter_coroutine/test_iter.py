# -*- encoding: utf-8 -*-

import unittest
import itertools


class IterTest(unittest.TestCase):
    class Node:
        def __init__(self, value):
            self._value = value
            self.children = []

        def __repr__(self):
            return 'Node(%s)' % self._value

        def __iter__(self):
            return iter(self.children)

        def add_child(self, c):
            self.children.append(c)

        def depth_first(self):
            yield self
            for c in self:
                yield from c.depth_first()

    def test_node(self):
        root = IterTest.Node(0)
        child1 = IterTest.Node(1)
        child2 = IterTest.Node(2)

        root.add_child(child1)
        root.add_child(child2)

        for c in root:
            print(c)

    @staticmethod
    def frange(start, stop, increase):
        f = start
        while f <= stop:
            yield f
            f += increase

    def test_frange(self):
        for f in IterTest.frange(1, 5, 0.5):
            print(f)

    def test_node_depth_first(self):
        root = IterTest.Node(0)
        child1 = IterTest.Node(1)
        child2 = IterTest.Node(2)
        child3 = IterTest.Node(3)
        child4 = IterTest.Node(4)
        child5 = IterTest.Node(5)
        child6 = IterTest.Node(6)
        child7 = IterTest.Node(7)

        child5.add_child(child7)
        child4.add_child(child6)
        child3.add_child(child5)
        child2.add_child(child4)
        child1.add_child(child3)
        root.add_child(child1)
        root.add_child(child2)

        for c in root.depth_first():
            print(c)

    class DepthFirstIterator:
        def __init__(self, start_node):
            self._node = start_node
            self._children_iter = None
            self._child_iter = None

        def __iter__(self):
            return self

        def __next__(self):
            if self._children_iter is None:
                self._children_iter = iter(self._node)
                return self._node
            elif self._child_iter:
                try:
                    next_child = next(self._child_iter)
                    return next_child
                except StopIteration:
                    self._child_iter = None
                    return next(self)
            else:
                self._child_iter = next(self._children_iter).depth_first()
                return next(self)

    class Node2(Node):
        def depth_first(self):
            return IterTest.DepthFirstIterator(self)

    def test_node2(self):
        root = IterTest.Node2(0)
        child1 = IterTest.Node2(1)
        child2 = IterTest.Node2(2)
        child3 = IterTest.Node2(3)
        child4 = IterTest.Node2(4)
        child5 = IterTest.Node2(5)
        child6 = IterTest.Node2(6)
        child7 = IterTest.Node2(7)

        child5.add_child(child7)
        child4.add_child(child6)
        child3.add_child(child5)
        child2.add_child(child4)
        child1.add_child(child3)
        root.add_child(child1)
        root.add_child(child2)

        for c in root.depth_first():
            print(c)

    class Countdown:
        def __init__(self, start):
            self._start = start

        def __iter__(self):
            n = self._start
            while n > 0:
                yield n
                n -= 1

        def __reversed__(self):
            n = 1
            while n <= self._start:
                yield n
                n += 1

    def test_reversed(self):
        cd = IterTest.Countdown(10)
        for n in cd:
            print(n)

        for n in reversed(cd):
            print(n)

    def test_islice(self):
        cd = IterTest.Countdown(30)
        for n in itertools.islice(cd, 10, 15):
            print(n)
        print('------')

        for n in itertools.islice(cd, None, 5):
            print(n)
        print('-------')

        for n in itertools.islice(cd, 25, None):
            print(n)

    def test_drop_while(self):
        cd = IterTest.Countdown(30)
        for n in itertools.dropwhile(lambda x: x > 5, cd):
            print(n)

    def test_permutations(self):
        a = ['a', 'b', 'c']
        for p in itertools.permutations(a):
            print(p)

        for p in itertools.permutations(a, 2):
            print(p)

    def test_combinations(self):
        a = ['a', 'b', 'c']
        for p in itertools.combinations(a, 3):
            print(p)

        for p in itertools.combinations(a, 2):
            print(p)

    def test_combinations_with_replacement(self):
        a = ['a', 'b', 'c']
        for p in itertools.combinations_with_replacement(a, 3):
            print(p)
