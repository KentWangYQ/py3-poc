# -*- coding: utf-8 -*-

import unittest


class ThreadStartTest(unittest.TestCase):
    @staticmethod
    def count_down(n):
        import time
        while n > 0:
            print('T-minus', n)
            n -= 1
            time.sleep(1)

    def test_start_thread(self):
        from threading import Thread
        thread = Thread(target=self.count_down, args=(5,))
        self.assertFalse(thread.is_alive())
        thread.start()
        self.assertTrue(thread.is_alive())
        # join在start之后
        thread.join()

    def test_daemon_thread(self):
        from threading import Thread
        thread = Thread(target=self.count_down, args=(5,), daemon=True)
        self.assertFalse(thread.is_alive())
        thread.start()
        self.assertTrue(thread.is_alive())
        thread.join()


class EventConditionTest(unittest.TestCase):
    def test_event(self):
        """
        event test
        1. 'count_down is running'总是在'count_down starting'之后，因为需要等待event变为True。
        2. event最好单次使用，虽然可以使用clear()清理，但很难保证完全清理。重复使用容易造成错过事件，死锁等问题。
        :return:
        """

        from threading import Event, Thread
        started_event = Event()

        def count_down(n):
            import time
            print('count_down starting')
            started_event.set()

            while n > 0:
                print('T-minus: %d' % n)
                n -= 1
                time.sleep(1)

        thread = Thread(target=count_down, args=(5,))
        self.assertFalse(thread.is_alive())
        thread.start()
        self.assertTrue(thread.is_alive())

        started_event.wait()
        print('count_down is running')
        thread.join()

    def test_condition(self):
        import threading
        import time

        class PeriodicTimer:
            def __init__(self, interval):
                self._interval = interval
                self._flag = 0
                self._cv = threading.Condition()

            def start(self):
                thread = threading.Thread(target=self.run, daemon=True)
                thread.start()

            def run(self):
                """
                Producer
                :return:
                """
                while True:
                    time.sleep(self._interval)
                    with self._cv:
                        self._flag ^= 1
                        self._cv.notify_all()

            def wait(self):
                """
                Consumer
                :return:
                """
                with self._cv:
                    flag = self._flag
                    while flag == self._flag:
                        self._cv.wait()

        p_timer = PeriodicTimer(1)
        p_timer.start()

        def count_down(n):
            while n > 0:
                p_timer.wait()
                print('T-minus', n)
                n -= 1

        def count_up(last):
            n = 0
            while n < last:
                p_timer.wait()
                print('counting', n)
                n += 1

        threading.Thread(target=count_down, args=(10,)).start()
        threading.Thread(target=count_up, args=(5,)).start()

    def test_semaphore(self):
        import threading
        import time

        def worker(n, semaphore: threading.Semaphore):
            semaphore.acquire()
            print('Working', n)

        semaphore = threading.Semaphore(0)
        n_workers = 5
        for n in range(n_workers):
            thread = threading.Thread(target=worker, args=(n, semaphore,))
            thread.start()

        for n in range(n_workers):
            time.sleep(1)
            semaphore.release()


class CommunicationTest(unittest.TestCase):
    def test_queue(self):
        """
        线程间通信：队列
        1. 线程间通信最安全的方式就是使用queue库的Queue。
        2. Queue对象已经包含了必要的锁，所以可以在多线程间安全共享数据。
        3. 使用Queue通信的一个问题是协调生产者和消费者关闭问题，通常做法是放置特殊值。
        4. 这里操作的一个特殊点是消费着收到关闭信号后，又将信号发回到队列，保证监听该队列的所有消费者都可以关闭。
        5. 基于Queue消息的通信机制可以扩展到更大的应用范畴，比如：多进程，甚至分布式，都无需改变底层队列结构。
        6. 使用Queue线程间通信传递的是对象的引用，如果担心对象的共享状态，最好使用不可修改的数据结构(整形、字符串或元组等)或者对象的深拷贝。
        7. Queue提供一些当前上下文很有用的附加特性：
            1. size：限制添加到queue的元素数量，当生产者和消费者有速度差异时，可以防止队列无限扩充，造成失控。
            2. blocking：非阻塞与固定大小配合实现流量控制，当queue已满时，执行不同的操作，如记录并丢弃元素。
            3. timeout：超时自动终止，以便检查终止标志。
            4. q.qsize(), q.full(), q.empty()实用方法获取队列的状态，但是非线程安全。
        :return:
        """
        import time
        from threading import Thread
        from queue import Queue

        _sentinel = object()  # 关闭信号

        def producer(q: Queue):
            for i in range(5):
                time.sleep(1)
                q.put(i)
                print('PUT data', i)
            q.put(_sentinel)  # 生产者发出关闭信号
            print('producer closed')

        def consumer(q: Queue):
            while True:
                data = q.get()
                if data == _sentinel:
                    # 收到生产者发送的关闭信号，关闭消费者
                    q.put(_sentinel)  # 将关闭信号发回队列，保证所有监听该队列的所有消费者都可以关闭
                    print('Consumer get sentinel, closed')
                    break
                else:
                    print('GET data', data)

        q = Queue()
        Thread(target=producer, args=(q,)).start()
        Thread(target=consumer, args=(q,)).start()

    def test_queue_task_done_and_join(self):
        """
        Queue对象提供基本的完成特性
        1. 使用Queue进行线程间通信是单向、不确定的过程。
        2. Queue提供基本的完成特性，task_done()和join()。
        :return:
        """
        import time
        from queue import Queue
        from threading import Thread

        _sentinel = object()  # 关闭信号

        def producer(q: Queue):
            for i in range(5):
                q.put(i)
                print('PUT', i)

            q.put(_sentinel)  # 向队列发送关闭信号
            print('Producer closed')

        def consumer(q: Queue):
            while True:
                time.sleep(1)
                data = q.get()
                if data == _sentinel:
                    print('GET sentinel, consumer closed')
                    q.task_done()  # 标记处理完成
                    break
                else:
                    print('GET', data)
                    q.task_done()  # 标记处理完成

        q = Queue()

        Thread(target=producer, args=(q,)).start()
        Thread(target=consumer, args=(q,)).start()

        q.join()  # 等待Queue中所有消息处理完成

    def test_event(self):
        """
        使用Event，消费者可以监控消费者处理过程。
        1. 消费者线程处理完特定的数据项时立即得到通知，可以将一个Event和数据放在一起。
        :return:
        """
        import time
        from queue import Queue
        from threading import Thread, Event

        _sentinel = object()

        def producer(q: Queue):
            for i in range(5):
                evt = Event()
                q.put((i, evt))
                print('PUT', i)
                evt.wait()

            q.put((_sentinel, None))

        def consumer(q: Queue):
            while True:
                time.sleep(1)
                data, evt = q.get()
                if data == _sentinel:
                    q.put(_sentinel)
                    print('Consumer closed')
                    break
                else:
                    print('GET', data)
                    evt.set()

        q = Queue()

        Thread(target=producer, args=(q,)).start()
        Thread(target=consumer, args=(q,)).start()

    def test_Condition(self):
        """
        使用Condition实现优先队列
        1. 创建自己的数据结构，并添加所需的锁和同步机制实现线程间通信。
        2. 最常见的方式就是使用Condition来包装我们的数据结果。
        :return:
        """
        import time
        import heapq
        import threading

        class PriorityQueue:
            def __init__(self):
                self._queue = []
                self._count = 0
                self._cv = threading.Condition()

            def put(self, item, priority):
                with self._cv:
                    heapq.heappush(self._queue, (-priority, self._count, item))
                    self._count += 1
                    self._cv.notify_all()

            def get(self):
                with self._cv:
                    while len(self._queue) == 0:
                        self._cv.wait()
                    return heapq.heappop(self._queue)[-1]

        _sentinel = object()

        def producer(pq: PriorityQueue):
            for i in range(5):
                time.sleep(1)
                pq.put('item%d' % i, i)
                print('PUT', 'item%d' % i)

            pq.put(_sentinel, float('-inf'))
            print('Producer closed')

        def consumer(pq: PriorityQueue):
            while True:
                data = pq.get()
                if data == _sentinel:
                    pq.put(_sentinel, float('-inf'))
                    print('GET _sentinel, consumer closed')
                    break
                else:
                    print('GET', data)

        pq = PriorityQueue()

        threading.Thread(target=producer, args=(pq,)).start()
        threading.Thread(target=consumer, args=(pq,)).start()
