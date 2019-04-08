# -*- coding: utf-8 -*-

import unittest
import threading
import time

from socket import socket, AF_INET, SOCK_STREAM
from functools import partial
from queue import Queue
from collections import defaultdict
from contextlib import contextmanager


class ThreadStartTest(unittest.TestCase):
    @staticmethod
    def count_down(n):
        while n > 0:
            print('T-minus', n)
            n -= 1
            time.sleep(1)

    def test_start_thread(self):
        thread = threading.Thread(target=self.count_down, args=(5,))
        self.assertFalse(thread.is_alive())
        thread.start()
        self.assertTrue(thread.is_alive())
        # join在start之后
        thread.join()

    def test_daemon_thread(self):
        thread = threading.Thread(target=self.count_down, args=(5,), daemon=True)
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

        started_event = threading.Event()

        def count_down(n):
            print('count_down starting')
            started_event.set()

            while n > 0:
                print('T-minus: %d' % n)
                n -= 1
                time.sleep(1)

        thread = threading.Thread(target=count_down, args=(5,))
        self.assertFalse(thread.is_alive())
        thread.start()
        self.assertTrue(thread.is_alive())

        started_event.wait()
        print('count_down is running')
        thread.join()

    def test_condition(self):
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
        threading.Thread(target=producer, args=(q,)).start()
        threading.Thread(target=consumer, args=(q,)).start()

    def test_queue_task_done_and_join(self):
        """
        Queue对象提供基本的完成特性
        1. 使用Queue进行线程间通信是单向、不确定的过程。
        2. Queue提供基本的完成特性，task_done()和join()。
        :return:
        """

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

        threading.Thread(target=producer, args=(q,)).start()
        threading.Thread(target=consumer, args=(q,)).start()

        q.join()  # 等待Queue中所有消息处理完成

    def test_event(self):
        """
        使用Event，消费者可以监控消费者处理过程。
        1. 消费者线程处理完特定的数据项时立即得到通知，可以将一个Event和数据放在一起。
        :return:
        """

        _sentinel = object()

        def producer(q: Queue):
            for i in range(5):
                evt = threading.Event()
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

        threading.Thread(target=producer, args=(q,)).start()
        threading.Thread(target=consumer, args=(q,)).start()

    def test_Condition(self):
        """
        使用Condition实现优先队列
        1. 创建自己的数据结构，并添加所需的锁和同步机制实现线程间通信。
        2. 最常见的方式就是使用Condition来包装我们的数据结果。
        :return:
        """
        import heapq

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


class LockTest(unittest.TestCase):
    def test_lock(self):
        """
        在多线程中的临界区加锁避免资源竞争
        1. Lock对象和with语句保证互斥执行。
        2. 线程调度本质上会不确定的，因此在多线程程序中错误的使用锁机制可能会导致随机数据损坏或其他异常行为。
            我们称之为竞争条件。最好只在临界区使用锁。
        3. 相比于显示调用锁获取释放，with语句更加优雅。
        4. 为了避免死锁，最好一个线程一次只获取一个锁，如果做不到，则需要使用高级锁。高级锁一般用于一些特殊情况。
        :return:
        """

        class ShareCounter:
            def __init__(self, init_value=0):
                self._value = init_value
                self._lock = threading.Lock()

            def incr(self, delta=1):
                with self._lock:
                    time.sleep(0.5)
                    self._value += delta
                    print('i', self._value)

            def decr(self, delta=1):
                with self._lock:
                    time.sleep(0.5)
                    self._value -= delta
                    print('d', self._value)

        def incr(sc: ShareCounter):
            for i in range(8):
                sc.incr()

        def decr(sc: ShareCounter):
            for i in range(5):
                sc.decr()

        sc = ShareCounter()
        t1 = threading.Thread(target=incr, args=(sc,))
        t2 = threading.Thread(target=decr, args=(sc,))
        t1.start()
        t2.start()

    def test_rlock(self):
        """
        RLock(可重入锁，递归锁)，可以被一个线程多次获取
        1. 主要用于实现基于监测对象模式的锁定和同步
        2. 当锁被持有时，只有一个线程可以使用完整的函数和类中的方法。
        3. 该用例中，没有对每个实例加锁，而是使用被所有实例共享的类级锁。
        4. 这个锁用来同步类方法。具体的说就是，同时只有一个线程可以调用这个类的方法。
        5. 如decr方法，已经持有该锁的方法在调用同样使用该锁的方法时，可以多次获取该锁，同时递归层级加一。
        6. 特点是该类无论有多少个实例都只用一个锁，因此需要大量使用计数器的情况下内存效率很高。
        7. 缺点是大量线程频繁更新计数器时会有争用锁的问题。
        :return:
        """

        class ShareCounter:
            _lock = threading.RLock()  # 类级锁

            def __init__(self, init_value=0):
                self._count = init_value

            def incr(self, delta=1):
                with ShareCounter._lock:
                    time.sleep(0.5)
                    self._count += delta
                    print(self._count)

            def decr(self, delta=1):
                with ShareCounter._lock:
                    self.incr(-delta)  # 持有锁时，调用需要锁的方法

        def incr(sc, count):
            for i in range(count):
                sc.incr()

        def decr(sc, count):
            for i in range(count):
                sc.decr()

        sc1 = ShareCounter()
        sc2 = ShareCounter()

        t1 = threading.Thread(target=incr, args=(sc1, 5,))
        t2 = threading.Thread(target=decr, args=(sc2, 5,))

        t1.start()
        t2.start()

    def test_semaphore(self):
        """
        信号量限制并发
        1. 信号量是建立在共享计数器基础上的同步原语。
        2. 如果计数器不为0，with语句将计数器减一，程序允许执行。with语句结束，计数器加一。
        3. 如果计数器为0，线程阻塞，直到其他线程结束计数器加一。
        4. 信号量的复杂性影响性能，不建议像标准锁一样使用信号量来锁线程同步。
        5. 信号量更适用于需要在线程之间引入信号或限制的程序。比如，限制一段代码的并发量。
        :return:
        """

        def url_open(url):
            # 模拟请求url
            time.sleep(1)
            return 'request url done', url

        _fetch_url_semaphore = threading.Semaphore(3)

        def fetch_url(url):
            with _fetch_url_semaphore:
                print(url_open(url))

        for i in range(10):
            threading.Thread(target=fetch_url, args=('url%d' % i,)).start()


class NoDeadLockTest(unittest.TestCase):
    """
    避免死锁
    1. 尽可能保证每一个线程只能同时保持一个锁，这样程序就不会死锁。
    2. 死锁的检测和恢复几乎没有优雅的解决方案，常见的两种方式：
        1. watchdog timer(看门狗计数器): 正常线程运行时每隔一段时间重置计数器，一旦发生死锁，则无法重置计数器，导致计数器超时，这是程序通过重启恢复。
        2. 进程获取锁的时候严格按照对象id升序排列获取。数学证明，可以保证不出现死锁。
    """
    from contextlib import contextmanager

    _local = threading.local()

    @contextmanager
    def acquire(self, *locks):
        """
        同时获取多个锁，保证顺序获取，防止死锁
        1. 为每个锁分配一个唯一ID，只允许按照升序规则来使用多个锁。
        2. 该规则使用上下文管理器非常容易实现。
        3. 无论是使用单个锁还是多个锁，都使用acquire函数来申请锁。
        4. 即便在不同的函数中使用不同的顺序获取锁也不会发生死锁，关键在于我们对锁进行了排序。通过排序，不管什么顺序请求，都会按照固定的顺序被获取。
        5. 如果多个acquire操作被嵌套调用，可以通过线程本地存储(TLS)来检测潜在的死锁问题。
        :return:
        """
        locks = sorted(locks, key=lambda x: id(x))

        acquired = getattr(self._local, 'acquired', [])  # type:list

        if acquired and max([id(lock) for lock in locks]) >= id(locks[0]):
            raise RuntimeError('Lock order Violation')

        acquired.extend(locks)

        self._local.acquired = acquired

        try:
            for lock in locks:
                lock.acquire()
            yield
        finally:
            for lock in reversed(locks):
                lock.release()
            del acquired[-len(locks):]

    def test_ordered_lock(self):
        """
        不同的函数中使用不同的顺序获取锁也不会发生死锁
        1. 即便在不同的函数中使用不同的顺序获取锁也不会发生死锁，关键在于我们对锁进行了排序。通过排序，不管什么顺序请求，都会按照固定的顺序被获取。
        :return:
        """
        x_lock = threading.Lock()
        y_lock = threading.Lock()

        def thread_1():
            for i in range(10):
                with self.acquire(x_lock, y_lock):
                    print('Thread-1')

        def thread_2():
            for i in range(10):
                with self.acquire(y_lock, x_lock):
                    print('Thread-2')

        threading.Thread(target=thread_1, daemon=True).start()
        threading.Thread(target=thread_2, daemon=True).start()

    def test_nested_acquire(self):
        """
        多个acquire操作被嵌套调用
        1. 如果多个acquire操作被嵌套调用，可以通过线程本地存储(TLS)来检测潜在的死锁问题。
        :return:
        """
        x_lock = threading.Lock()
        y_lock = threading.Lock()

        def thread_1():
            try:
                while True:
                    with self.acquire(x_lock):
                        with self.acquire(y_lock):
                            print('Thread-1')
            except RuntimeError as err:
                print(err)

        def thread_2():
            try:
                while True:
                    with self.acquire(y_lock):
                        with self.acquire(x_lock):
                            print('Thread-2')
            except RuntimeError as err:
                print(err)

        threading.Thread(target=thread_1, daemon=True).start()
        threading.Thread(target=thread_2, daemon=True).start()
        time.sleep(1)

    def test_philosopher(self):
        """
        哲学家就餐问题
        五位哲学家围坐就餐，没人面前只有一只筷子，拿到两只筷子才能吃饭。
        :return:
        """

        def philosopher(left, right):
            with self.acquire(left, right):
                print(threading.currentThread(), 'eating')

        NSTICKS = 5
        chopsticks = [threading.Lock() for _ in range(NSTICKS)]

        for n in range(NSTICKS):
            threading.Thread(target=philosopher, args=(chopsticks[n], chopsticks[(n + 1) % NSTICKS])).start()


class LocalStorageTest(unittest.TestCase):
    class LazyConnection:
        """
        可用于多线程的LazyConnection上下文管理类
        1. threading.local()创建一个本地线程存储对象。
        2. 该对象的属性的保存和读取操作都只会对执行线程可见，其他线程不可见。
        """

        def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
            self._address = address
            self._family = family
            self._type = type
            self._local = threading.local()

        def __enter__(self):
            if hasattr(self._local, 'sock'):
                raise RuntimeError('Already connected')
            self._local.sock = socket(family=self._family, type=self._type)
            self._local.sock.connect(self._address)
            return self._local.sock

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._local.sock.close()
            del self._local.sock

    def test_local(self):
        """
        使用LazyConnection进行多线程请求操作
        1. 每个线程会创建属于自己的专属套接字连接，并存储为local.sock。
        2. 不同线程操作的是不同的套接字，因此不会相互影响。
        :return:
        """

        def t(host, conn):
            with conn as s:
                s.send(b'GET / HTTP/1.0\r\n')
                s.send(b'Host: %s\r\n' % host)
                s.send(b'\r\n')
                resp = b''.join(iter(partial(s.recv, 8192), b''))

            print('Got %d bytes' % len(resp))

        host1 = b'www.python.org'
        host2 = b'www.bing.com'
        conn1 = self.LazyConnection((host1, 80))
        conn2 = self.LazyConnection((host2, 80))
        t1 = threading.Thread(target=t, args=(host1, conn1,))
        t2 = threading.Thread(target=t, args=(host2, conn2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()


class ActorTest(unittest.TestCase):
    """
    actor模式
    1. actor模式是一种最古老的也是最简单的并行和分布式计算解决方案。
    2. 其天生的简单性是它如此受欢迎的原因之一。
    3. 简单来讲，一个actor就是一个并发执行的任务，只是简单的执行发送给它的消息任务。
    4. 响应消息时，actor还可以给其他actor发送进一步的消息。
    5. actor之间的通信是单向和异步的。因此，消息发送者不知道消息是什么时候被发送的，也不会接收到一个消息处理之后回应或通知。
    6. “发送”一个任务消息的概念可以被扩展到多进程甚至大型分布式系统中去。
        例如，一个类actor对象的send()方法可以编程为让它能在一个套接字连接上传输数据或通过某些消息中间件来发送。
    """

    class ActorExit(Exception):
        """
        哨兵，用来关闭actor
        """
        pass

    class Actor:
        """
        actor实现
        1. 如果放宽对于同步和异步消息发送的要求，该类也可以使用生成器来简化定义。
        """

        def __init__(self):
            self._mail_box = Queue()
            self._terminated = threading.Event()

        def send(self, msg):
            # 通过该actor发送消息
            self._mail_box.put(msg)

        def recv(self):
            msg = self._mail_box.get()
            if msg is ActorTest.ActorExit:  # 如果接收到哨兵，则关闭该actor
                raise ActorTest.ActorExit
            return msg

        def close(self):
            # 通过向队列发送哨兵的方式安全的关闭actor
            self.send(ActorTest.ActorExit)

        def start(self):
            # 在一个内部线程中处理消息
            threading.Thread(target=self._boot_strap, daemon=True).start()

        def _boot_strap(self):
            # 引导
            try:
                self.run()
            except ActorTest.ActorExit:
                pass
            finally:
                self._terminated.set()

        def join(self):
            self._terminated.wait()

        def run(self):
            while True:
                self.recv()

    def test_actor(self):
        a = ActorTest.Actor()
        a.start()
        a.send('Hello')
        a.send('World')
        a.close()
        a.join()

    class PrintActor(Actor):
        def run(self):
            while True:
                msg = self.recv()
                print('Got', msg)

    def test_print_actor(self):
        p = ActorTest.PrintActor()
        p.start()
        p.send('Hello')
        p.send('World')
        p.close()
        p.join()

    class Result:
        """
        结果对象，用于在actor中返回结果
        """
        def __init__(self):
            self._evt = threading.Event()
            self._result = None

        def set_result(self, value):
            self._result = value
            self._evt.set()

        def result(self):
            self._evt.wait()
            return self._result

    class Worker(Actor):
        """
        actor允许在一个工作者中运行任意的函数，并通过一个特殊的Result对象返回结果
        """
        def submit(self, func, *args, **kwargs):
            r = ActorTest.Result()
            self.send((func, args, kwargs, r))
            return r

        def run(self):
            while True:
                func, args, kwargs, r = self.recv()
                r.set_result(func(*args, **kwargs))

    def test_worker(self):
        worker = ActorTest.Worker()
        worker.start()
        r = worker.submit(pow, 2, 3)
        print(r.result())


class ExchangeTest(unittest.TestCase):
    """
    消息发布/订阅模型
    1. 要实现发布/订阅的消息通信模式，通常需要引入一个单独的“交换机”或“网关”对象作为所有消息的中介。
    2. 不直接将消息从一个任务发送到另一个，而是将其发送给交换机，然后再由交换机将它发送给一个或多个被关联的任务。
    3. 一个交换机就是一个普通的对象，负责维护一个活跃的订阅者集合，并为绑定、解绑和发送消息提供响应的方法。
    4. 尽管有很多变种，不过万变不离其宗。消息会发送给一个交换机，然后交换机会将他们发送给被绑定的订阅者。
    5. 通过队列发送消息的任务或线程的模式很容易被实现并且也非常普遍。不过，使用发布/订阅模式的好处更加明显：
        1. 使用一个交换机可以简化大部分涉及到线程通信的工作。类似于日志模块的工作原理。
        2. 交换机广播消息给多个订阅者的能力带来了一个全新的通信模式。例如，可以使用多任务系统、广播或扇出。
        3. 兼容多种"task-like"对象。如actor、协程、网络连接或任何实现了正确send()方法的对象。
    6. 关于交换机的思想有很多中的扩展实现。例如，交换机可以实现一整个消息通道集合或提供交换机名称的模式匹配规则。
    7. 交换机可以扩展到分布式计算程序中，比如将消息路由到不同机器上面的任务中去。
    """

    class Exchange:
        """
        交换机实现
        """

        def __init__(self):
            self._subscribers = set()

        def attach(self, task):
            self._subscribers.add(task)

        def detach(self, task):
            self._subscribers.remove(task)

        @contextmanager
        def subscribe(self, *tasks):
            """
            使用上下文管理器，避免出现忘记detach的情况
            :param tasks:
            :return:
            """
            for task in tasks:
                self.attach(task)
            try:
                yield
            finally:
                for task in tasks:
                    self.detach(task)

        def send(self, msg):
            for subscriber in self._subscribers:
                subscriber.send(msg)

    # 使用dict管理交换机
    _exchanges = defaultdict(Exchange)

    @staticmethod
    def get_exchange(name):
        """
        通过name获取交换机
        :param name:
        :return:
        """
        return ExchangeTest._exchanges[name]

    class Task:
        """
        模拟task
        """

        def send(self, msg):
            print('Send msg in task: ', msg)

    class DisplayMessages:
        """
        简单诊断类
        以普通订阅者身份绑定来构建调试和诊断工具
        """

        def __init__(self):
            self.counter = 0

        def send(self, msg):
            self.counter += 1
            print('msg[%d]: %s' % (self.counter, msg))

    def test_exchange(self):
        """
        交换机测试
        :return:
        """
        task_a = ExchangeTest.Task()
        task_b = ExchangeTest.Task()
        task_display_msg = ExchangeTest.DisplayMessages()

        exc = ExchangeTest.get_exchange('te')

        threading.Thread(target=exc.attach, args=(task_a,)).start()
        threading.Thread(target=exc.attach, args=(task_b,)).start()
        threading.Thread(target=exc.attach, args=(task_display_msg,)).start()

        exc.send('Hello ')
        exc.send('world!')

        exc.detach(task_a)
        exc.detach(task_b)
        exc.detach(task_display_msg)

    def test_exchange_with(self):
        """
        交换机上下文管理器测试
        :return:
        """
        task_a = ExchangeTest.Task()
        task_b = ExchangeTest.Task()
        task_display_msg = ExchangeTest.DisplayMessages()

        exc = ExchangeTest.get_exchange('tew')

        with exc.subscribe(task_a, task_b, task_display_msg):
            exc.send('Hello')
            exc.send('world!')
