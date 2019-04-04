# -*- coding: utf-8 -*-

import unittest

import socket
import threading
import urllib.request
from concurrent import futures
from queue import Queue


class ThreadPoolTest(unittest.TestCase):
    """
    使用线程池调用多线程
    1. 现在操作系统可以很轻松的创建几千个线程的线程池，而且几千个线程同时等待工作并不会对其他代码产生性能影响。
    2. 但是大量线程同时被唤醒并立即在CPU上执行就不同了，特别是有了全局解释器锁GIL。
    3. 通常只应该在I/O处理相关代码中使用线程池。
    4. 创建线程时，操作系统会预留一个虚拟内存区域来放置线程执行栈，通常为8MB，但虚拟内存只有一小片段被实际映射到真是内存。
        2000个线程，会使用9GB虚拟内存，但是只会使用大约70MB真实内存。
    5. threading.stack_size()可以来调整预留区域大小。
    """

    def test_thread_pool(self):
        """
        使用线程池实现响应客户端请求
        :return:
        """

        def echo_client(sock: socket.socket, client_addr):
            # 处理客户端连接，接收和返回数据
            print('Got connection from', client_addr)
            while True:
                msg = sock.recv(65536)
                if not msg:
                    break
                print(b'client: ' + msg)
                sock.sendall(b'server response: ' + msg)
            print('Client closed connection')
            sock.close()

        def echo_server(addr, event):
            # 启动TCP服务器
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(addr)
            sock.listen(5)
            sock.settimeout(5.0)  # 设置sock操作超时，用于退出case
            with futures.ThreadPoolExecutor(128) as pool:  # 128个workers的线程池
                event.set()
                while True:
                    try:
                        client_sock, client_addr = sock.accept()  # 等待客户端连接
                        pool.submit(echo_client, client_sock, client_addr)  # 将客户端连接交于线程池处理
                    except socket.timeout:
                        # 长时间没有客户端连接，timeout，服务器退出。
                        # 用于退出case
                        print('No client connect before timeout, server closed')
                        sock.close()
                        break

        def client(addr):
            # 客户端连接
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(addr)
            s.send(b'hello')
            print(s.recv(8192))
            s.close()

        event = threading.Event()
        addr = ('', 15000)
        threading.Thread(target=echo_server, args=(addr, event,)).start()  # 新线程中运行服务器
        event.wait()
        client(addr)

    def test_use_queue_as_thread_pool(self):
        """
        使用Queue来手动实现线程池
        1. 功能与上一个case相同，只是用Queue手动来实现ThreadPool
        :return:
        """

        def echo_client(q: Queue):
            while True:
                sock, client_addr = q.get()  # 等待获取客户端连接
                print(threading.currentThread(), 'Got connection from: ', client_addr)
                while True:
                    msg = sock.recv(65536)
                    if not msg:
                        break
                    print(b'client ' + msg)
                    sock.sendall(b'server response: ' + msg)

                print(threading.currentThread(), 'Client closed connection')
                sock.close()

        def echo_server(addr, workers, q, event):
            for _ in range(workers):  # 启动多个线程
                threading.Thread(target=echo_client, args=(q,), daemon=True).start()  # 使用Queue来分配客户端连接

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(addr)
            sock.listen(5)
            sock.settimeout(5.0)
            event.set()  # 服务器启动完成，通知主线程继续执行
            while True:
                try:
                    client_sock, client_addr = sock.accept()
                    q.put((client_sock, client_addr))  # 新的客户端连接进来，丢入Queue，再由线程池中的worker获取处理
                except socket.timeout:
                    print('No client connect before timeout, server closed')
                    sock.close()
                    break

        def client(addr):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(addr)
            s.send(b'hello')
            print(s.recv(8192))
            s.close()

        event = threading.Event()  # 使用event来延缓client连接
        addr = ('', 15001)
        threading.Thread(target=echo_server, args=(addr, 128, Queue(), event,)).start()
        event.wait()  # 等待服务器启动完成
        client(addr)

    def test_fetch_url(self):
        """
        多线程并发访问url
        :return:
        """

        def fetch(url):
            return urllib.request.urlopen(url).read()

        pool = futures.ThreadPoolExecutor(10)

        # 向线程池提交任务，返回future
        a = pool.submit(fetch, 'http://www.python.org')
        b = pool.submit(fetch, 'http://www.pypy.org')

        # future.result()会阻塞，等待结果
        print(a.result())
        print(b.result())


class ProcessPoolTest(unittest.TestCase):
    """
    多进程
    1. 适合于计算密集型函数，可以利用多核优势。
    2. concurrent.futures库的ProcessPoolExecutor提供了进程池功能。
    3. ProcessPoolExecutor默认创建N个Python解释器，N为CPU数。
    4. 使用with语句配合线程池时，线程池会等待最后一个语句执行完成，然后关闭线程池。
    5. 尽管进程池很容易使用，在设计大程序时还是要注意如下几点：
        1. 这种并行处理技术只适用于那些可以被分解为相互独立部分的问题。
        2. 被提交的任务必须是简单的函数形式。对于方法、闭包和其他类型的并行执行还不支持。
        3. 函数参数和返回值必须兼容pickle，因为要使用到进程间的通信，所有的解释器之间的数据交换必须序列化。
        4. 被提交的任务函数不应保留状态或有副作用。除了打印日志之类的简单事情。
    6. 子进程一旦启动，我们就不能控制其任何行为，因此最好保持简单和单纯，函数不要修改环境。
    7. 在第一次调用pool.map()或pool.submit()之后才会实际fork进程。
    """

    @staticmethod
    def CPU_consumption(num):
        # 通过大量运算来实现消耗CPU资源
        b = 10 ** 7
        r = 0.0
        for i in range(num * b, (num + 1) * b):
            r += i
        return r

    def test_single_process(self):
        # 单进程测试
        result = 0
        for r in map(ProcessPoolTest.CPU_consumption, range(3)):
            result += r
        print(result)

    def test_process_pool(self):
        """
        进程池执行CPU密集运算
        :return:
        """
        result = 0
        with futures.ProcessPoolExecutor() as pool:  # 进程池配合with使用
            # with语句中的所有语句执行完成后，pool才会关闭
            # ================map语法demo start=====================
            for r in pool.map(ProcessPoolTest.CPU_consumption, range(10)):  # map语法
                result += r
            # ================map语法demo end=====================

            # ================submit+result()语法demo start=====================
            # futures_list = []
            # for i in range(10):
            #     _future = pool.submit(ProcessPoolTest.CPU_consumption, i)  # submit语法
            #     futures_list.append(_future)
            #
            # for _f in futures_list:
            #     result += _f.result()  # 使用result()获取结果语法

            # ================submit+result()语法demo end=====================

            # ================submit+callback语法demo start=====================
            # def when_done(_future):
            #     print(_future.result())
            #
            # for i in range(10):
            #     _future = pool.submit(ProcessPoolTest.CPU_consumption, i)  # submit语法
            #     _future.add_done_callback(when_done)
            # ================submit+callback语法demo end=====================
        print(result)
