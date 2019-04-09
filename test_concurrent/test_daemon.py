# -*- coding: utf-8 -*-

import unittest
import os
import sys
import threading
import time

import atexit
import signal


class DaemonTest(unittest.TestCase):
    """
    守护进程
    1. 创建一个守护进程的步骤大体思想如下：
        1. 首先，一个守护进程必须要从父进程中脱离，这是由os.fork()操作来完成的，并立即被父进程终止。
        2. 在子进程成为孤儿后，调用os.setsid()创建了一个全新的进程会话，并设置子进程为首领。它会设置这个子进程为新的进程组的首领，并确保不会再有控制终端。
            因为需要将守护进程与终端分离，并确保信号机制对它不起作用。
        3. 调用os.chdir()和os.umask(0)改变了当前工作目录并重置文件权限掩码。修改目录通常是个好主意，因为这样可以使得它不再工作在被启动时的目录。
        4. 第二次os.fork在这里更加神秘。这一步使得守护进程失去了获取新的控制终端的能力，并且让它更加独立(本质上，该daemon放弃了它的会话首领地位，因此
            再也没有权限去打开控制终端了)。尽管我们可以忽略这一步，但是最好不要这么做。
        5. 一旦守护进程被正确的分离，它会重新初始化标准I/O流指向用户指定的文件。这一部分有点难理解。跟标准I/O流相关的文件对象的引用在解释器中多个地方被找到
            (sys.stdout, sys.__stdout__等)。仅仅简单的关闭sys.stdout并重新指定它是行不通的，因为没有办法知道是否所有的地方都是使用的sys.stdout。这里，
            我们打开了一个单独的文件对象，并调用os.dup2()，用它来替代被sys.stdout使用的文件描述符。这样，sys.stdout使用的原始文件会被关闭并由新的来替换。
            还要强调的是任何用于文件编码或文本处理的标准I/O流还会保留原状。
        6. 守护进程的一个通常实践是在一个文件中写进程ID，可以被其他程序后面使用到。daemonize()函数的最后部分写列这个文件，但是在程序终止时删除了它。
            atexit.register()函数注册了一个函数在Python解释器终止时执行。一个对于SIGTERM的信号处理器的定义同样需要被优雅的关闭。信号处理器简单的抛出了
            SystemExit()异常。或许这一步看上去没有必要，但是没有它，终止信号会使得未执行atexit.register()注册的清理操作时就杀掉解释器。一个杀掉进程的例子
            代码可以在程序最后的stop命令的操作中看到。
    2. 更多关于编写守护进程的信息可以查看《UNIX环境高级编程》。
    """

    @staticmethod
    def daemonize(pidfile, *, stdin='/dev/null',
                  stdout='/dev/null',
                  stderr='/dev/null'):

        if os.path.exists(pidfile):
            raise RuntimeError('Already running')

        # First fork (detaches from parent)
        try:
            if os.fork() > 0:
                raise SystemExit(0)  # Parent exit
        except OSError as e:
            raise RuntimeError('fork #1 failed.')

        os.chdir('/')
        os.umask(0)
        os.setsid()
        # Second fork (relinquish session leadership)
        try:
            if os.fork() > 0:
                raise SystemExit(0)
        except OSError as e:
            raise RuntimeError('fork #2 failed.')

        # Flush I/O buffers
        sys.stdout.flush()
        sys.stderr.flush()

        # Replace file descriptors for stdin, stdout, and stderr
        with open(stdin, 'rb', 0) as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
        with open(stdout, 'ab', 0) as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
        with open(stderr, 'ab', 0) as f:
            os.dup2(f.fileno(), sys.stderr.fileno())

        # Write the PID file
        with open(pidfile, 'w') as f:
            print(os.getpid(), file=f)

        # Arrange to have the PID file removed on exit/signal
        atexit.register(lambda: os.remove(pidfile))

        # Signal handler for termination (required)
        def sigterm_handler(signo, frame):
            raise SystemExit(1)

        signal.signal(signal.SIGTERM, sigterm_handler)

    def daemon(self, PIDFILE):
        try:
            DaemonTest.daemonize(PIDFILE, stdout='/tmp/daemon.log', stderr='/tmp/daemon.log')
        except RuntimeError as e:
            print(e, file=sys.stderr)
            raise SystemExit(1)

        sys.stdout.write('Daemon started with pid %s\n' % os.getpid())
        sys.stdout.write('Daemon alive!\n')

        if os.path.exists(PIDFILE):
            with open(PIDFILE) as f:
                os.kill(int(f.read()), signal.SIGTERM)

    PIDFILE = '/tmp/daemon.pid'

    def test_daemon(self):
        t = threading.Thread(target=self.daemon, args=(self.PIDFILE,))
        t.start()
        time.sleep(1)
        if os.path.exists(self.PIDFILE):
            os.remove(self.PIDFILE)
