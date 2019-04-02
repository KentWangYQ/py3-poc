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
                thread = threading.Thread(target=self.run)
                thread.daemon = True
                thread.start()

            def run(self):
                while True:
                    time.sleep(self._interval)
                    with self._cv:
                        self._flag ^= 1
                        self._cv.notify_all()

            def wait_for_tick(self):
                with self._cv:
                    last_flag = self._flag
                    while last_flag == self._flag:
                        self._cv.wait()

        p_timer = PeriodicTimer(1)
        p_timer.start()

        def count_down(n_ticks):
            while n_ticks > 0:
                p_timer.wait_for_tick()
                print('T-minus', n_ticks)
                n_ticks -= 1

        def count_up(last):
            n = 0
            while n < last:
                p_timer.wait_for_tick()
                print('counting', n)
                n += 1

        threading.Thread(target=count_down, args=(10,)).start()
        threading.Thread(target=count_up, args=(5,)).start()