# -*- coding: utf-8 -*-

import unittest


class GILTest(unittest.TestCase):
    """
    关于Python中的GIL(全局解释器锁)
    1. 尽管Python完全支持多线程编程，但是解释器的C语言实现部分在完全并行执行时并不是线程安全的。
    2. 实际上，解释器被一个全局的解释器锁(GIL)保护着，其确保任何时候只有一个Python线程在执行。
    3. GIL最大的问题在于Python的多线程并不能利用多核CPU的优势，多线程进行计算密集型程序只会在单个CPU上运行。
    4. GIL只影响多线程，不影响多进程。
    5. 对于依赖CPU资源的程序，根据运算特点使用不同的优化策略：
        1. 优化底层算法会非常有效。
        2. 使用C语言扩展模块，比如使用NumPy来操作数组。
        3. 使用PyPy，它通过JIT编译器来优化执行效率。
        4. etc.
    6. 有两种策略来优化GIL的缺点：
        1. 如果完全工作与Python，使用进程池来处理CPU密集型任务。计算任务在单独的解释器中运行，不会受限与GIL。线程等待结果的时候会释放GIL。
        2. 使用C扩展编程技术。将计算密集型任务转移给C，跟Python独立，在C代码中释放GIL。释放动作可以使用特殊宏完成，有些Cython库会自动释放GIL，如ctypes。
    7. 无论是使用进程池还是C扩展，都会带来额外的开销，要平衡好开销与收益。足够量的CPU密集任务才能获得收益。
    8. 如果将多线程和多进程混合使用，最好在程序启动时，创建任何线程之前先创建一个单利的进程池。后续线程都是用该进程池来进行计算密集型任务。
    9. C扩展最重要的特性是它们与Python解释器保持独立。这意味着不要在C扩展中使用Python数据结构。
    10. 这些解决GIL的方案并不适用于所有问题。例如，某些任务分解为多个进程处理的话并不能很好的工作，也就不能将它的代码改为C语言执行。
        对于这些应用，就需要根据需求找解决方案了(比如多进程访问共享内存区，多解析器运行与同一个进程等)。或者考虑一下其他解析器实现，如PyPy。
    """
    pass