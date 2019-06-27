---
title: yield和iter的关系
---

iter定义
========

内建函数iter()可以生成一个iterator迭代器。

-   优点

    相比list来说，iterator不需要很大的内存空间。

    使用迭代器解决了空间占用的问题，不过代码也太繁琐了，一点没有python风格。

-   操作

    迭代器通过next()来遍历元素，并且完成遍历时抛出*StopIteration()*异常。

<!-- -->

        it = iter(range(5))
        print it.next()         # 0
        print it.next()         # 1
        print it.next()         # 2
        print it.next()         # 3
        print it.next()         # 4

        print it.next()         # StopIteration()

用for循环对迭代器进行遍历

        it = iter(range(5))
        for i in it:
            print i

        print it.next()         # StopIteration()

遍历完成时，调用it.next()，抛出StopIteration()异常。可以看出for循环调用的是next()方法。就像下边这样。

        while True:
        try:
            print it.next()
        except StopIteration:
            break    

yield定义
=========

yield是为了应对iter代码太复杂的简化方案，是一个可迭代的并且简洁的方案。

-   特点：

    使用next()方法会依次返回元素，并且越界时报StopIteration异常。

<!-- -->

    def myEnumerate(seq, start=0):
        n = start
        for i in seq:
            yield n, i
            n += 1
