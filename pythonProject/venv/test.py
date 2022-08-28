# import RPi.GPIO as GPIO
# GPIO.output(1,GPIO.HIGH)
# print('good')


# import multiprocessing
# import time
#
#
# def worker(d, key, value):
#     d[key] = value
#
#
# if __name__ == '__main__':
#     mgr = multiprocessing.Manager()
#     d = mgr.dict()
#
#     jobs = [multiprocessing.Process(target=worker, args=(d, i, i * 2))
#             for i in range(10)
#             ]
#     for j in jobs:
#         j.start()
#     for j in jobs:
#         j.join()
#     print('Results:')
#     for key, value in enumerate(dict(d)):
#         print("%s=%s" % (key, value))

import multiprocessing, time


def task(args):
    count = args[0]
    queue = args[1]
    for i in xrange(count):
        queue.put("%d mississippi" % i)
    return "Done"


def main():
    q = multiprocessing.Queue()
    pool = multiprocessing.Pool()
    result = pool.map_async(task, [(x, q) for x in range(10)])
    time.sleep(1)
    while not q.empty():
        print
        q.get()
    print
    result.get()


if __name__ == "__main__":
    main()



