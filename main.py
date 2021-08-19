import time
from counter import Counter

if __name__ == '__main__':
    start_time = time.time()
    counter = Counter('assets/2021_08_15.mp4', False)
    counter.calculate()
    print("--- %s seconds ---" % (time.time() - start_time))

