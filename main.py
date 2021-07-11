import time
from Counter import Counter

if __name__ == '__main__':
    start_time = time.time()
    counter = Counter('assets/PXL_20201219_110148516.mp4', True)
    counter.calculate()
    print("--- %s seconds ---" % (time.time() - start_time))
