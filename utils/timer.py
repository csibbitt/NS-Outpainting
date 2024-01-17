import time

class Timer():
    def __init__(self, label):
        self.label = label
        self.start = time.perf_counter()
        self.lapstart = self.start

    def stop(self):
        print('{} taking : {}s'.format(self.label, int(time.perf_counter() - self.start)))

    def lap(self):
        now = time.perf_counter()
        print('{} lap taking : {}s'.format(self.label, int(now - self.lapstart)))
        self.lapstart = now

    def reset(self):
        self.start = time.perf_counter()
        self.lap = self.start
