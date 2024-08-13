import time

class Timer():
    def __init__(self):
        self.start_time = time.time()
        self._seconds = 0.0

    def start(self) -> 'Timer':
        self.start_time = time.time()
        return self

    def stop(self):
        self._seconds = time.time() - self.start_time

    @property
    def seconds(self) -> float:
        """Get the number of seconds for either stopped or running timer."""
        return self._seconds if self._seconds > 0.0 else time.time() - self.start_time

    def reset(self):
        self.start_time = time.time()
        self._seconds = 0.0