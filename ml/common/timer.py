from time import time
from datetime import timedelta
import logging


class Timer:
    def __init__(self, label):
        self.label = label
        self.begin = time()
        self.paused_seconds = 0.0
        self.pause_time = None

    def start(self):
        self.begin = time()

    def pause(self):
        if self.pause_time is not None:
            logging.error('Timer already paused')
            return
        self.pause_time = time()

    def resume(self):
        self.paused_seconds += (time() - self.pause_time)
        self.pause_time = None

    @staticmethod
    def _format_time(seconds):
        delta = timedelta(seconds=seconds)
        if delta.total_seconds() < 1.:
            return f'{delta.microseconds/1000.} ms'
        elif delta.total_seconds() < 60.:
            return f'{delta.total_seconds():.3} s'
        else:
            return str(delta)

    def stop_and_log(self):
        seconds = time() - self.begin - self.paused_seconds
        delta_str = Timer._format_time(seconds)
        logging.info(f'{self.label} took {delta_str}')
        return seconds

    def stop_and_log_average(self, iterations):
        end = time()
        seconds = (end-self.begin-self.paused_seconds)/iterations

        delta_str = Timer._format_time(seconds)
        logging.info(f'{self.label} took {delta_str}')
        return seconds

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.stop_and_log()
