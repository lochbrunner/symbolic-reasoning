from time import time
from datetime import timedelta
import logging


class Timer:
    def __init__(self, label):
        self.label = label
        self.begin = time()

    def start(self):
        self.begin = time()

    @staticmethod
    def _format_time(seconds):
        delta = timedelta(seconds=seconds)
        if delta.total_seconds() < 1.:
            return f'{delta.microseconds/1000.} ms'
        elif delta.total_seconds() < 60.:
            return f'{delta.total_seconds()} s'
        else:
            return str(delta)

    def stop_and_log(self):
        end = time()
        delta_str = Timer._format_time(end-self.begin)
        logging.info(f'{self.label} took {delta_str}')

    def stop_and_log_average(self, iterations):
        end = time()
        seconds = (end-self.begin)/iterations

        delta_str = Timer._format_time(seconds)

        logging.info(f'{self.label} took {delta_str}')
