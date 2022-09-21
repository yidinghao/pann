import time
from contextlib import contextmanager
from typing import Callable, Generator


@contextmanager
def timer(message: str) -> Generator[Callable[[None], float], None, None]:
    """
    A timer that measures the time spent running code within a with-
    block.

    :param message: A message to be printed when the timer starts.
    :return: A generator that yields a function that returns the current
        time elapsed.
    """
    print_delay(message)

    # Start timer
    start_time = time.time()
    yield lambda: time.time() - start_time

    # Stop timer
    end_time = time.time()
    elapsed = end_time - start_time
    print_delay("Done. Time elapsed: {:.3f} seconds".format(elapsed))


def print_delay(*args, delay: float = .05, **kwargs):
    time.sleep(delay)
    print(*args, **kwargs)
    time.sleep(delay)
