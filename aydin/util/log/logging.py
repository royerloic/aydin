import math
import platform
import sys
import time

from decorator import contextmanager


# Globals:


# Initialise:
___current_section = ''
___depth = 0
___max_depth = math.inf
___log_elapsed_time = True


def __native_print(*args, sep=' ', end='\n', file=None):
    print(*args, sep=sep, end=end, file=sys.__stdout__)


# Define special characters:

__vl__ = '│'  # 'Vertical Line'
__br__ = '├'  # 'Branch Right'
__bd__ = '├╗'  # 'Branch Down'
__tb__ = '┴'  # 'Terminate Branch'
__la__ = '«'  # 'Left Arrow'

#  Windows terminal is dumb. We can't use our fancy characters from Yesteryears, sad:
if platform.system() == "Windows":
    __vl__ = '|'
    __br__ = '|->'
    __bd__ = '|\ '
    __tb__ = '-'
    __la__ = '<<'


## PUBLIC API BELOW:


def set_log_elapsed_time(log_elapsed_time: bool):
    ___log_elapsed_time = log_elapsed_time


def set_log_max_depth(max_depth: int):
    global ___max_depth
    ___max_depth = max(0, max_depth - 1)


def lprint(*args, sep=' ', end='\n'):
    global ___depth

    if ___depth <= ___max_depth:
        level = min(___max_depth, ___depth)
        __native_print(__vl__ * level + __br__ + ' ', end='')
        __native_print(*args, sep=sep)


@contextmanager
def lsection(section_header: str, intersept_print=False):
    global ___current_section
    global ___depth

    ___current_section = section_header

    if ___depth + 1 <= ___max_depth:
        __native_print(__vl__ * ___depth + __bd__ + ' ' + section_header)  # ≡
    ___depth += 1

    start = time.time()
    yield
    stop = time.time()

    ___depth -= 1
    if ___depth + 1 <= ___max_depth:

        if ___log_elapsed_time:
            elapsed = stop - start

            if elapsed < 0.001:
                __native_print(
                    __vl__ * (___depth + 1)
                    + __tb__
                    + __la__
                    + f' {elapsed * 1000 * 1000:.2f} microseconds'
                )
            elif elapsed < 1:
                __native_print(
                    __vl__ * (___depth + 1)
                    + __tb__
                    + __la__
                    + f' {elapsed * 1000:.2f} milliseconds'
                )
            elif elapsed < 60:
                __native_print(
                    __vl__ * (___depth + 1)
                    + __tb__
                    + __la__
                    + f' {elapsed:.2f} seconds'
                )
            elif elapsed < 60 * 60:
                __native_print(
                    __vl__ * (___depth + 1)
                    + __tb__
                    + __la__
                    + f' {elapsed / 60:.2f} minutes'
                )
            elif elapsed < 24 * 60 * 60:
                __native_print(
                    __vl__ * (___depth + 1)
                    + __tb__
                    + __la__
                    + f' {elapsed / (60 * 60):.2f} hours'
                )

        __native_print(__vl__ * (___depth + 1))
