import math
import sys
from io import TextIOWrapper

from decorator import contextmanager


# Globals:


# Initialise:
___current_section = ''
___depth = 0
___max_depth = math.inf


def __native_print(*args, sep=' ', end='\n', file=None):
    print(*args, sep=sep, end=end, file=sys.__stdout__)


## PUBLIC API BELOW:


def set_log_max_depth(max_depth):
    global ___max_depth
    ___max_depth = max(0, max_depth - 1)


def lprint(*args, sep=' ', end='\n'):
    global ___depth

    if ___depth <= ___max_depth:
        level = min(___max_depth, ___depth)
        __native_print('│' * level + '├ ', end='')
        __native_print(*args, sep=sep)


@contextmanager
def lsection(section_header: str, intersept_print=False):
    global ___current_section
    global ___depth

    ___current_section = section_header

    if ___depth <= ___max_depth:
        __native_print('│' * ___depth + '├╗ ' + section_header)  # ≡
    ___depth += 1

    yield

    ___depth -= 1
    if ___depth <= ___max_depth:
        __native_print('│' * (___depth + 1))
