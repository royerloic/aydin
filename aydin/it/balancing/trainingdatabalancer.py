import math
import random

import numpy


class TrainingDataBalancer:
    """
    Training data balancer : Avoids having an excess of one class (y values) in the training data

    """

    def __init__(
        self,
        total_entries,
        number_of_bins,
        keep_ratio=1,
        tolerance=0.01,
        is_active=True,
    ):
        """

        """
        self.is_active = is_active
        self.total_entries = total_entries
        self.number_of_bins = number_of_bins
        self.keep_ratio = keep_ratio
        self.tolerance = tolerance
        self.histogram = numpy.ones(number_of_bins)

        self.max_entries_per_bin = int(
            math.ceil(self.total_entries / self.number_of_bins)
        )

    def add_entry(self, array):

        mean = numpy.mean(array)
        index = int(self.number_of_bins * mean)
        index = min(len(self.histogram) - 1, index)

        value = self.histogram[index]

        if (not self.is_active) or (
            (value < self.max_entries_per_bin * (1 + self.tolerance))
            and random.random() <= self.keep_ratio
        ):
            self.histogram[index] = value + 1
            return True
        else:
            return False

    def get_histogram_as_string(self):

        return (
            '│'
            + ''.join(
                (_value_to_fill_char(x / self.max_entries_per_bin))
                for x in self.histogram
            )
            + '│'
        )

    def percentage_kept(self):
        return min(1, self.histogram.sum() / self.total_entries)


def _value_to_fill_char(x):
    if x <= 0.01:
        return ' '
    if x < 0.25:
        return '·'
    if x < 0.50:
        return '░'
    elif x < 0.75:
        return '▒'
    elif x < 1.00:
        return '▓'
    elif x >= 1:
        return '█'  # ■
