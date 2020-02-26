import math
import random

import numpy


class DataHistogramBalancer:
    """
    Training data balancer : Avoids having an excess of one class (y values) in the training data

    """

    def __init__(
        self,
        total_entries,
        number_of_bins,
        keep_ratio=1,
        tolerance=0.5,
        is_active=True,
        use_median=False,
    ):
        """

        """
        self.use_median = use_median
        self.is_active = is_active
        self.total_entries = total_entries
        self.number_of_bins = number_of_bins
        self.keep_ratio = keep_ratio
        self.tolerance = tolerance
        self.histogram_kept = numpy.zeros(number_of_bins)
        self.histogram_all = numpy.zeros(number_of_bins)

        self.max_entries_per_bin = int(
            math.ceil(self.total_entries / self.number_of_bins)
        )

    def add_entry(self, array):

        if self.use_median:
            intensity = numpy.median(array)
        else:
            intensity = numpy.mean(array)

        index = int(self.number_of_bins * intensity)
        index = min(len(self.histogram_kept) - 1, index)

        self.histogram_all[index] += 1

        if not self.is_active or (
            (random.random() <= self.keep_ratio)
            and (
                self.histogram_kept[index]
                < self.max_entries_per_bin * (1 + self.tolerance)
            )
        ):
            self.histogram_kept[index] += 1
            return True

        return False

    def get_histogram_kept_as_string(self):

        return (
            '│'
            + ''.join(
                (_value_to_fill_char(x / self.max_entries_per_bin))
                for x in self.histogram_kept
            )
            + '│'
        )

    def get_histogram_all_as_string(self):

        return (
            '│'
            + ''.join(
                (_value_to_fill_char(x / self.max_entries_per_bin))
                for x in self.histogram_all
            )
            + '│'
        )

    def get_histogram_dropped_as_string(self):

        return (
            '│'
            + ''.join(
                (_value_to_fill_char((a - k) / self.max_entries_per_bin))
                for k, a in zip(self.histogram_kept, self.histogram_all)
            )
            + '│'
        )

    def total_kept(self):
        return int(self.histogram_kept.sum())

    def percentage_kept(self):
        return min(1, self.histogram_kept.sum() / self.total_entries)


def _value_to_fill_char(x):
    if x <= 0.00:
        return ' '
    elif x <= 0.05:
        return '_'
    elif x <= 0.10:
        return '·'
    elif x <= 0.25:
        return '░'
    elif x <= 0.50:
        return '▒'
    elif x <= 0.75:
        return '▓'
    elif x > 0.75:
        return '█'
