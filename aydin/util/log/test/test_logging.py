from aydin.util.combinatorics import closest_product

import logging

from aydin.util.log.logging import lprint, lsection


def test_logging():

    lprint('Test')

    with lsection('a section'):
        lprint('a line')
        lprint('another line')
        lprint('we are done')

        with lsection('a subsection'):
            lprint('another line')
            lprint('we are done')

    lprint('test is finished...')
