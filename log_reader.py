from __future__ import print_function

import csv
import numpy as np
import re
import sys


def read(log_file):
    log_handle = open(log_file, 'r')
    reader = csv.reader(log_handle)
    reader.next()

    values = []
    for (num_row, row) in enumerate(reader):
        (key, label, value) = (
            row[0],
            int(row[1]),
            np.fromstring(re.sub('[\[\]]', '', row[2].replace('\n', '')), sep=' '))
        values.append((key, label, value))
        print('\033[2K\rReading log file #%d: %s (%s)' % (num_row, key, label), end='')
        sys.stdout.flush()
    values = map(np.vstack, zip(*values))
    log_handle.close()
    print('')

    return values
