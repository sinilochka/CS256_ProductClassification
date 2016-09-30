__author__ = 'annasinilo'

import math
from sys import argv


def read_file_and_get_count(filename, result_column):
    file = open(filename)
    header = file.readline()
    result = {}
    for line in file:
        columns = line.strip().split(',')
        datetime = columns[0]
        try:
            count = int(columns[result_column])
        except Exception:
            continue
        result[datetime] = count
    file.close()
    return result

print argv
real_results = read_file_and_get_count(argv[1], 11)
print real_results['2012-12-19 23:00:00']
submission_results = read_file_and_get_count(argv[2], 1)
print submission_results['2012-12-19 23:00:00']

metric = 0
n = len(real_results)
for datetime in real_results:
    x = math.log(submission_results[datetime] + 1) - math.log(real_results[datetime] + 1)
    metric += x ** 2
print math.sqrt(metric / n)