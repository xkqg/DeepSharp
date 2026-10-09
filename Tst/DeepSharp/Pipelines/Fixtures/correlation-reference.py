# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# What numpy and scipy answer for the correlations CorrelationMatrixTests and ProfileRankAlertTests pin. Its output, when it
# was run, is correlation-reference.txt beside it; the numbers in the tests are copied from there, never worked out by the
# code they test.
#
# Run with Python 3.14.7, numpy 2.5.3 and scipy 1.18.1:
#
#     python -I correlation-reference.py > correlation-reference.txt
#
#   flock      five columns over ten rows with ties: x and y, e = exp(x) (the same order as x, a curve), n = -x, c = x again
#   eight      the eight rows two columns of which Spearman's tie rule is told by (average ranks, 0.9007775105401477)
#   constant   x, a column that never changes, and y: no coefficient is defined for the constant one
#   drift      a column of 0.1 ten times: its sum over ten, taken as a mean, is not 0.1, which must not make a coefficient

import platform
import warnings

import numpy as np
import scipy
from scipy import stats

warnings.simplefilter("ignore")


def show(name, matrix):
    print(name)
    for row in matrix:
        print("  [" + ", ".join(repr(float(value)) for value in row) + "],")


def spearman(columns):
    count = len(columns)
    result = np.full((count, count), np.nan)
    for one in range(count):
        for other in range(count):
            result[one, other] = stats.spearmanr(columns[one], columns[other]).statistic
    return result


print("python", platform.python_version(), "numpy", np.__version__, "scipy", scipy.__version__)

x = np.array([1, 2, 2, 3, 5, 5, 5, 9, 10, 12], dtype=float)
y = np.array([2, 1, 2, 4, 4, 6, 7, 8, 7, 11], dtype=float)
flock = [x, y, np.exp(x), -x, x.copy()]
print("flock columns: x, y, exp(x), -x, x")
print("x =", x.tolist())
print("y =", y.tolist())
show("flock pearson (numpy.corrcoef)", np.corrcoef(flock))
show("flock spearman (scipy.stats.spearmanr, average ranks)", spearman(flock))

eight_x = np.array([1, 2, 2, 3, 5, 5, 5, 9], dtype=float)
eight_y = np.array([2, 1, 2, 4, 4, 6, 7, 8], dtype=float)
print("eight spearman", repr(float(stats.spearmanr(eight_x, eight_y).statistic)))
print("eight pearson", repr(float(np.corrcoef(eight_x, eight_y)[0, 1])))
print("eight Pearson of exp(x) and y", repr(float(np.corrcoef(np.exp(eight_x), eight_y)[0, 1])))
print("eight Spearman of exp(x) and y", repr(float(stats.spearmanr(np.exp(eight_x), eight_y).statistic)))

constant = [x, np.ones(len(x)), y]
show("constant pearson (numpy.corrcoef)", np.corrcoef(constant))
show("constant spearman (scipy.stats.spearmanr)", spearman(constant))

drift = [x, np.full(len(x), 0.1), y]
show("drift pearson (numpy.corrcoef)", np.corrcoef(drift))
show("drift spearman (scipy.stats.spearmanr)", spearman(drift))
running = 0.0
for _ in range(10):
    running += 0.1
print("numpy mean of ten 0.1 is 0.1:", float(np.mean(np.full(10, 0.1))) == 0.1, "| ten additions of 0.1, over ten, is 0.1:", running / 10 == 0.1)

# The pair the profile's rank alert is tried on: a column and its exponential, and one that follows it with noise.
rows = np.arange(1, 13, dtype=float)
noise = np.array([0.3, -0.9, 0.5, 1.4, -1.1, 0.2, -0.4, 1.0, -0.7, 0.8, -1.3, 0.6])
follows = rows + 2.5 * noise
print("rows =", rows.tolist())
print("noise =", noise.tolist())
print("follows = rows + 2.5 * noise =", follows.tolist())
print("spearman(rows, follows)", repr(float(stats.spearmanr(rows, follows).statistic)))
print("spearman(rows, exp(rows))", repr(float(stats.spearmanr(rows, np.exp(rows)).statistic)))
print("pearson(rows, exp(rows))", repr(float(np.corrcoef(rows, np.exp(rows))[0, 1])))

# Two columns that have nothing to do with one another, and the same two with a gap in each of them (the rows a profile's
# alert has to use are the ones where both are there).
one = np.array([5, 12, 3, 8, 1, 10, 7, 2, 11, 4, 9, 6], dtype=float)
other = np.array([2, 9, 6, 12, 4, 1, 10, 5, 3, 11, 7, 8], dtype=float)
print("one =", one.tolist())
print("other =", other.tolist())
print("spearman(one, other)", repr(float(stats.spearmanr(one, other).statistic)))
gappy_rows = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
gappy_other = np.exp(gappy_rows)
keep = np.ones(12, dtype=bool)
keep[[2, 7]] = False
print("spearman(rows, exp(rows)) over the ten rows left", repr(float(stats.spearmanr(gappy_rows[keep], gappy_other[keep]).statistic)), "n =", int(keep.sum()))

# A third column nearest to the second of two: the follows column with its first two values swapped.
near = follows.copy()
near[[0, 1]] = near[[1, 0]]
print("near =", near.tolist())
print("spearman(near, follows)", repr(float(stats.spearmanr(near, follows).statistic)))
print("spearman(near, rows)", repr(float(stats.spearmanr(near, rows).statistic)))
