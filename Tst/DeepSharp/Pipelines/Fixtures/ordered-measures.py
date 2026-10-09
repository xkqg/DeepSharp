# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# The numbers the measures of a distribution are held to: the earth mover's distance, the Kullback-Leibler divergence and
# the ranked probability score of a few rows of answers and predictions, each row in its own units and compared as
# shares of its own total. Its output, when it was run, is ordered-measures.txt beside it.
#
# Run with scipy 1.18.1, numpy 2.5.3, xskillscore 0.0.29 and xarray 2026.9.0 on Python 3.14.7:
#
#     python ordered-measures.py
#
# emd is scipy's Wasserstein distance between the two rows over the band positions 0, 1, 2, ..., in bands; kl is scipy's
# entropy of the answers relative to the predictions, natural logarithm, no number added to a nought; rps is xskillscore's
# ranked probability score of the predicted shares against the answered ones, summed over the bands as Weigel's is. Each
# is the mean over the rows. With a remainder, the last answer is what is left of the whole: emd and rps compare the
# bands alone, as shares of what they hold together, and kl compares every answer.

import platform

import numpy as np
import scipy
import xarray as xr
import xskillscore as xs
from scipy import stats


def emd(actual, predicted):
    places = np.arange(len(actual[0]))
    return float(np.mean([stats.wasserstein_distance(places, places, p, a) for a, p in zip(actual, predicted)]))


def kl(actual, predicted):
    return float(np.mean([stats.entropy(a, p) for a, p in zip(actual, predicted)]))


def rps(actual, predicted):
    actual = np.array(actual, dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float64)
    held = xr.DataArray(actual / actual.sum(axis=1, keepdims=True), dims=["row", "category"])
    said = xr.DataArray(predicted / predicted.sum(axis=1, keepdims=True), dims=["row", "category"])
    return float(xs.rps(held, said, category_edges=None, dim="row", input_distributions="p"))


def show(name, actual, predicted, remainder=False):
    bands = [row[:-1] for row in actual] if remainder else actual
    said = [row[:-1] for row in predicted] if remainder else predicted
    print(f"{name} emd: {emd(bands, said)!r}")
    print(f"{name} kl:  {kl(actual, predicted)!r}")
    print(f"{name} rps: {rps(bands, said)!r}")


print(f"scipy {scipy.__version__}, numpy {np.__version__}, xskillscore {xs.__version__}, xarray {xr.__version__}, Python {platform.python_version()}")

show(
    "ordered",
    [[2, 5, 3, 0], [1, 1, 1, 1], [0, 0, 4, 4], [3, 1, 0, 0]],
    [[1.5, 4, 3.5, 1], [0.5, 2, 1, 0.5], [0, 1, 3, 4], [2.5, 1.2, 0.3, 0]])

show(
    "nought where held",
    [[1, 1, 0, 0], [2, 2, 0, 0]],
    [[2, 0, 1, 1], [1, 1, 1, 1]])

show(
    "remainder",
    [[2, 5, 3, 0, 2], [1, 1, 1, 1, 4], [0, 0, 4, 4, 0]],
    [[1.5, 4, 3.5, 1, 2], [0.5, 2, 1, 0.5, 4], [0, 1, 3, 4, 0.5]],
    remainder=True)
