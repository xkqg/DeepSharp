# Sample data

Three published datasets, fetched once and kept here so a sample behaves the same on every machine and on a
day the network is having trouble. None was invented for this repository, and that is the point: made-up
numbers demonstrate the happy path and nothing else.

## `titanic.csv`

From the [seaborn-data](https://github.com/mwaskom/seaborn-data) collection
(`https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv`), the version of the Titanic
passenger list that ships with the seaborn plotting library. 891 rows, 15 columns.

What it is here for, measured rather than assumed:

- **Gaps of two very different sizes.** `age` is empty in 177 rows (19.9 per cent) — the honest case for
  filling. `deck` is empty in 688 (77.2 per cent), where filling would manufacture most of the column.
- **Categories**, one of which has gaps of its own: `embarked` is `C`, `Q`, `S`, and empty twice.
- **A boolean dialect**: `adult_male` and `alone` hold `True` and `False`, capitalised.
- **Columns that restate other columns**: `alive` is `survived` in words, `class` is `pclass` in words. A
  pipeline that carries everything in the file hands the model its own answer.
- **No time column at all**, so it cannot be split by time — which is exactly why no split is the default.

## `apple.csv`

From the [plotly datasets](https://github.com/plotly/datasets) collection
(`https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv`): daily Apple prices
with a date, open, high, low, close, volume, three derived columns and a category. 506 rows.

This is the other shape: a series in time, where the split runs along the date and a feature may only look
backwards. Together with the passenger list it covers both halves of what a pipeline has to get right: rows in no
order, and rows in time.

## `bikes.csv`

From the [Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) in the UCI Machine
Learning Repository (doi:10.24432/C5W894), by Hadi Fanaee-T of the University of Porto, under the
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) licence: two years of a bike-sharing
scheme in Washington, D.C., with the weather and the season of each day. Using it in a publication asks for this
citation: Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge",
Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.

The publisher ships two tables, one row a day and one row an hour. This file is the two put together, one row a day,
and nothing in it is computed but the arrangement: every column of the daily table except the record number and the
split into casual and registered riders, then `h00` to `h23`, the bikes rented in each hour of that day, taken from
the hourly table. An hour the hourly table does not list is an hour nobody rented a bike, and it is written as 0.
731 rows, 37 columns.

What it is here for, measured rather than assumed:

- **An answer that is a whole divided into parts.** The twenty-four hours of a day add up to its `cnt` on every one
  of the 731 days, so the hours divided by the day's total are one answer of twenty-four shares that sum to one, and
  `cnt` is what brings the shares back as bikes.
- **Hours with nothing in them.** 655 days have a rental in every hour; on the other 76 some hours are empty, and on
  the day of the fewest rentals, 22 in all, twenty-three hours are.
- **Categories written as digits.** `season` is 1 to 4 and `weathersit` 1 to 3, and neither is a quantity: they are
  declared as categories rather than read as numbers.
- **True and false as noughts and ones**, in `holiday` and `workingday`.
- **Weather the publisher already scaled.** `temp`, `atemp`, `hum` and `windspeed` are divided by their largest
  possible values, so they arrive between nought and one.
