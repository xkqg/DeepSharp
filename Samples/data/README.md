# Sample data

Two published datasets, fetched once and kept here so a sample behaves the same on every machine and on a
day the network is having trouble. Neither was invented for this repository, and that is the point: made-up
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
backwards. Together the two files cover both halves of what a pipeline has to get right.
