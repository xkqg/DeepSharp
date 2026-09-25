# The verbs

Every step a pipeline file may hold: what it does, and what it takes. This page is written by the
steps themselves, so it is exactly what the library reads, and a step that changes changes it too.

A step is one JSON object in the file's `declaration`, named by its `step`, with its parameters
beside it. A key a step does not take is refused, and so is a word it does not know.

| verb | what it does |
|---|---|
| [`declare`](#declare) | Names the columns that take part, says what each holds, and decides what becomes of the rest. |
| [`drop.columns`](#dropcolumns) | Leaves columns out from here on. |
| [`drop.gaps`](#dropgaps) | Drops every row that has a gap in any of the named columns, before the rows are divided. |
| [`drop.warmup`](#dropwarmup) | Drops the rows at the start that an indicator cannot yet speak for. |
| [`encode`](#encode) | Writes a column of words down as numbers, using the categories the training rows held. |
| [`encode.categories`](#encodecategories) | Writes every column that stands for a group down as numbers, each by the categories the training rows held. |
| [`evidence.correlation`](#evidencecorrelation) | Sets out the rows a correlation between columns is drawn from, on the rows the split trains on, and how many it kept. |
| [`evidence.profile`](#evidenceprofile) | Profiles the columns where it stands, on the rows the split trains on, and names the step that answers each thing it finds. |
| [`feature.add`](#featureadd) | Adds a column worked out from two others by plain arithmetic. |
| [`feature.cyclical`](#featurecyclical) | Writes a moment in time as a place on a circle, so that the ends of a cycle meet. |
| [`feature.indicator`](#featureindicator) | Adds a market indicator worked out from the rows that came before: an average, a strength index, a band. |
| [`feature.timeParts`](#featuretimeparts) | Takes a moment in time apart into the pieces people reason with: an hour, a weekday, a month. |
| [`fill.missing`](#fillmissing) | Fills the gaps in a column the named way, with a value learned from the training rows, and marks where they were. |
| [`fill.nan`](#fillnan) | Deals with a value that is not a number a model can use; refusing it is the default. |
| [`maths`](#maths) | Pulls a column into another shape by arithmetic that learns nothing: a logarithm, a root, a reciprocal. |
| [`normalise`](#normalise) | Brings a column onto a comparable scale, by numbers learned from the training rows. |
| [`normalise.row`](#normaliserow) | Brings each row onto a comparable scale across the columns that make it up, learning nothing. |
| [`order.by`](#orderby) | Puts the rows in order by one or more columns, smallest first, for the steps that read the rows before a row. |
| [`outliers.clip`](#outliersclip) | Holds the extreme values of a column to bounds learned from the training rows. |
| [`read.csv`](#readcsv) | Reads the rows from a comma-separated file. |
| [`read.rows`](#readrows) | Takes rows that are handed in rather than opened: a table already in memory, a reader over a query. |
| [`split.atRandom`](#splitatrandom) | Divides the rows at random, the same way every time for the same seed. |
| [`split.byTime`](#splitbytime) | Divides the rows by when they happened: the earliest to learn from, the latest to be measured on. |
| [`split.stratified`](#splitstratified) | Divides the rows at random while keeping the mixture of one column the same in every part. |
| [`target`](#target) | Names the column a model is asked to predict, which is handed over apart from the numbers it is shown. |
| [`target.distribution`](#targetdistribution) | Names the columns a model is asked to predict as one answer: how a whole is divided among them, in their order. |

## `declare`

Names the columns that take part, says what each holds, and decides what becomes of the rest.

```json
{"step":"declare","remainder":"drop","columns":[{"name":"column","kind":"number","optional":false}]}
```

| key | holds | a new block starts with |
|---|---|---|
| `remainder` | one of `drop`, `keep` or `refuse` | `"drop"` |
| `columns` | a list of columns, each with a `name`, a `kind` that is one of `text`, `number`, `integer`, `boolean`, `timestamp` or `category`, and whether it is `optional` | `[{"name":"column","kind":"number","optional":false}]` |

- **`remainder`**: What becomes of the columns the schema does not name: dropped, kept as text, or refused.
- **`columns`**: The columns that take part, what each holds, and whether the source may lack it.

Means what it says from version 1 of the file.

## `drop.columns`

Leaves columns out from here on.

```json
{"step":"drop.columns","columns":["column"]}
```

| key | holds | a new block starts with |
|---|---|---|
| `columns` | a list of the names of columns holding anything, each named once | `["column"]` |

- **`columns`**: The columns to leave out from here on.

Means what it says from version 2 of the file.

## `drop.gaps`

Drops every row that has a gap in any of the named columns, before the rows are divided.

```json
{"step":"drop.gaps","columns":["column"]}
```

| key | holds | a new block starts with |
|---|---|---|
| `columns` | a list of the names of columns holding anything, each named once | `["column"]` |

- **`columns`**: The columns a row may not have a gap in.

Means what it says from version 2 of the file.

## `drop.warmup`

Drops the rows at the start that an indicator cannot yet speak for.

```json
{"step":"drop.warmup","atMost":1000}
```

| key | holds | a new block starts with |
|---|---|---|
| `atMost` | a whole number, at least 0 | `1000` |

- **`atMost`**: The most rows this may drop from the start; beyond it the run stops rather than shrink the data to nothing.

Means what it says from version 2 of the file.

## `encode`

Writes a column of words down as numbers, using the categories the training rows held.

```json
{"step":"encode","column":"column","as":"onehot","unseen":"reserve"}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of a column holding anything | `"column"` |
| `as` | one of `onehot` or `ordinal` | `"onehot"` |
| `unseen` | one of `reserve` or `refuse` | `"reserve"` |

- **`column`**: The column of words to write down as numbers.
- **`as`**: How the categories are written down: one column per category, or one column of places.
- **`unseen`**: What happens to a category the training rows never held: a place kept for it, or a refusal.

Means what it says from version 1 of the file.

## `encode.categories`

Writes every column that stands for a group down as numbers, each by the categories the training rows held.

```json
{"step":"encode.categories","as":"onehot","unseen":"reserve"}
```

| key | holds | a new block starts with |
|---|---|---|
| `as` | one of `onehot` or `ordinal` | `"onehot"` |
| `unseen` | one of `reserve` or `refuse` | `"reserve"` |

- **`as`**: How each category is written down: one column per category, or one column of places.
- **`unseen`**: What happens to a category the training rows never held: a place kept for it, or a refusal.

Means what it says from version 2 of the file.

## `evidence.correlation`

Sets out the rows a correlation between columns is drawn from, on the rows the split trains on, and how many it kept.

```json
{"step":"evidence.correlation","columns":["left","right"],"shown":"drawn"}
```

| key | holds | a new block starts with |
|---|---|---|
| `columns` | a list of the names of columns holding a number, a whole number or true or false, each named once | `["left","right"]` |
| `shown` | one of `drawn` or `numbers` | `"drawn"` |

- **`columns`**: The columns to correlate with one another: two or more, each holding numbers.
- **`shown`**: How it is shown: drawn as a coloured grid, or as the numbers themselves.

Means what it says from version 2 of the file.

## `evidence.profile`

Profiles the columns where it stands, on the rows the split trains on, and names the step that answers each thing it finds.

```json
{"step":"evidence.profile","columns":["column"]}
```

| key | holds | a new block starts with |
|---|---|---|
| `columns` | a list of the names of columns holding anything, each named once; may be left out | `["column"]` |

- **`columns`**: The columns to profile; left out, every column where the step stands.

Means what it says from version 2 of the file.

## `feature.add`

Adds a column worked out from two others by plain arithmetic.

```json
{"step":"feature.add","column":"feature","left":"left","arithmetic":"minus","right":"right"}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of the column it makes | `"feature"` |
| `left` | the name of a column holding a number, a whole number or true or false | `"left"` |
| `arithmetic` | one of `plus`, `minus`, `times` or `dividedby` | `"minus"` |
| `right` | the name of a column holding a number, a whole number or true or false | `"right"` |

- **`column`**: What the new column is called.
- **`left`**: The column on the left of the arithmetic.
- **`arithmetic`**: What is done with the two columns: added, subtracted, multiplied or divided.
- **`right`**: The column on the right of the arithmetic.

Means what it says from version 1 of the file.

## `feature.cyclical`

Writes a moment in time as a place on a circle, so that the ends of a cycle meet.

```json
{"step":"feature.cyclical","column":"when","period":"monthofyear","form":"signed"}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of a column holding a moment in time | `"when"` |
| `period` | one of `hourofday`, `dayofweek`, `dayofmonth` or `monthofyear` | `"monthofyear"` |
| `form` | one of `signed`, `unit` or `splitsign` | `"signed"` |

- **`column`**: The column holding the moment in time.
- **`period`**: Which cycle the moment is placed on: the hour of the day, the day of the week or of the month, the month of the year.
- **`form`**: How the two values are written down: as they are, shifted between nothing and one, or split into how far up and how far down.

Means what it says from version 1 of the file.

## `feature.indicator`

Adds a market indicator worked out from the rows that came before: an average, a strength index, a band.

```json
{"step":"feature.indicator","column":"indicator","indicator":"sma","period":14,"columns":["close"]}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of the column it makes | `"indicator"` |
| `indicator` | one of `sma`, `ema`, `rsi`, `atr`, `adx`, `cci`, `williamsr`, `obv`, `macd`, `bollingerbands`, `stochastic` or `vwap` | `"sma"` |
| `period` | a whole number, at least 1 | `14` |
| `columns` | a list of the names of columns holding a number, a whole number or true or false | `["close"]` |

- **`column`**: What the new column is called; an indicator with several parts adds a suffix for each.
- **`indicator`**: Which indicator, worked out from the rows that came before.
- **`period`**: The look-back in rows, for an indicator that takes one.
- **`columns`**: The columns it reads, in the order the indicator expects them.

Means what it says from version 1 of the file.

## `feature.timeParts`

Takes a moment in time apart into the pieces people reason with: an hour, a weekday, a month.

```json
{"step":"feature.timeParts","column":"when","asCategories":true,"parts":["month"]}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of a column holding a moment in time | `"when"` |
| `asCategories` | `true` or `false` | `true` |
| `parts` | a list of one or more of `minute`, `hour`, `dayofweek`, `dayofmonth`, `month`, `quarter`, `season` or `year` | `["month"]` |

- **`column`**: The column holding the moment in time.
- **`asCategories`**: Whether the pieces stand for a group, which they do unless the order is the point.
- **`parts`**: Which pieces of the moment become columns of their own.

Means what it says from version 1 of the file.

## `fill.missing`

Fills the gaps in a column the named way, with a value learned from the training rows, and marks where they were.

```json
{"step":"fill.missing","column":"column","with":"median"}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of a column holding a number or a whole number | `"column"` |
| `with` | one of `"mean"`, `"median"`, `"zero"`, `"previous"`, `"refuse"` or `{"kind": "constant", "value": a number}` | `"median"` |
| `refuseAbove` | a share, from nought to one; left out, there is none | left out |

- **`column`**: The column with gaps in it.
- **`with`**: What goes in the gaps, learned from the training rows: mean, median, zero, previous, constant, or refuse.
- **`refuseAbove`**: The share of the training rows that may be gaps and still be filled. Above it the column is not filled, and the column that says where the gaps were speaks for it. Left out, every share is filled.

Means what it says from version 1 of the file.

## `fill.nan`

Deals with a value that is not a number a model can use; refusing it is the default.

```json
{"step":"fill.nan","column":"column","with":"refuse"}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of a column holding a number | `"column"` |
| `with` | one of `"refuse"`, `"mean"`, `"median"`, `"zero"` or `{"kind": "constant", "value": a number}` | `"refuse"` |

- **`column`**: The column to watch for values that are not numbers a model can use.
- **`with`**: What happens to a value that is not a number: refuse, which is the default and usually the right answer, or mean, median, zero or constant.

Means what it says from version 1 of the file.

## `maths`

Pulls a column into another shape by arithmetic that learns nothing: a logarithm, a root, a reciprocal.

```json
{"step":"maths","column":"column","maths":"log1p","into":"column"}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of a column holding a number, a whole number or true or false | `"column"` |
| `maths` | one of `log`, `log1p`, `reciprocal`, `sqrt`, `square`, `arcsin`, `abs` or `sign` | `"log1p"` |
| `into` | the name of the column it makes; left out, the step decides | `"column"` |

- **`column`**: The column to pull into another shape.
- **`maths`**: Which shape: a logarithm, a root, a reciprocal, a square, an arcsine, the magnitude or the sign.
- **`into`**: What the result is called; the same column, unless a file says otherwise.

Means what it says from version 1 of the file.

## `normalise`

Brings a column onto a comparable scale, by numbers learned from the training rows.

```json
{"step":"normalise","column":"column","scale":"standard","outOfRange":"pass"}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of a column holding a number, a whole number or true or false | `"column"` |
| `scale` | one of `standard`, `minmax`, `maxabs`, `robust`, `quantile` or `power` | `"standard"` |
| `outOfRange` | one of `pass`, `clip` or `refuse` | `"pass"` |

- **`column`**: The column to bring onto a comparable scale.
- **`scale`**: Which kind of scaling: what the fit learns from the training rows.
- **`outOfRange`**: What happens to a value outside the range the fit learned: let it through, hold it at the edge, or refuse.

Means what it says from version 1 of the file.

## `normalise.row`

Brings each row onto a comparable scale across the columns that make it up, learning nothing.

```json
{"step":"normalise.row","norm":"l2","columns":["left","right"]}
```

| key | holds | a new block starts with |
|---|---|---|
| `norm` | one of `l1`, `l2` or `max` | `"l2"` |
| `columns` | a list of the names of columns holding a number, a whole number or true or false, each named once | `["left","right"]` |

- **`norm`**: How the row's size is measured: the sum of the magnitudes, the length, or the largest.
- **`columns`**: The columns that make up the row, scaled together.

Means what it says from version 1 of the file.

## `order.by`

Puts the rows in order by one or more columns, smallest first, for the steps that read the rows before a row.

```json
{"step":"order.by","columns":["when"]}
```

| key | holds | a new block starts with |
|---|---|---|
| `columns` | a list of the names of columns holding a moment in time, a whole number or a number, each named once | `["when"]` |

- **`columns`**: The columns the rows are put in order by: the first decides, each next one decides between rows the ones before call equal.

Means what it says from version 2 of the file.

## `outliers.clip`

Holds the extreme values of a column to bounds learned from the training rows.

```json
{"step":"outliers.clip","column":"column","bounds":"iqr","at":1.5,"outlier":"clip"}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of a column holding a number, a whole number or true or false | `"column"` |
| `bounds` | one of `quantile`, `sigma` or `iqr` | `"iqr"` |
| `at` | a number above 0 | `1.5` |
| `outlier` | one of `clip`, `blank` or `refuse` | `"clip"` |

- **`column`**: The column whose extremes are held.
- **`bounds`**: How the bounds are worked out: by quantile, by spread, or by the middle half.
- **`at`**: How far out the bounds sit: a share for a quantile, a multiple of the spread or the middle half otherwise.
- **`outlier`**: What happens to a value outside the bounds: held at the edge, made a gap, or refused.

Means what it says from version 1 of the file.

## `read.csv`

Reads the rows from a comma-separated file.

```json
{"step":"read.csv","path":"data.csv"}
```

| key | holds | a new block starts with |
|---|---|---|
| `path` | the path of a file; a relative one is read from the pipeline's folder | `"data.csv"` |

- **`path`**: Where the comma-separated file will be, when the pipeline runs.

Means what it says from version 1 of the file.

## `read.rows`

Takes rows that are handed in rather than opened: a table already in memory, a reader over a query.

```json
{"step":"read.rows","description":"rows handed in"}
```

| key | holds | a new block starts with |
|---|---|---|
| `description` | words | `"rows handed in"` |

- **`description`**: What the rows are, for whoever reads the file later.

Means what it says from version 1 of the file.

## `split.atRandom`

Divides the rows at random, the same way every time for the same seed.

```json
{"step":"split.atRandom","train":0.7,"validation":0.15,"test":0.15,"predict":0,"seed":20260923}
```

| key | holds | a new block starts with |
|---|---|---|
| `train` | the share the model learns from: above nought, at most one | `0.7` |
| `validation` | the share used while choosing between models: nought to one | `0.15` |
| `test` | the share kept back until the end: above nought, at most one | `0.15` |
| `predict` | the share held back to predict on: nought to one, and none when it is left out | `0` |
| `seed` | a whole number | `20260923` |

- **`train`**: How the rows are shared out: training, validation, test and a part to predict on, which together make the whole.
- **`seed`**: The number that makes the shuffle repeatable: the same seed deals the same rows the same way.

Means what it says from version 2 of the file.

## `split.byTime`

Divides the rows by when they happened: the earliest to learn from, the latest to be measured on.

```json
{"step":"split.byTime","column":"when","train":0.7,"validation":0.15,"test":0.15,"predict":0}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of a column holding a moment in time, a whole number or a number | `"when"` |
| `train` | the share the model learns from: above nought, at most one | `0.7` |
| `validation` | the share used while choosing between models: nought to one | `0.15` |
| `test` | the share kept back until the end: above nought, at most one | `0.15` |
| `predict` | the share held back to predict on: nought to one, and none when it is left out | `0` |
| `gap` | a whole number, at least 0; left out, 0 | left out |

- **`column`**: The column that says when a row happened.
- **`train`**: How the rows are shared out: training, validation, test and a part to predict on, which together make the whole.
- **`gap`**: How many of the last moments of every part are kept apart, fitted on by nothing and handed to nothing: at least as many as the rows an answer reads ahead. Left out, none.

Means what it says from version 2 of the file.

## `split.stratified`

Divides the rows at random while keeping the mixture of one column the same in every part.

```json
{"step":"split.stratified","column":"class","train":0.7,"validation":0.15,"test":0.15,"predict":0,"seed":20260923}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of a column holding anything | `"class"` |
| `train` | the share the model learns from: above nought, at most one | `0.7` |
| `validation` | the share used while choosing between models: nought to one | `0.15` |
| `test` | the share kept back until the end: above nought, at most one | `0.15` |
| `predict` | the share held back to predict on: nought to one, and none when it is left out | `0` |
| `seed` | a whole number | `20260923` |

- **`column`**: The column whose mixture of values is kept the same in every part.
- **`train`**: How the rows are shared out: training, validation, test and a part to predict on, which together make the whole.
- **`seed`**: The number that makes the shuffle repeatable: the same seed deals the same rows the same way.

Means what it says from version 2 of the file.

## `target`

Names the column a model is asked to predict, which is handed over apart from the numbers it is shown.

```json
{"step":"target","column":"answer"}
```

| key | holds | a new block starts with |
|---|---|---|
| `column` | the name of a column holding anything | `"answer"` |

- **`column`**: The column a model is asked to predict, handed over apart from the numbers it is shown.

Means what it says from version 1 of the file.

## `target.distribution`

Names the columns a model is asked to predict as one answer: how a whole is divided among them, in their order.

```json
{"step":"target.distribution","columns":["share1","share2"]}
```

| key | holds | a new block starts with |
|---|---|---|
| `columns` | a list of the names of columns holding a number, a whole number or true or false, each named once | `["share1","share2"]` |
| `scaleBy` | the name of a column holding a number, a whole number or true or false; may be left out | left out |

- **`columns`**: The columns the answer is divided among, in their order: at least two.
- **`scaleBy`**: The column saying how many the shares are shares of, so predictions come back as how many fell in each; left out, they come back as shares.

Means what it says from version 2 of the file.
