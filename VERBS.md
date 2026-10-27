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
| [`encode.piecewise`](#encodepiecewise) | Writes a column of numbers down as how far each value fills a few bins, their edges the training rows' quantiles. |
| [`evidence.correlation`](#evidencecorrelation) | Sets out the rows a correlation between columns is drawn from, on the rows the split trains on, and how many it kept. |
| [`evidence.likeness`](#evidencelikeness) | Sets out how alike the parts a split made are, column by column, and flags the ones further apart than a bound. |
| [`evidence.profile`](#evidenceprofile) | Profiles the columns where it stands, on the rows the split trains on, and names what is wrong or should not be there, with how each is answered. |
| [`evidence.report`](#evidencereport) | Names the measures a trained model is held to, the parts they are taken on and how they are shown, each in the answer's own units. |
| [`feature.add`](#featureadd) | Adds a column worked out from two others by plain arithmetic. |
| [`feature.cycle`](#featurecycle) | Writes a number as a place on a circle of a length you give, so that the ends of the cycle meet. |
| [`feature.cyclical`](#featurecyclical) | Writes a moment in time as a place on a circle, so that the ends of a cycle meet. |
| [`feature.indicator`](#featureindicator) | Adds a market indicator worked out from the rows that came before: an average, a strength index, a band. |
| [`feature.timeParts`](#featuretimeparts) | Takes a moment in time apart into the pieces people reason with: an hour, a weekday, a month. |
| [`fill.missing`](#fillmissing) | Fills the gaps in a column the named way, below the split, and marks where they were. A value no row decided is settle.gaps, above it. |
| [`fill.nan`](#fillnan) | Deals with a value that is not a number a model can use; refusing it is the default. |
| [`history`](#history) | Writes beside each row the mean of a column over the earlier training rows of its key, pulled towards the training rows' mean. |
| [`learn.ml`](#learnml) | Names the trainer from ML.NET this pipeline is declared for: which trainer, the settings it takes and the seed it repeats from. |
| [`learn.network`](#learnnetwork) | Names the network this pipeline is declared for: its layers, what moves them, what judges them, when the run stops and the engine it runs on. |
| [`maths`](#maths) | Pulls a column into another shape by arithmetic that learns nothing: a logarithm, a root, a reciprocal. |
| [`normalise`](#normalise) | Brings a column onto a comparable scale, by numbers learned from the training rows. |
| [`normalise.row`](#normaliserow) | Brings each row onto a comparable scale across the columns that make it up, learning nothing. |
| [`order.by`](#orderby) | Puts the rows in order by one or more columns, smallest first, for the steps that read the rows before a row. |
| [`outliers.clip`](#outliersclip) | Holds the extreme values of a column to bounds learned from the training rows. |
| [`read.csv`](#readcsv) | Reads the rows from a comma-separated file. |
| [`read.excel`](#readexcel) | Reads the rows from a sheet of an Excel workbook, the first unless one is named, its first row naming the columns. |
| [`read.images`](#readimages) | Reads the rows from a folder of pictures, a folder to a class: a picture a row at the size and channels the step says, the folder it stands in the answer, and a column of nought and one for each class. |
| [`read.join`](#readjoin) | Reads two comma-separated files as one source: each row of the left file beside the one row of the right file its key names. Rows served later arrive already joined. |
| [`read.json`](#readjson) | Reads the rows from a JSON file holding an array of records, one object a row, every value as the file writes it. |
| [`read.parquet`](#readparquet) | Reads the rows from an Apache Parquet file, which says what each of its columns holds. |
| [`read.rows`](#readrows) | Takes rows that are handed in rather than opened: a table already in memory, a reader over a query. |
| [`read.sounds`](#readsounds) | Reads the rows from a folder of WAV recordings, a folder to a class: a recording a row at the length, channels and rate the step says, the folder it stands in the answer, and a column of nought and one for each class. |
| [`read.texts`](#readtexts) | Reads the rows from a folder of text files, a folder to a class: a file a row, the folder it stands in the answer, and a column of nought and one for each class. |
| [`scale.given`](#scalegiven) | Scales a column into a range from bounds you give, above the split, so a feature worked out after it is worked out from scaled columns. |
| [`settle.gaps`](#settlegaps) | Settles the gaps in a column with a value no row decided, so a feature worked out from it is not a gap. |
| [`shuffle`](#shuffle) | Puts the rows in an order drawn from a seed, before they are divided, so every part holds the same mixture. |
| [`split.atRandom`](#splitatrandom) | Divides the rows at random, the same way every time for the same seed. |
| [`split.byTime`](#splitbytime) | Divides the rows by when they happened: the earliest to learn from, the latest to be measured on. |
| [`split.stratified`](#splitstratified) | Divides the rows at random while keeping the mixture of one column the same in every part. |
| [`target`](#target) | Names the column a model is asked to predict, which is handed over apart from the numbers it is shown. |
| [`target.ahead`](#targetahead) | Names an answer read from a column rows later in the declared order: the value then, or the return on the row's own value by then. |
| [`target.distribution`](#targetdistribution) | Names the columns a model is asked to predict as one answer: how a whole is divided among them, in their order. |
| [`target.labels`](#targetlabels) | Names the columns a model is asked to predict as one answer of labels, each nought or one on every row. |
| [`target.numbers`](#targetnumbers) | Names the columns a model is asked to predict as one answer of free numbers, each any finite number. |
| [`tokenize`](#tokenize) | Cuts a column of text into ids of the words the training rows held most often, every row the same number of them, for an embedding to take. |
| [`watch.reference`](#watchreference) | Keeps the training rows of the columns it watches, as counts, so rows served later can be measured against them. |

## `declare`

Names the columns that take part, says what each holds, and decides what becomes of the rest.

```json
{"step":"declare","remainder":"drop","columns":[{"name":"column","kind":"number","optional":false}]}
```

| key | holds | an example |
|---|---|---|
| `remainder` | one of `drop`, `keep` or `refuse` | `"drop"` |
| `columns` | a list of columns, each with a `name`, a `kind` that is one of `text`, `number`, `integer`, `boolean`, `timestamp`, `category` or `media`, and whether it is `optional`; a column may say it is `excluded` — named, its kind kept, and read by nothing — a category may say which kind it `was` before it became one, a timestamp the `format` its moments are written in, as .NET writes a date format, without which they are read as ISO 8601 writes them, and any column the value that is `missing` there, which is read as a gap; one column of whole numbers, a category or text may say it is the `id`, which names each row, is carried beside it and is never a feature | `[{"name":"column","kind":"number","optional":false}]` |

- **`remainder`**: What becomes of the columns the schema does not name: dropped, kept as text, or refused.
- **`columns`**: The columns that take part, what each holds, and whether the source may lack it.

Means what it says from version 1 of the file.

## `drop.columns`

Leaves columns out from here on.

```json
{"step":"drop.columns","columns":["column"]}
```

| key | holds | an example |
|---|---|---|
| `columns` | a list of the names of columns holding anything, each named once | `["column"]` |

- **`columns`**: The columns to leave out from here on.

Means what it says from version 2 of the file.

## `drop.gaps`

Drops every row that has a gap in any of the named columns, before the rows are divided.

```json
{"step":"drop.gaps","columns":["column"]}
```

| key | holds | an example |
|---|---|---|
| `columns` | a list of the names of columns holding anything, each named once | `["column"]` |

- **`columns`**: The columns a row may not have a gap in.

Means what it says from version 2 of the file.

## `drop.warmup`

Drops the rows at the start that an indicator cannot yet speak for.

```json
{"step":"drop.warmup","atMost":1000}
```

| key | holds | an example |
|---|---|---|
| `atMost` | a whole number, at least 0 | `1000` |

- **`atMost`**: The most rows this may drop from the start; beyond it the run stops rather than shrink the data to nothing.

Means what it says from version 2 of the file.

## `encode`

Writes a column of words down as numbers, using the categories the training rows held.

```json
{"step":"encode","column":"column","as":"onehot","unseen":"reserve"}
```

| key | holds | an example |
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

| key | holds | an example |
|---|---|---|
| `as` | one of `onehot` or `ordinal` | `"onehot"` |
| `unseen` | one of `reserve` or `refuse` | `"reserve"` |

- **`as`**: How each category is written down: one column per category, or one column of places.
- **`unseen`**: What happens to a category the training rows never held: a place kept for it, or a refusal.

Means what it says from version 2 of the file.

## `encode.piecewise`

Writes a column of numbers down as how far each value fills a few bins, their edges the training rows' quantiles.

```json
{"step":"encode.piecewise","column":"column","bins":8,"outOfRange":"pass"}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding a number, a whole number or true or false | `"column"` |
| `bins` | a whole number, at least 2 | `8` |
| `outOfRange` | one of `pass`, `clip` or `refuse` | `"pass"` |

- **`column`**: The column of numbers to write down as bins.
- **`bins`**: How many bins the training rows' quantiles divide the values into, at least two; fewer where edges fall together.
- **`outOfRange`**: What happens to a value beyond the edges the training rows set: let through the end bins, held at them, or refused.

Means what it says from version 10 of the file.

## `evidence.correlation`

Sets out the rows a correlation between columns is drawn from, on the rows the split trains on, and how many it kept.

```json
{"step":"evidence.correlation","columns":["left","right"],"shown":"drawn"}
```

| key | holds | an example |
|---|---|---|
| `columns` | a list of the names of columns holding a number, a whole number or true or false, each named once | `["left","right"]` |
| `shown` | one of `drawn` or `numbers` | `"drawn"` |
| `coefficient` | one of `pearson` or `spearman`; left out, pearson | left out |

- **`columns`**: The columns to correlate with one another: two or more, each holding numbers.
- **`shown`**: How it is shown: drawn as a coloured grid, or as the numbers themselves.
- **`coefficient`**: Which coefficient is shown: how well two columns lie on a line (pearson), or how well they keep the same order (spearman). Left out, pearson.

Means what it says from version 2 of the file.

## `evidence.likeness`

Sets out how alike the parts a split made are, column by column, and flags the ones further apart than a bound.

```json
{"step":"evidence.likeness","warnAbove":0.1,"columns":["column"]}
```

| key | holds | an example |
|---|---|---|
| `warnAbove` | a number, 0 or more | `0.1` |
| `columns` | a list of the names of columns holding anything, each named once; may be left out | `["column"]` |

- **`warnAbove`**: The distance above which two parts are flagged as unlike: a share above nought, at most one.
- **`columns`**: The columns to compare across the parts; left out, every column where it stands but the answer.

Means what it says from version 10 of the file.

## `evidence.profile`

Profiles the columns where it stands, on the rows the split trains on, and names what is wrong or should not be there, with how each is answered.

```json
{"step":"evidence.profile","columns":["column"]}
```

| key | holds | an example |
|---|---|---|
| `columns` | a list of the names of columns holding anything, each named once; may be left out | `["column"]` |
| `rankAbove` | a share, from nought to one; left out, there is none | left out |

- **`columns`**: The columns to profile; left out, every column where the step stands.
- **`rankAbove`**: How alike in order two columns of numbers may be before the profile says so: a share above nought, at most one. Two columns whose Spearman coefficient, on the rows where both hold a number, is above this in size are flagged. Left out, none is.

Means what it says from version 2 of the file.

## `evidence.report`

Names the measures a trained model is held to, the parts they are taken on and how they are shown, each in the answer's own units.

```json
{"step":"evidence.report","metrics":["rmse"],"parts":["validation","test"],"shown":["numbers"]}
```

| key | holds | an example |
|---|---|---|
| `metrics` | a list of one or more of `rmse`, `mae`, `r2`, `accuracy`, `precision`, `recall`, `confusionmatrix`, `emd`, `kl`, `rps`, `f1`, `auc`, `mape`, `brier`, `coverage` or `width` | `["rmse"]` |
| `parts` | a list of one or more of `train`, `validation` or `test` | `["validation","test"]` |
| `shown` | a list of one or more of `drawn` or `numbers` | `["numbers"]` |

- **`metrics`**: The measures, in the order they are shown: rmse, mae, r2 and mape for amounts, and the coverage and width of an interval; accuracy, precision, recall, the confusion matrix, f1, auc and brier for classes; kl for shares of a whole; emd and rps for shares in an order.
- **`parts`**: The parts they are taken on, side by side: the rows a model learns from, the rows it is chosen on and the rows it is tested on. The rows held back to predict on, and the rows a split keeps apart, are measured on by nothing.
- **`shown`**: How they are shown: drawn, as the numbers themselves, or both.

Means what it says from version 3 of the file.

## `feature.add`

Adds a column worked out from two others by plain arithmetic.

```json
{"step":"feature.add","column":"feature","left":"left","arithmetic":"minus","right":"right"}
```

| key | holds | an example |
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

## `feature.cycle`

Writes a number as a place on a circle of a length you give, so that the ends of the cycle meet.

```json
{"step":"feature.cycle","column":"age","length":7,"form":"signed"}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding a number or a whole number | `"age"` |
| `length` | a number above 0 | `7` |
| `form` | one of `signed`, `unit` or `splitsign` | `"signed"` |

- **`column`**: The column holding the number.
- **`length`**: How long one turn is, in the column's own units: 7 for an age in days on a week. Said by you, never worked out from the rows.
- **`form`**: How the two values are written down: as they are, shifted between nothing and one, or split into how far up and how far down.

Means what it says from version 8 of the file.

## `feature.cyclical`

Writes a moment in time as a place on a circle, so that the ends of a cycle meet.

```json
{"step":"feature.cyclical","column":"when","period":"monthofyear","form":"signed"}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding a moment in time | `"when"` |
| `period` | one of `hourofday`, `dayofweek`, `dayofmonth`, `monthofyear`, `dayofyear` or `season` | `"monthofyear"` |
| `form` | one of `signed`, `unit` or `splitsign` | `"signed"` |

- **`column`**: The column holding the moment in time.
- **`period`**: Which cycle the moment is placed on: the hour of the day, the day of the week, of the month or of the year, the month of the year, the season.
- **`form`**: How the two values are written down: as they are, shifted between nothing and one, or split into how far up and how far down.

Means what it says from version 1 of the file.

## `feature.indicator`

Adds a market indicator worked out from the rows that came before: an average, a strength index, a band.

```json
{"step":"feature.indicator","column":"indicator","indicator":"sma","period":14,"columns":["close"]}
```

| key | holds | an example |
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

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding a moment in time | `"when"` |
| `asCategories` | `true` or `false` | `true` |
| `parts` | a list of one or more of `minute`, `hour`, `dayofweek`, `dayofmonth`, `month`, `quarter`, `season` or `year` | `["month"]` |

- **`column`**: The column holding the moment in time.
- **`asCategories`**: Whether the pieces stand for a group, which they do unless the order is the point.
- **`parts`**: Which pieces of the moment become columns of their own.

Means what it says from version 1 of the file.

## `fill.missing`

Fills the gaps in a column the named way, below the split, and marks where they were. A value no row decided is settle.gaps, above it.

```json
{"step":"fill.missing","column":"column","with":"median"}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding a number or a whole number | `"column"` |
| `with` | one of `"mean"`, `"median"`, `"zero"`, `"previous"`, `"refuse"` or `{"kind": "constant", "value": a number}` | `"median"` |
| `refuseAbove` | a share, from nought to one; left out, there is none | left out |

- **`column`**: The column with gaps in it.
- **`with`**: What goes in the gaps: the mean or the median of the training rows, the value before the gap, nought, a constant, or refuse.
- **`refuseAbove`**: The share of the training rows that may be gaps and still be filled. Above it the column is not filled, and the column that says where the gaps were speaks for it. Left out, every share is filled.

Means what it says from version 1 of the file.

## `fill.nan`

Deals with a value that is not a number a model can use; refusing it is the default.

```json
{"step":"fill.nan","column":"column","with":"refuse"}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding a number | `"column"` |
| `with` | one of `"refuse"`, `"mean"`, `"median"`, `"zero"` or `{"kind": "constant", "value": a number}` | `"refuse"` |

- **`column`**: The column to watch for values that are not numbers a model can use.
- **`with`**: What happens to a value that is not a number: refuse, which is the default and usually the right answer, or mean, median, zero or constant.

Means what it says from version 1 of the file.

## `history`

Writes beside each row the mean of a column over the earlier training rows of its key, pulled towards the training rows' mean.

```json
{"step":"history","key":"key","value":"value","orderBy":["when"],"shrink":3,"window":3,"into":"value_history_key"}
```

| key | holds | an example |
|---|---|---|
| `key` | the name of a column holding anything | `"key"` |
| `value` | the name of a column holding a number, a whole number or true or false | `"value"` |
| `orderBy` | a list of the names of columns holding a moment in time, a whole number or a number, each named once | `["when"]` |
| `shrink` | a number, 0 or more | `3` |
| `window` | a whole number, at least 1; left out, 0 | `3` |
| `into` | the name of the column it makes; left out, the step decides | `"value_history_key"` |

- **`key`**: The column whose value says which rows are of one kind: a farm, a type.
- **`value`**: The column of numbers whose mean over the earlier rows of the same kind is read.
- **`orderBy`**: The columns that say which rows came before a row: the first decides, each next one decides between rows the ones before call equal.
- **`shrink`**: How many rows' worth of weight the mean of every training row has beside a key's own rows, above nothing: the larger, the more a key with few rows is pulled towards it.
- **`window`**: How many of the last places of the order a row reads, rows of one place counting as one; left out, every earlier row. A window of nothing is read with a weight of nothing too.
- **`into`**: The name of the column it makes; left out, the value's name, 'history' and the key's name.

Means what it says from version 10 of the file.

## `learn.ml`

Names the trainer from ML.NET this pipeline is declared for: which trainer, the settings it takes and the seed it repeats from.

```json
{"step":"learn.ml","trainer":{"kind":"fastTree","leaves":20,"trees":100,"leastRows":10,"rate":0.2},"seed":20260929}
```

| key | holds | an example |
|---|---|---|
| `trainer` | a part, written as a name and its settings: `fastTree`, which takes `leaves`, `trees`, `leastRows` or `rate` or `fastForest`, which takes `leaves`, `trees` or `leastRows` | `{"kind":"fastTree","leaves":20,"trees":100,"leastRows":10,"rate":0.2}` |
| `seed` | a whole number | `20260929` |

- **`trainer`**: The trainer these rows are prepared for, and the settings it takes.
- **`seed`**: The number the trainer's own random draws are worked out from, so the same declaration gives the same model.

Means what it says from version 7 of the file.

## `learn.network`

Names the network this pipeline is declared for: its layers, what moves them, what judges them, when the run stops and the engine it runs on.

```json
{"step":"learn.network","layers":[{"kind":"dense","units":16},{"kind":"relu"}],"optimizer":{"kind":"adam","rate":0.001,"firstMoment":0.9,"secondMoment":0.999,"epsilon":1E-08},"loss":{"kind":"meanSquaredError"},"stopping":{"kind":"never"},"engine":"light","seed":20260929,"epochs":100,"batch":32}
```

| key | holds | an example |
|---|---|---|
| `layers` | a list of parts, each a name and its settings: `dense`, which takes `units` and may leave out `bias`, `weights`, `biasStart`, `l1`, `l2`, `maxNorm`, `learns`, `rateShare`, `rank`, `alpha`, `name` and `takes`, `relu`, which may leave out `name` and `takes`, `tanh`, which may leave out `name` and `takes`, `sigmoid`, which may leave out `name` and `takes`, `dropout`, which takes `rate` and may leave out `name` and `takes`, `batchNorm`, which takes `momentum` or `epsilon` and may leave out `scale`, `shift`, `learns`, `rateShare`, `name` and `takes`, `layerNorm`, which takes `epsilon` and may leave out `scale`, `shift`, `learns`, `rateShare`, `name` and `takes`, `gelu`, which takes `approximate` and may leave out `name` and `takes`, `silu`, which may leave out `name` and `takes`, `mish`, which may leave out `name` and `takes`, `leakyRelu`, which takes `negativeSlope` and may leave out `name` and `takes`, `prelu`, which takes `eachFeature` and may leave out `name` and `takes`, `elu`, which takes `alpha` and may leave out `name` and `takes`, `selu`, which may leave out `name` and `takes`, `softmax`, which may leave out `name` and `takes`, `rmsNorm`, which takes `epsilon` and may leave out `scale`, `learns`, `rateShare`, `name` and `takes`, `periodic`, which takes `frequencies`, `sigma`, `width` or `lite` and may leave out `name` and `takes`, `randomFlip`, which takes `horizontal`, `vertical` or `seed` and may leave out `name` and `takes`, `randomCrop`, which takes `pad` or `seed` and may leave out `name` and `takes`, `gaussianNoise`, which takes `spread` or `seed` and may leave out `name` and `takes`, `flatten`, which may leave out `name` and `takes`, `tokenEmbedding`, which takes `vocabulary` or `width` and may leave out `keepSteps`, `name` and `takes`, `unpadded`, which may leave out `name` and `takes`, `lstm`, which takes `units` and may leave out `layers`, `bothWays`, `sequences`, `name` and `takes`, `gru`, which takes `units` and may leave out `layers`, `bothWays`, `sequences`, `name` and `takes`, `multiHeadAttention`, which takes `heads` and may leave out `causal`, `dropout`, `name` and `takes`, `attention`, which takes `heads` and may leave out `causal`, `dropout`, `name` and `takes`, `sinePositions`, which may leave out `name` and `takes`, `add`, which may leave out `name` and `takes`, `concatenate`, which may leave out `name` and `takes`, `columns`, which takes `first` or `count` and may leave out `name` and `takes` or `embedding`, which takes `column` or `width` | `[{"kind":"dense","units":16},{"kind":"relu"}]` |
| `optimizer` | a part, written as a name and its settings: `sgd`, which takes `rate` or `momentum` and may leave out `weightDecay`, `dampening` and `nesterov`, `adam`, which takes `rate`, `firstMoment`, `secondMoment` or `epsilon` and may leave out `weightDecay` and `amsgrad`, `adamw`, which takes `rate`, `firstMoment`, `secondMoment`, `epsilon` or `weightDecay` and may leave out `amsgrad`, `rmsprop`, which takes `rate`, `alpha`, `epsilon` or `momentum` and may leave out `weightDecay` and `centered`, `nadam`, which takes `rate`, `firstMoment`, `secondMoment`, `epsilon` or `momentumDecay` and may leave out `weightDecay`, `radam`, which takes `rate`, `firstMoment`, `secondMoment` or `epsilon` and may leave out `weightDecay` and `decoupledWeightDecay`, `lion`, which takes `rate`, `firstMoment` or `secondMoment` and may leave out `weightDecay` or `adan`, which takes `rate`, `firstMoment`, `secondMoment`, `thirdMoment` or `epsilon` and may leave out `weightDecay` | `{"kind":"adam","rate":0.001,"firstMoment":0.9,"secondMoment":0.999,"epsilon":1E-08}` |
| `schedule` | a part, written as a name and its settings: `constant`, which takes nothing, `stepDecay`, which takes `every` or `factor`, `exponentialDecay`, which takes `factor`, `cosineDecay`, which takes `epochs` or `minimum`, `linearWarmup`, which takes `epochs` or `start` or `trainingPlateau`, which takes `patience`, `factor`, `threshold`, `cooldown` or `minimum`; left out, `constant` | left out |
| `clip` | a number, 0 or more; left out, 0 | left out |
| `clipValue` | a number, 0 or more; left out, 0 | left out |
| `stepEvery` | a whole number, at least 1; left out, 1 | left out |
| `average` | a number, 0 or more; left out, 0 | left out |
| `judgedByAverage` | `true` or `false`; left out, `false` | left out |
| `lossScale` | a number, 0 or more; left out, 0 | left out |
| `recompute` | `true` or `false`; left out, `false` | left out |
| `loss` | a part, written as a name and its settings: `meanSquaredError`, which takes nothing, `crossEntropy`, which may leave out `smoothing`, `binaryCrossEntropy`, which may leave out `smoothing`, `earthMoversDistance`, which takes nothing, `meanAbsoluteError`, which takes nothing, `huber`, which takes `delta`, `quantile`, which takes `quantile`, `focal`, which takes `gamma` or `alpha`, `hinge`, which takes nothing or `poisson`, which takes `logInput` or `epsilon` | `{"kind":"meanSquaredError"}` |
| `stopping` | a part, written as a name and its settings: `never`, which takes nothing or `patience`, which takes `patience`, `least` or `best` | `{"kind":"never"}` |
| `engine` | words | `"light"` |
| `seed` | a whole number | `20260929` |
| `epochs` | a whole number, at least 1 | `100` |
| `batch` | a whole number, at least 1 | `32` |

- **`layers`**: The layers, from the one the prepared rows reach first.
- **`optimizer`**: What moves the network's numbers at every step.
- **`schedule`**: How the rate changes from epoch to epoch; left out, the optimizer's own rate every epoch.
- **`clip`**: The most norm a step's gradients may have together before they move the network; left out, or nought, they are not clipped.
- **`clipValue`**: The most any value of a step's gradients may reach, either side of nothing, before they move the network; left out, or nought, they are not clipped by value.
- **`stepEvery`**: How many batches' gradients are averaged into one step; left out, a step every batch.
- **`average`**: The share an average of the network's weights keeps of itself at every step, above nothing and below one, written over the weights when the run ends; left out, or nought, none is kept.
- **`judgedByAverage`**: Whether every epoch is judged by the average of the weights rather than by the weights as they are, so early stopping watches, and restores, the weights the run ends holding; left out, by the weights as they are.
- **`lossScale`**: The scale the loss is multiplied by before the gradients are worked back from it, a power of two, so that small gradients survive an engine that keeps its work in a narrow width: lowered when a step's gradients are not numbers, raised after a run of clean steps, as PyTorch's gradient scaler does; left out, or nought, the loss is not scaled.
- **`recompute`**: Whether the workings of every layer are thrown away after its forward pass and worked out again when the way back reaches it, so that a larger batch fits in the same memory, at the price of one more forward pass and with no number of the run changed; left out, every layer keeps its workings.
- **`loss`**: What the network's answers are judged by while it trains.
- **`stopping`**: When the run stops: at its last pass, or once the validation loss stops falling.
- **`engine`**: The name of the engine the run's arithmetic happens on; which engine it stands for — and which device it works on — is the application's to say.
- **`seed`**: The number every random draw of the run is worked out from, so the same declaration gives the same run.
- **`epochs`**: How many times the run goes over the training rows, at most.
- **`batch`**: How many rows one step of the run is worked out from.

Means what it says from version 5 of the file.

## `maths`

Pulls a column into another shape by arithmetic that learns nothing: a logarithm, a root, a reciprocal.

```json
{"step":"maths","column":"column","maths":"log1p","into":"column"}
```

| key | holds | an example |
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
{"step":"normalise","column":"column","scale":"midrange","outOfRange":"pass"}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding a number, a whole number or true or false | `"column"` |
| `scale` | one of `standard`, `minmax`, `maxabs`, `robust`, `quantile`, `power` or `midrange` | `"midrange"` |
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

| key | holds | an example |
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

| key | holds | an example |
|---|---|---|
| `columns` | a list of the names of columns holding a moment in time, a whole number or a number, each named once | `["when"]` |

- **`columns`**: The columns the rows are put in order by: the first decides, each next one decides between rows the ones before call equal.

Means what it says from version 2 of the file.

## `outliers.clip`

Holds the extreme values of a column to bounds learned from the training rows.

```json
{"step":"outliers.clip","column":"column","bounds":"iqr","at":1.5,"outlier":"clip"}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding a number, a whole number or true or false | `"column"` |
| `bounds` | one of `quantile`, `sigma` or `iqr` | `"iqr"` |
| `at` | a number, 0 or more | `1.5` |
| `outlier` | one of `clip`, `blank` or `refuse` | `"clip"` |

- **`column`**: The column whose extremes are held.
- **`bounds`**: How the bounds are worked out: by quantile, by spread, or by the middle half.
- **`at`**: How far out the bounds sit: a share for a quantile, nothing for the training extremes; a multiple of the spread or the middle half otherwise.
- **`outlier`**: What happens to a value outside the bounds: held at the edge, made a gap, or refused.

Means what it says from version 1 of the file.

## `read.csv`

Reads the rows from a comma-separated file.

```json
{"step":"read.csv","path":"data.csv"}
```

| key | holds | an example |
|---|---|---|
| `path` | the path of a file; a relative one is read from the pipeline's folder | `"data.csv"` |

- **`path`**: Where the comma-separated file will be, when the pipeline runs.

Means what it says from version 1 of the file.

## `read.excel`

Reads the rows from a sheet of an Excel workbook, the first unless one is named, its first row naming the columns.

```json
{"step":"read.excel","path":"data.xlsx"}
```

| key | holds | an example |
|---|---|---|
| `path` | the path of a file; a relative one is read from the pipeline's folder | `"data.xlsx"` |
| `sheet` | words; may be left out | left out |

- **`path`**: Where the workbook will be, when the pipeline runs: .xlsx, .xls or .xlsb.
- **`sheet`**: The sheet the rows are on. Left out, the first sheet.

Means what it says from version 1 of the file.

## `read.images`

Reads the rows from a folder of pictures, a folder to a class: a picture a row at the size and channels the step says, the folder it stands in the answer, and a column of nought and one for each class.

```json
{"step":"read.images","path":"pictures","width":28,"height":28,"channels":3}
```

| key | holds | an example |
|---|---|---|
| `path` | the path of a file; a relative one is read from the pipeline's folder | `"pictures"` |
| `width` | a whole number, at least 1 | `28` |
| `height` | a whole number, at least 1 | `28` |
| `channels` | a whole number, at least 1 | `3` |

- **`path`**: The folder the pictures will be in, when the pipeline runs: a folder to each class, pictures of .png, .jpg, .bmp, .gif or .tga in each.
- **`width`**: How many pixels across every picture is; one that is not is refused, never resized.
- **`height`**: How many pixels down every picture is; one that is not is refused, never resized.
- **`channels`**: How many numbers each pixel is: 1 for grey, 3 for red green blue, 4 with how much shows through.

Means what it says from version 12 of the file.

## `read.join`

Reads two comma-separated files as one source: each row of the left file beside the one row of the right file its key names. Rows served later arrive already joined.

```json
{"step":"read.join","left":{"kind":"csv","path":"planned.csv"},"right":{"kind":"csv","path":"arrived.csv"},"on":["key"],"unmatched":"refuse"}
```

| key | holds | an example |
|---|---|---|
| `left` | a part, written as a name and its settings: `csv`, which takes `path` | `{"kind":"csv","path":"planned.csv"}` |
| `right` | a part, written as a name and its settings: `csv`, which takes `path` | `{"kind":"csv","path":"arrived.csv"}` |
| `on` | a list of the names of columns holding anything, each named once | `["key"]` |
| `unmatched` | one of `refuse` or `drop` | `"refuse"` |

- **`left`**: The file whose rows the joined rows are: each of its rows, in its order, beside its partner.
- **`right`**: The file each left row takes its partner from: the one row whose key is the left row's key.
- **`on`**: The columns a left row and its partner share, named alike in both files: each compared as the exact text of its cells, the spaces around it taken off.
- **`unmatched`**: What becomes of a left row whose key the right file does not hold: refuse stops the run, naming the row; drop leaves it out and counts it in the fit.

Means what it says from version 8 of the file.

## `read.json`

Reads the rows from a JSON file holding an array of records, one object a row, every value as the file writes it.

```json
{"step":"read.json","path":"data.json"}
```

| key | holds | an example |
|---|---|---|
| `path` | the path of a file; a relative one is read from the pipeline's folder | `"data.json"` |

- **`path`**: Where the JSON file will be, when the pipeline runs: an array of records, one object a row.

Means what it says from version 1 of the file.

## `read.parquet`

Reads the rows from an Apache Parquet file, which says what each of its columns holds.

```json
{"step":"read.parquet","path":"data.parquet"}
```

| key | holds | an example |
|---|---|---|
| `path` | the path of a file; a relative one is read from the pipeline's folder | `"data.parquet"` |

- **`path`**: Where the Parquet file will be, when the pipeline runs.

Means what it says from version 1 of the file.

## `read.rows`

Takes rows that are handed in rather than opened: a table already in memory, a reader over a query.

```json
{"step":"read.rows","description":"rows handed in"}
```

| key | holds | an example |
|---|---|---|
| `description` | words | `"rows handed in"` |

- **`description`**: What the rows are, for whoever reads the file later.

Means what it says from version 1 of the file.

## `read.sounds`

Reads the rows from a folder of WAV recordings, a folder to a class: a recording a row at the length, channels and rate the step says, the folder it stands in the answer, and a column of nought and one for each class.

```json
{"step":"read.sounds","path":"recordings","rate":16000,"length":16000,"channels":1}
```

| key | holds | an example |
|---|---|---|
| `path` | the path of a file; a relative one is read from the pipeline's folder | `"recordings"` |
| `rate` | a whole number, at least 1 | `16000` |
| `length` | a whole number, at least 1 | `16000` |
| `channels` | a whole number, at least 1 | `1` |

- **`path`**: The folder the recordings will be in, when the pipeline runs: a folder to each class, files of .wav in each.
- **`rate`**: How many samples a second every recording has; one that has another rate is refused, never resampled.
- **`length`**: How many samples each channel of every recording holds; a longer or shorter one is refused, never cut or padded.
- **`channels`**: How many channels every recording has; one that has another number is refused, never mixed.

Means what it says from version 12 of the file.

## `read.texts`

Reads the rows from a folder of text files, a folder to a class: a file a row, the folder it stands in the answer, and a column of nought and one for each class.

```json
{"step":"read.texts","path":"texts"}
```

| key | holds | an example |
|---|---|---|
| `path` | the path of a file; a relative one is read from the pipeline's folder | `"texts"` |

- **`path`**: The folder the text files will be in, when the pipeline runs: a folder to each class, files of .txt in each.

Means what it says from version 12 of the file.

## `scale.given`

Scales a column into a range from bounds you give, above the split, so a feature worked out after it is worked out from scaled columns.

```json
{"step":"scale.given","column":"column","lowest":0,"highest":1,"lands":"signed"}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding a number or a whole number | `"column"` |
| `lowest` | a number | `0` |
| `highest` | a number | `1` |
| `lands` | one of `signed`, `unit` or `splitsign` | `"signed"` |

- **`column`**: The column to scale.
- **`lowest`**: The lowest value the column can hold.
- **`highest`**: The highest value it can hold.
- **`lands`**: Where the scaled values land: between minus one and one, or between nothing and one.

Means what it says from version 7 of the file.

## `settle.gaps`

Settles the gaps in a column with a value no row decided, so a feature worked out from it is not a gap.

```json
{"step":"settle.gaps","column":"column","with":"zero"}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding a number or a whole number | `"column"` |
| `with` | one of `"zero"`, `"refuse"` or `{"kind": "constant", "value": a number}` | `"zero"` |

- **`column`**: The column with gaps in it.
- **`with`**: What goes in the gaps, decided by nobody but you: zero, a constant, or refuse for a column that is not supposed to have gaps at all.

Means what it says from version 6 of the file.

## `shuffle`

Puts the rows in an order drawn from a seed, before they are divided, so every part holds the same mixture.

```json
{"step":"shuffle","seed":20260929}
```

| key | holds | an example |
|---|---|---|
| `seed` | a whole number | `20260929` |
| `byId` | `true` or `false`; left out, `false` | left out |

- **`seed`**: The number that makes the shuffle repeatable: the same seed puts the same rows in the same order.
- **`byId`**: Whether to rank by the declared id rather than the whole record. Left out, by the whole record.

Means what it says from version 7 of the file.

## `split.atRandom`

Divides the rows at random, the same way every time for the same seed.

```json
{"step":"split.atRandom","train":0.7,"validation":0.15,"test":0.15,"predict":0,"seed":20260923,"testSeed":99,"folds":5,"fold":0}
```

| key | holds | an example |
|---|---|---|
| `train` | the share the model learns from: above nought, at most one | `0.7` |
| `validation` | the share used while choosing between models: nought to one | `0.15` |
| `test` | the share kept back until the end: above nought, at most one | `0.15` |
| `predict` | the share held back to predict on: nought to one, and none when it is left out | `0` |
| `calibration` | the share an interval is calibrated on: nought to one, and none when it is left out | left out |
| `seed` | a whole number | `20260923` |
| `testSeed` | a whole number, at least 0; left out, -1 | `99` |
| `folds` | a whole number, at least 2; left out, 0 | `5` |
| `fold` | a whole number, at least 0; left out, -1 | `0` |
| `byId` | `true` or `false`; left out, `false` | left out |

- **`train`**: How the rows are shared out: training, validation, test, a part to predict on and a part to calibrate an interval on, which together make the whole.
- **`seed`**: The number that makes the shuffle repeatable: the same seed deals the same rows the same way.
- **`testSeed`**: The number that deals the rows to be measured on, none or more. Given, those rows are the same whatever the seed is, and the seed deals the rest. Left out, the seed deals every part.
- **`folds`**: How many slices the rows that are neither test nor predict are cut into, at least two; one of them is validation and the others are training. Left out, the rows are cut once, as shares say.
- **`fold`**: Which of the slices is validation, counted from nought; said with folds, and below them.
- **`byId`**: Whether to rank by the declared id rather than the whole record. Left out, by the whole record.

Means what it says from version 2 of the file.

## `split.byTime`

Divides the rows by when they happened: the earliest to learn from, the latest to be measured on.

```json
{"step":"split.byTime","column":"when","train":0.7,"validation":0.15,"test":0.15,"predict":0,"origins":3,"origin":0}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding a moment in time, a whole number or a number | `"when"` |
| `train` | the share the model learns from: above nought, at most one | `0.7` |
| `validation` | the share used while choosing between models: nought to one | `0.15` |
| `test` | the share kept back until the end: above nought, at most one | `0.15` |
| `predict` | the share held back to predict on: nought to one, and none when it is left out | `0` |
| `calibration` | the share an interval is calibrated on: nought to one, and none when it is left out | left out |
| `gap` | a whole number, at least 0; left out, 0 | left out |
| `origins` | a whole number, at least 2; left out, 0 | `3` |
| `origin` | a whole number, at least 0; left out, -1 | `0` |

- **`column`**: The column that says when a row happened.
- **`train`**: How the rows are shared out: training, validation, test, a part to predict on and a part to calibrate an interval on, which together make the whole.
- **`gap`**: How many of the last moments of every part are kept apart, fitted on by nothing and handed to nothing: at least as many as the rows an answer reads ahead. Left out, none.
- **`origins`**: How many places the cut between training and validation is moved to, at least two; the rows that are neither test nor predict are cut into one block more. Left out, the rows are cut once, as shares say.
- **`origin`**: Which of the places this is, counted from nought; said with origins, and below them.

Means what it says from version 2 of the file.

## `split.stratified`

Divides the rows at random while keeping the mixture of one column the same in every part.

```json
{"step":"split.stratified","column":"class","train":0.7,"validation":0.15,"test":0.15,"predict":0,"seed":20260923,"folds":5,"fold":0}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding anything | `"class"` |
| `train` | the share the model learns from: above nought, at most one | `0.7` |
| `validation` | the share used while choosing between models: nought to one | `0.15` |
| `test` | the share kept back until the end: above nought, at most one | `0.15` |
| `predict` | the share held back to predict on: nought to one, and none when it is left out | `0` |
| `calibration` | the share an interval is calibrated on: nought to one, and none when it is left out | left out |
| `seed` | a whole number | `20260923` |
| `folds` | a whole number, at least 2; left out, 0 | `5` |
| `fold` | a whole number, at least 0; left out, -1 | `0` |
| `byId` | `true` or `false`; left out, `false` | left out |

- **`column`**: The column whose mixture of values is kept the same in every part.
- **`train`**: How the rows are shared out: training, validation, test, a part to predict on and a part to calibrate an interval on, which together make the whole.
- **`seed`**: The number that makes the shuffle repeatable: the same seed deals the same rows the same way.
- **`folds`**: How many slices the rows that are neither test nor predict are cut into, at least two; one of them is validation and the others are training. Left out, the rows are cut once, as shares say.
- **`fold`**: Which of the slices is validation, counted from nought; said with folds, and below them.
- **`byId`**: Whether to rank by the declared id rather than the whole record. Left out, by the whole record.

Means what it says from version 2 of the file.

## `target`

Names the column a model is asked to predict, which is handed over apart from the numbers it is shown.

```json
{"step":"target","column":"answer"}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding anything | `"answer"` |

- **`column`**: The column a model is asked to predict, handed over apart from the numbers it is shown.

Means what it says from version 1 of the file.

## `target.ahead`

Names an answer read from a column rows later in the declared order: the value then, or the return on the row's own value by then.

```json
{"step":"target.ahead","column":"close","ahead":1,"as":"value"}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding a number, a whole number or true or false | `"close"` |
| `ahead` | a whole number, at least 1 | `1` |
| `as` | one of `value` or `return` | `"value"` |

- **`column`**: The column the answer is read from, rows later.
- **`ahead`**: How many rows later the answer is read, in the declared order: at least one.
- **`as`**: What the answer is: the value itself then, or the return on the row's own value by then.

Means what it says from version 2 of the file.

## `target.distribution`

Names the columns a model is asked to predict as one answer: how a whole is divided among them, in their order.

```json
{"step":"target.distribution","columns":["share1","share2"]}
```

| key | holds | an example |
|---|---|---|
| `columns` | a list of the names of columns holding a number, a whole number or true or false, each named once, 2 at least | `["share1","share2"]` |
| `scaleBy` | the name of a column holding a number, a whole number or true or false; may be left out | left out |
| `ordered` | `true` or `false`; left out, `false` | left out |
| `remainder` | the name of the column it makes; left out, it makes none | left out |

- **`columns`**: The columns the answer is divided among, in their order: at least two.
- **`scaleBy`**: The column saying how many the shares are shares of, so predictions come back as how many fell in each; left out, they come back as shares.
- **`ordered`**: Whether the columns are in an order that means something, as bands of weight are, so a prediction is measured by how far its shares lie from the answer's along it; left out, they are not.
- **`remainder`**: The column made for what is left of the whole once the columns are counted, when fewer arrive than the column saying how many there were; left out, the columns hold the whole.

Means what it says from version 2 of the file.

## `target.labels`

Names the columns a model is asked to predict as one answer of labels, each nought or one on every row.

```json
{"step":"target.labels","columns":["label1","label2"]}
```

| key | holds | an example |
|---|---|---|
| `columns` | a list of the names of columns holding a number, a whole number or true or false, each named once, 2 at least | `["label1","label2"]` |
| `ones` | a whole number, at least 0; left out, 0 | left out |

- **`columns`**: The columns holding the labels, each nought or one on every row: at least two.
- **`ones`**: How many of the columns hold a one on every row: one when a row is exactly one of its things; left out, any number.

Means what it says from version 2 of the file.

## `target.numbers`

Names the columns a model is asked to predict as one answer of free numbers, each any finite number.

```json
{"step":"target.numbers","columns":["number1","number2"]}
```

| key | holds | an example |
|---|---|---|
| `columns` | a list of the names of columns holding a number, a whole number or true or false, each named once, 2 at least | `["number1","number2"]` |

- **`columns`**: The columns holding the numbers of the answer, in their order: at least two.

Means what it says from version 8 of the file.

## `tokenize`

Cuts a column of text into ids of the words the training rows held most often, every row the same number of them, for an embedding to take.

```json
{"step":"tokenize","column":"review","length":32,"vocabulary":5000}
```

| key | holds | an example |
|---|---|---|
| `column` | the name of a column holding words | `"review"` |
| `length` | a whole number, at least 1 | `32` |
| `vocabulary` | a whole number, at least 3 | `5000` |

- **`column`**: The column of text to cut into ids.
- **`length`**: How many ids every row is written as: a longer text is cut at it, a shorter one filled up with the padding id, nought.
- **`vocabulary`**: How many ids there are, the padding id (nought) and the id of a word never seen (one) among them: the words the training rows held most often take the rest.

Means what it says from version 12 of the file.

## `watch.reference`

Keeps the training rows of the columns it watches, as counts, so rows served later can be measured against them.

```json
{"step":"watch.reference","warnAbove":0.1,"columns":["column"]}
```

| key | holds | an example |
|---|---|---|
| `warnAbove` | a number, 0 or more | `0.1` |
| `columns` | a list of the names of columns holding anything, each named once; may be left out | `["column"]` |

- **`warnAbove`**: The distance above which two parts are flagged as unlike: a share above nought, at most one.
- **`columns`**: The columns to keep the training rows of; left out, every column where it stands but the id, the answer and the columns the rows are ordered or divided in time by.

Means what it says from version 11 of the file.
