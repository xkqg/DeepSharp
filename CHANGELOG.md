# Changelog

What changed in each release, and what it means for you. The heading of a section is the version it shipped
as. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0]

The first release. It is the foundation the rest of the library is built on: the numbers, their size, and
where the arithmetic happens.

### Added

- **`Shape` — how big a tensor is.** `new Shape(2, 3)` is two rows of three, and it prints as `2x3`. Two
  shapes with the same sizes count as the same shape, so you can compare them and use them as keys. A size
  that cannot exist is refused the moment you write it: a negative length, or sizes so large that the total
  could not be counted. A shape with no sizes at all is a single number — which is what a loss is.

- **`Tensor` — the numbers themselves.** A shape, and the values filling it. A tensor never changes once it
  exists, so handing the same one to two parts of your network is safe. `Tensor.From` takes a copy of what
  you give it, so reusing your own array afterwards cannot alter a tensor you have already passed on.

- **`CpuBackend` — the arithmetic.** Adds and multiplies tensors, using your processor's vector
  instructions through .NET's own maths library. There is nothing native behind it and nothing to install.
  Adding two tensors of different sizes is refused, with a message naming both.

- **`ITensorBackend` — the plug it all goes through.** Every calculation a network does goes through this,
  which is why a model written today can run its heavy work on something else later without being rewritten.
  You create a backend and hand it to what needs it; nothing goes looking for a shared one of its own accord.

- **`DeepSharp.Pipelines` — a pipeline you can write down.** A second package, for the path your data takes
  before a model ever sees it. `Pdd.Create()` starts a pipeline and every verb after it *records* what is to
  be done rather than doing it, so what you wrote can be saved as a file, handed to somebody, and replayed
  later. `ToJson` writes the declaration out and `FromJson` reads it back as the same declaration — a
  property with a test on it, because a promise that both ways reach equally far decays silently otherwise.

- **The split is a line you cannot step over.** `SplitByTime` hands back a different kind of builder, and the
  steps that learn from the data — filling a gap, and everything that follows it — exist only on that one.
  The rule holds wherever a step arrives from: through the chain, through the extension point another
  package uses, or out of a file somebody edited by hand, because it is checked on the declaration itself.
  A step that learns says so in its type, and one that divides the rows says so too. A file naming a step
  nothing has registered is refused rather than read with the step left out, and a message says which verb
  and whether it is unknown or merely not installed.

- **A pipeline you split is finished.** The builder you started with cannot be written to afterwards, so a
  feature cannot arrive on the far side of the line, and two arrangements of the same data are two
  pipelines rather than one declaration containing both.

- **A gap is filled by a name you can check.** `With.Mean`, `With.Median`, `With.Zero`, `With.Previous` and
  `With.Constant(0)` — a word a file did not define is refused rather than carried to whatever a default
  branch would have done with it, and a strategy that takes a number is written with one.

- **It reads real files.** `Declare` names the columns and their kinds and says what becomes of the rest —
  dropped by default, because a column nobody declared is a column nobody checked. An empty cell is a gap
  and stays one; a cell that cannot be read as what it was declared to be is refused with its row, its
  column and its value. True and false are accepted in the spellings files actually use, and numbers and
  dates are read the same way on every machine.

- **Three ways to divide the rows, and none of them a default.** By time, which refuses a row that has no
  time rather than guessing where it belongs; at random with a seed written into the declaration; and
  stratified, which deals each group out separately so a rare answer survives into validation.

- **Features, before the split, because they learn nothing.** A column worked out from two others, and a
  moment in time written as a place on a circle so that eleven at night and midnight are neighbours — in
  whichever of three forms you ask for: the plain value, the same value between nothing and one, or two
  columns saying how far up and how far down.

- **Everything that learns, after it.** Filling a gap, encoding a category, and six ways of bringing a
  column onto a comparable scale. Each is fitted on the training rows alone and then replayed unchanged
  over validation, test and anything that arrives later; what each one learned is written into the file
  beside the declaration. What happens to a value outside the learned range is your choice, said out loud —
  a price meets a new high the first week it is in production.

- **A handover, and a way back in.** `Target` names the column being predicted and it is handed over apart
  from the numbers, never among them. `Batch(split)` gives rows of numbers with their names in a fixed
  order, and refuses a column that still holds words, a gap nobody filled, or a target that no longer
  exists. `Replay` runs the same declaration over rows nobody had seen, with the numbers the training rows
  produced and nothing fitted again — which is what serving is — and a saved pipeline can be loaded back
  from its own file to do exactly that.

- **A value that is not a number gets its own verb.** `FillNaN` stops the run by default, because a
  not-a-number is somebody's broken division and carrying it into a model as if it were a measurement is
  the one thing a pipeline should not do quietly. Two more scales complete the set: by rank, which keeps
  the whole shape of the training distribution, and a reshaping towards a bell curve for a column that
  leans heavily one way.

- **`DeepSharp.Pipelines.DataFrame` — one reader for the long tail.** A third package, reading a pipeline's
  rows out of a `Microsoft.Data.Analysis` data frame, and so out of anything that can fill one: a file, a
  database query, rows you already had. `ReadDataFrame`, `ReadCsvFrame` and `ReadDbAsync` all arrive through
  the same door the pipeline already had for rows handed in, which is why the pipeline itself still needs no
  reader beyond its own. An absent value stays absent rather than becoming a not-a-number.

- **`DeepSharp.Pipelines.Indicators` — a fourth package, and not a line of indicator arithmetic.** Moving
  averages, RSI, ATR, ADX, MACD, Bollinger bands, the stochastic and VWAP, borrowed from MatPlotLibNet's
  published package. They stand above the split because they learn nothing, and two things about them are
  measured rather than promised: every one is held to an impulse test — change one row and no earlier row
  may move — and the warm-up arrives as an absence rather than as a number nobody took.

- **Which columns are categories is said where the data is declared.** `Category("sex", "embarked")` in the
  schema, and `EncodeCategories()` takes them by name afterwards, so adding one to the schema does not mean
  remembering a second line further down. A category that never became numbers is refused at the handover
  rather than dropped in silence.

- **It fits an application that has a host.** `services.AddDeepSharpPipelines()` registers the pipeline
  factory and the catalog of verbs, so a pipeline is resolved the way everything else in a .NET application
  is. A package that brings verbs of its own registers them as a contribution, and every contribution is
  applied whichever order the registrations happen to be written in. None of it is required: a console
  program that writes `Pdd.Create()` with no container anywhere works exactly the same, and a test holds
  that door open. `Samples/DeepSharp.Sample.Pipelines` shows both.
