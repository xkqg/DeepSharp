# Changelog

What changed in each release, and what it means for you. The heading of a section is the version it shipped
as. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0]

A pipeline you can look at: every step written as a block of a notebook, and the data at any block shown on
request. Underneath it, the pipeline became strict where it was only polite. Every way of writing one keeps
the same rules, a row is known by what it says rather than where it stands, and a file says everything that is
wrong with it at once, each at its line and column.

### Upgrading from 0.2

- **A file is read with a catalog.** `PipelineDeclaration.FromJson(json)` and `PreparedData.FromJson(json)`
  are gone; write `FromJson(json, StepCatalog.BuiltIn())`, and add `.WithIndicators()` when the file holds
  indicators. Without a catalog a file could only ever hold this package's own verbs, and it was read as if
  nothing else existed.

- **The file names its version, and what was fitted names the steps it was fitted behind.** A pipeline is
  written as `{"version": 2, "declaration": [...], "fitted": [...]}`. Every fitted entry carries a key made
  from its own step and every step above it, so a fit is never used under steps that changed after it was
  learned: a fit spliced under another declaration used to serve a price of 135.7 where 0.9048 was meant. A
  file from 0.2 names no version and is read as the first one. Its declaration still loads, except where a
  verb's meaning has changed since — the three splits and `drop.warmup` — and those are refused by name
  rather than run the new way. Write them again. The fitted half of a 0.2 file is refused: fit again. A verb
  that is new in this release needs `"version": 2` in the file that names it.

- **The columns are declared directly after the source.** `Declare` comes straight after `ReadCsv` or `Read`,
  and every other step stands below it. A step written above the schema used to fail a whole run later, as a
  column nobody could find.

- **`DeclareStep.Columns` is every column the schema names; `DeclareStep.Taking` is the ones that take part.** A
  schema can now name a column and exclude it, so the two can differ. Code that reads `Columns` to learn which
  columns a pipeline carries should read `Taking`. A schema written with 0.2 excludes nothing, and there the two
  are the same.

- **The rules hold wherever a step comes from.** A pipeline has one source, one schema, one split, one order
  and one output. Every step does something the run acts on. Rows are dropped and put in order before the
  split, never after it. A column is read only where it exists, and only by a step that can work on its kind.
  The chain, the extension point, a hand-written file and a notebook all meet these rules in the same place,
  and a refusal names every fault at once: `DeclarationException.Faults`, each with its step.

- **A split divides the rows by what they say.** Each row is ranked by a digest of its own contents and the
  seed rather than by its place in the file. So the same rows land in the same parts whatever order they
  arrive in, and every copy of a repeated row lands where its first copy does. The parts hold as many rows as
  before, but not the same ones, so what is learned from the training rows moves a little: on the Titanic
  data the median age a fill learns goes from 28 to 29. A split in time keeps every row of one moment on the
  same side of the line.

- **The order of the rows is declared before anything reads it.** An indicator, `DropWarmUp` and a fill that
  carries the previous value forward all read the rows in their order, and nothing said that order was time.
  `OrderBy("Date")` above them says it; without it they are refused. The same prices listed newest first
  gave a five-day average of 98.352 where the right one is 99.74.

- **A list of columns names each column once.** `NormaliseRow` refuses a column written twice, in C# and in
  a file, a 0.2 file included; 0.2 accepted it and counted the column twice in the row's size. The same holds
  for every new verb that takes a list. An indicator is the exception: its roles may name one price in more
  than one place.

- **A step reads what it is given and nothing else.** A key a step does not take is refused, naming the keys
  it does take; it used to be read past and then vanish when the step was written back. A whole number has
  to be whole: a seed of 3.5 was truncated, and is now refused.

- **What learns refuses what is not a number.** A fit that meets a not-a-number or an infinity among the
  training values refuses it and names `fill.nan`, instead of learning a centre from it. `fill.nan` now
  catches infinities as well, and `Batch` refuses both. A cell that overflows, such as `1e999`, is refused
  where it is read.

- **`EncodeCategories()` is one step**, `encode.categories`, with one list of categories per column, rather
  than one `encode` step per column.

- **What a fill counts is counted on the training rows.** The number of gaps a fill records is taken from
  the rows it learns from, like everything else it learns: 133 in the Titanic ages, not the 177 in the file.

- **For somebody writing a step of their own.** `IAssignsParts` and `ILearnsFromData` are merged into
  `ISplitStep` and `IFittedStep`: saying a step splits or learns is now the same as doing it. A step
  implements exactly one of the capabilities the run acts on — opening rows, declaring columns, ordering,
  adding, dropping rows or columns, splitting, learning, producing evidence. `IPipelineStep<TSelf>` asks for
  a `Purpose` and its `Parameters`, and the step is written, checked and described from those, so a key is
  typed in one place. The `StepCatalog.Register` that took a name and a function is gone: a verb arrives with
  its description or not at all. `IOpensRows.Open` is handed the `SourceFolder` a relative path is read from.
  What a model is asked to predict is a step that implements `INamesTheAnswer`, and one that makes its answer
  from the rows implements `IMakesTheAnswer`. A step that drops rows is shown a served row without the answers
  it awaits: those are gaps in every served row, and no reason to drop one.

### Added

- **`DeepSharp.Verso.Notebooks` — a pipeline written as a notebook.** An extension for
  [Verso](https://www.versonotebooks.com/) in which every block is one step, written as the step's own
  JSON, and the blocks in the order they stand are the pipeline. A block is edited as text, with the verbs,
  keys and columns offered as you type, or field by field in Verso's properties panel; both write the same
  text. In the schema's form a column set "not taken" stays named, excluded with its kind. Each block shows
  what its step is and does, or every fault at its line. "Show the data here" runs the
  pipeline down to that block and shows the rows there: fifty at a time, each column coloured over the
  training rows, with every row knowing the part the split below will put it in. On the grid every column has
  a box for whether it is in and, when the schema takes it, one for whether it is a category: unticking and
  ticking again write the steps that do it and take them back, a column's kind kept. "Choose the columns", on the
  schema's block, lists every column of the source with its first values and a box that takes it in or leaves it
  out, marking new the columns the saved file never showed; each row's kind select commits the kind a keyboard walk
  ends on, as one pick from the state the list was drawn in, so a category walked away from and back to remembers
  the kind it was. Above the rows a select picks the kind of output, and each row's output box puts its column into
  that output or takes it out, with a box of its own that takes the output away; the output's own values — rows
  ahead, a return, how many ones, what the shares are shares of — are selects offering only the values the rules keep.
  What the blocks decide about
  their columns is saved beside the notebook, as `<notebook>.columns.json`, after every change they accept and
  every run of the whole pipeline. An output block is a stage of
  its own, and its form swaps it for any other kind of output. A profile block names, for
  every problem it finds, the step that answers it, and a correlation block is drawn as a heatmap over the
  complete training rows. The toolbar runs the whole pipeline, fitting every step on the training rows, and
  exports it as the same pipeline file the chain writes. Its "Take over the saved columns" lists, at the schema's
  block, every column whose decision taking the saved ones over would change — whether the source may lack it too, and
  what becomes of the columns the schema does not name — with every saved drop the blocks cannot make and why, and
  changes nothing until the list's own box is ticked. C# cells in the same notebook are handed the
  pipeline as text, under `deepsharp.pipeline`, with the notebook's folder under `deepsharp.folder`. It is
  installed from Verso's Extensions panel and runs in Verso's VS Code extension, in the browser editor
  `verso serve` opens, and inside an application that takes Verso's engine as a dependency.

- **What a model is asked to predict comes in kinds.** `Target(column)` names one column, as before.
  `Distribution(columns, scaleBy)`, `target.distribution`, names the columns a whole is divided among — a flock
  weighed in seventy bands of fifty grams — whose shares are at least nought and sum to one on every row; named
  with the column saying how many there were, which is none of the shares, the shares come back as how many fell in
  each band. `Labels(columns, ones)`, `target.labels`, names columns that are each nought or one, with how many ones
  a row holds when that is said, never more than there are labels. `Ahead(column, ahead, as)`, `target.ahead`,
  makes its answer from a column rows later in the declared order: the price five days on, or the return on today's
  price by then. Reading ahead is held to rules of its own: only an output may, below a split in time whose gap is
  at least as wide, the rows ordered by that split's column alone; and a return stands above every step that
  changes the column it is made from.

- **Every answer is handed over.** `Batch.AnswerNames` and `Batch.Answers` hold as many numbers a row as the
  output names, in its order; `Labels` stays the one number a row for an output of one answer. Each answer is
  refused for a gap or a value that is not finite, as a feature is, and each kind of output refuses a row its
  answers could not be: shares that do not sum to one, a label that is neither nought nor one.

- **A split in time can keep a gap.** `SplitByTime(column, train, validation, gap)` and the `gap` of
  `split.byTime` set apart the last moments of every part, before each line and at the end, fitted on by nothing
  and handed to nothing: `Part.Gap`. Without it, the last training rows of an answer read five days ahead learn
  their answers from the rows a model is measured on. The fit writes down how many rows the gap held, and a file
  without a gap is written exactly as before.

- **The way back in your own units, for a part and for served rows.** `BackToOriginal(predictions, part)` puts
  the predictions for a part back, and `BackToOriginal(predictions, served, rows)` the predictions for served
  rows, each row found again by where it was handed in and checked by its key, `ServedBatch.Keys`, so rows
  handed in again in another order are refused rather than answered with another row's numbers. A way back
  that needs the row a number belongs to reads it as it was read: a share of a flock comes back as birds by the
  flock's own size, a return as a price by the day's own price. For somebody writing a step of their own,
  `IUndoesItself` can say which columns it undoes, what a column was made from, and read that row.

- **A column can be left out and brought back as it was.** A declared column can say it is `excluded`: the schema
  still names it, with its kind, but nothing reads it and the source is not asked for it, and a step that reads it
  is refused at that step, in the words "which the schema excludes". Taking it in again is clearing the word, and
  the column comes back with the kind it had. A category can say which kind it `was` before it became one, so it
  can go back to it. Both are written into the file only where they say something, so a schema that uses neither
  is written exactly as before and keeps its key. A schema that excludes every column it names and keeps none of
  the rest is refused, since no column would take part.

- **What can be done to a column is said once.** `Including`, `Excluding` and `WithKind` on a declaration hand
  back the steps a change to one column makes: a column taken back in as it was, or out of the drop that left it
  out; a column no step reads excluded in the schema with its kind, and any other dropped below the last step that
  reads it and the step that makes it; a category that remembers the kind it came from. Asked for what already is,
  each hands back the steps it was given. `ChoicesFor(columns)` says how each column stands — taking part, excluded,
  dropped, made by a step and which one, kept with the rest, or not declared, and whether the source may lack it —
  and what can be done to it without
  breaking a rule, and `KindsFor(column, header)` the kinds it can be given, so the answer is never
  offered to be left out and a column a later step scales as a number is never offered to become a category. The
  schema's own `WithColumn`, `WithColumnExcluded` and `WithColumnKind` change one block alone. `WithOutput` places
  an output, in the place of the one standing or at the end, and a return directly after the split; `WithoutOutput`
  takes it away; each row says what its column is to the output — an answer comes back to it, or the way back reads
  it — and `ChoicesFor` names the output it asked about.

- **An output is made under its verb in one place.** `StepCatalog.Make(verb, stated, carrying)` makes a step from
  the verb's template, the values the step it replaces holds under the verb's keys, and the values said, which win;
  it is read as a file's step is, so the verb's own rules hold. The notebook's form swaps one kind of output for
  another through it.

- **What a pipeline decided about its columns can be saved on its own.** `PipelinePreset` holds the schema, the
  columns dropped after the steps that read them, the output, and the source's columns as they were last shown
  — nothing fitted and no rows, so a pipeline written again next month over new rows can take it over.
  `PipelinePreset.Of(declaration, header)` takes it from a pipeline, `ToJson()` writes it, and
  `PipelinePreset.FromJson(text, catalog)` reads it back through the same door as a pipeline file: every fault at
  once at its line and column, and a file from a newer version refused whole. A part it does not hold is not
  written. `preset.TakeOver(pipeline, header)` makes the steps the preset's decisions make of a pipeline and lists,
  before anything is applied, every column whose decision changes — its standing, its kind, the kind a category was,
  whether the source may lack it — the output before and after, the schema's order and what it does with the columns
  it does not name when those change, the source's columns the preset never showed, and every saved drop it cannot
  make because its column no longer reaches the end, with how that column stands; steps that would break a rule are
  refused with every fault. `preset.NewColumns(header)` names the source's columns the decisions never showed.
  `CsvRowSource.HeaderOf(path)` reads a file's header alone. In a chain,
  `.Declare(preset, out declared)` takes the saved schema at the source and `.Output(preset, out taken)` the drops and
  the output where the chain names its answer, each saying what it decides before anything runs.

- **Every package runs on .NET 8 as well as .NET 10.** Verso's browser editor runs on .NET 8 for as long as
  .NET 8 is installed, and a package built for .NET 10 alone does not load there. Each package now carries a
  build for both, a host takes the one for the runtime it is on, and every test runs on both.

- **`OrderBy`, `order.by`** — the rows in the order of one or more columns, smallest first, so a time column
  puts the oldest row first. It refuses a gap in a key, and two rows whose keys are equal: nothing says which
  of them came first.

- **`Drop`, `drop.columns`** — a column taken away, above the split or below it.

- **`DropGaps`, `drop.gaps`** — the rows with a gap in the named columns, dropped before the split. It
  replaces the `With.DropRow` that was once planned as a fill strategy: dropping rows after the split would
  quietly change the shares the split promised.

- **`FillMissing(column, strategy, refuseAbove: 0.5)`** — a point beyond which filling is invention. Above
  that share of gaps in the training rows the column is not filled, and the column that marks where the gaps
  were speaks for it.

- **`Profile` and `Correlation`, `evidence.profile` and `evidence.correlation`** — proof declared with the
  pipeline and produced by every run, measured on the rows the split trains on. A profile gives, per column,
  the rows, gaps and values that are not numbers, the distinct values and, for numbers, the smallest, largest,
  mean and median, and names the step that answers each thing it finds. A correlation says which training
  rows it is drawn from and how many were left out. The results are in `PreparedData.Evidence`, and never in
  the pipeline file: evidence is output, not the pipeline.

- **`Pipeline.ViewAt(steps)`** — the data after any number of steps, with where every row stands: the part
  the split puts it in, dropped before the split reaches it, or undivided when nothing divides it. A range or
  a profile drawn above the split is drawn over the rows the split below will train on, not over all of them.

- **Faults at their line and column.** A file that cannot be read throws `PipelineFileException`, whose
  `Faults` hold every problem at once, each with its line and column. A verb nobody has heard of is told the
  nearest one there is; a verb another DeepSharp package brings is told which package to reference. A value
  a step refuses is said in the file's words, without the name of a C# parameter the file never had.

- **A description every door is derived from.** The JSON Schema of the pipeline file
  ([`pipeline.schema.json`](https://github.com/xkqg/DeepSharp/blob/main/pipeline.schema.json)), the
  reference of every verb ([`VERBS.md`](https://github.com/xkqg/DeepSharp/blob/main/VERBS.md)), the
  template a new step starts from and the notebook's form are all generated from the steps' own parameters,
  and tests fail the moment one of them falls behind.

- **`PreparedData.Served(rows)`** — rows that arrived after training, replayed and handed over without an
  answer, each saying which of the handed-in rows it is, since a replay can drop rows and put them in order.

- **`SourceFolder`** — one rule for where a relative path is read from: the folder of the file or notebook
  the pipeline came from, or the working directory for one written in code.

- **`Table.Duplicates(parts)`** — the rows that are there more than once, and how many of them landed in
  different parts.

### Fixed

- **A run and a replay walk the same steps in the same order.** The run lifted every feature above every
  dropped row and the replay dropped nothing, so the same sixty rows came out as fifty-six from one and sixty
  from the other. There is one walk now, in the order the steps were written, and a replay drops and orders
  rows exactly as the run did.

- **The way back is checked on the right rows.** With `DropWarmUp` in the pipeline, the check that a
  prediction leads back to the value that was read compared each row with the one four places further on,
  and refused a pipeline that was fine: 127.83 came back as 133.

- **A row can be served without the answer.** Serving refused a row that lacked the column being predicted,
  so a host had to invent an answer to ask the question.

- **An answer made from another column comes back in the units that column was read in.** A logarithm taken
  into a new column from a scaled fare came back in the scaled units — 0.014 where the fare was 7.25 — and the
  run's check could not tell, because it compared with the answer as read and an answer a step made was never
  read. The way back now goes on through whatever was done to the column the answer was made from, and the
  check compares with that column as it was read.

- **An answer that is a moment in time no longer stops the run.** The check of the way back read it as numbers
  and refused the whole pipeline.

- **Two things 0.2.0 said now hold.** A file naming a verb that is not registered is told whether the verb is
  unknown or belongs to a package that is not installed; 0.2.0 gave one message for both. And a declaration
  holds one split: the builder refused a second one, but the extension point and a file could still carry
  it.

- **`DropWarmUp` refuses a warm-up that covers every row**, instead of leaving none.

## [0.2.1]

### Changed

- **The reader reaches a data frame through MatPlotLibNet.** `DeepSharp.Pipelines.DataFrame` no longer
  asks for `Microsoft.Data.Analysis` itself. It depends on `MatPlotLibNet.DataFrame`, which is built on
  that same frame and carries it along, and which is where the charts and the indicators over a frame
  already live — the door `DeepSharp.Pipelines.Indicators` was already using. Nothing you write changes:
  the frame is the same type it always was, and a package that wants it directly may still say so.

  One thing stays deliberately apart. That package's column reader turns an absent value into a
  not-a-number, which is right for drawing, where a not-a-number means "do not draw this point". In a
  pipeline it would mean "the arithmetic went wrong", and a gap would disappear into a number nobody
  measured. So the reader here keeps its own conversion, with the test that says why.

## [0.2.0]

The data half of the library: everything a set of rows goes through before a model ever sees it, declared
once as a file you can save and replay. The tensors of the first release are untouched.

### Added

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

- **A row belongs to a `Part`, not to a "split".** You *split* the rows; what each row lands in is a part
  of the data — `Part.Train`, `Part.Validation`, `Part.Test`, `Part.Predict` — which is what `Batch`,
  `CountIn` and `PreparedData.Parts` speak in. The verbs keep the word for the act: `SplitByTime`,
  `SplitAtRandom`, `SplitStratified`.

- **The share you are measured on is never written down.** You say what training takes and what validation
  takes; test is whatever is left. Three numbers that must add to one is a rule that breaks quietly — 0.70,
  0.15 and 0.10 leaves a twentieth of the rows in no split at all and every number afterwards is computed
  over less data than you think — so the third number is not a parameter at all. `80, 10` and `0.80, 0.10`
  say the same thing, and mixing percentages and fractions in one call is refused rather than read.
  `SplitAtRandom(0.80)` is the two-way division, with nothing held for validation.

- **`Predict(10)` holds a part back that takes no part in anything.** Nothing is fitted on it and nothing
  is measured on it, so running a trained network over those rows is the closest thing to running it
  tomorrow; with a split in time they are the newest rows in the file. Training, validation and test are
  then the ninety that remain, and `Batch(Part.Predict)` hands them over like any other part. A pipeline
  that holds rows back and then never splits is refused where it is built.

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
  from the numbers, never among them. `Batch(part)` gives rows of numbers with their names in a fixed
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

- **With indicators on the data, the row count is the rows minus the longest warm-up.** `DropWarmUp()` cuts
  the rows at the start that no column can speak for yet, before the split, because a row nobody can use
  should never land in one. An indicator of period N does not say "nothing happened" about the first N
  rows, it says "not enough history yet" — and filling that with a number learned from training would
  invent a measurement nobody took. 506 rows with a twenty-period average on them are 487 rows of data.

- **A prediction comes back in the units it was read in.** Scale what a model is asked to predict and its
  predictions come back scaled, so `BackToOriginal` walks the steps that touched the target backwards and
  undoes each — every scaling, and a logarithm, a root or a reciprocal. The run checks it before handing
  anything over: transform, undo, compare against what was read, and stop if it does not lead back. A step
  that threw information away says so rather than returning a number in units nobody can name, and undoing
  a logarithm gives the middle value rather than the average one, which is said where it matters.

- **The variance-stabilising shapes, and the tail.** `Reshape(column, Maths.Log)` — and Log1P, Reciprocal,
  Sqrt, Square, ArcSin, Abs, Sign — above the split, because the logarithm of a number does not depend on
  any other number. `ClipOutliers` holds the extremes to bounds learned from the training rows, by
  quantile, by spread, or by the middle half, and what happens outside them is your choice: held at the
  edge, made a gap, or refused.

- **A moment in time comes apart into the pieces people reason with.** `TimeParts("Date", TimePart.Season,
  TimePart.Quarter)` — and the minute, the hour, the day of the week, the day of the month, the month and
  the year. They arrive as **categories**, because a month is not a quantity: March is not three of
  anything and December is not twelve times January. `TimePartsAsNumbers` is there for the piece where the
  order really is the point, and where time wraps round a circle says it better than either.

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
