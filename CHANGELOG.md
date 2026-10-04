# Changelog

What changed in each release, and what it means for you. The heading of a section is the version it shipped
as. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.6.2]

Where a gap is settled before the features are worked out. A column derived from one with a gap is itself a gap, and
what fills a gap is learned from the training rows, so the fill stands below the split and cannot reach back into a
feature worked out above it. Settling is the other half of the same work: a nought, a number you choose, or a refusal —
none of them a value any row decided — so it stands where the features are, and a feature worked out after it is worked
out from settled columns. It learns nothing, so it writes nothing into the fitted half of the file.

### Upgrading from 0.6.1

- **A file this version writes names version 6, and every DeepSharp before it refuses that file whole** — whether or not
  it uses the new verb, because the version is stamped by whoever writes. A model trained on 0.6.2 cannot be served by
  0.6.1. Every file written by 0.2 through 0.6.1 keeps reading exactly as it did, and a trained model keeps matching its
  pipeline, because the digest a model is bound by leaves the version out.

- **The published schema is served from this repository's main branch**, so between this commit and the release an
  editor on 0.6.1 is told that version 6 and `settle.gaps` are valid while the installed reader still refuses them.

### Added

- **`settle.gaps`, and `.SettleGaps` beside `.DropGaps`.** One column at a time, or one line for many:
  `.SettleGaps(gaps => gaps.Zero("age").Constant(-1, "fare"))`. The kinds are the ones no row decides — `Zero`,
  `Constant`, `Refuse` — and the ways a fill learns from the training rows are not among them: a mean or a median comes
  from the rows, and carrying the value before a gap forward reads them in their order, so both stay below the split
  where `.FillMissing` offers them. The step writes no entry into the fitted half, because it learned nothing; a replay
  does what the run did by doing it again.

- **The column that says where the gaps were is named and written in one place.** Three verbs write it — filling a gap,
  settling one, and encoding a column whose cell was empty — and each said the name for itself until the third made it a
  sentence written three times. The name a reader sees does not change.

- **A gap the profile finds names both verbs and where each stands**, in the one sentence it already said.

### Changed

- **The pipeline file is at version 6.** Only the new verb is new in it: no verb that shipped before means anything else
  than it did, and the committed schema and verb reference are projected from the steps as they always were.

## [0.6.1]

Where one line does one thing to many columns. Every verb whose first word is a column — the scalings, the gaps, the
reshapings, the moments on a circle, the clipped extremes and the pieces of a moment — gains a line that names a kind
and lists the columns it holds for, and a pipeline says once, above the split, where its features land. What reaches
the declaration is one step a column, exactly as writing the verb once a column does, so the file, a notebook's blocks
and every fit are what they were: the line is a door a person writes through, never a new shape in the artefact.

### Upgrading from 0.6

- **A scaling that names no kind now lands its rows between minus one and one.** The chain and the step's own
  constructor said standard where the value a new block starts with said midrange, so the same omission meant two
  different things depending on which door it came through; it is written down once now, as
  `NormaliseStep.DefaultScale`, and every door reads it. No pipeline file changes meaning: a file always writes the
  scale it was given, and one that leaves the key out is refused rather than defaulted. What changes is what new code
  means — `.Normalise("age", "fare")` and `scale => scale.Columns("age")` now land their rows between minus one and
  one, which is what a network takes. A caller who wants the old behaviour names it: `.Normalise("age",
  Scale.Standard)`. The one case no source diff shows is a caller who recompiles nothing and swaps the assembly: a
  default is settled where the caller was compiled, so their omitted scale stays standard until they build again.

- **`.Normalise(string, Scale, OutOfRange)` no longer defaults its scale.** `.Normalise("age")` now reaches the
  overload that takes the columns and lands them where the pipeline says its features land; naming a scale is
  unchanged. Nothing else about the signature moved.

### Added

- **One line does one thing to many columns, and the kind is the method.**
  `.Normalise(scale => scale.MidRange("age", "fare").Robust("volume"))` names a kind per group of columns, and
  `.FillMissing(fill => fill.Median("age").Mean("trades"))`, `.FillNaN`, `.Reshape`, `.Cyclical`, `.ClipOutliers`,
  `.TimeParts` and `.TimePartsAsNumbers` take the same shape. What reaches the declaration is one step a column, as
  writing the verb once a column does, so the file, a notebook's blocks and every fit are unchanged — the line is a
  door. What happens outside a learned range is said on the kind that can hold a value in one, and the kinds that land
  their rows in no range have no method that offers the choice. A line that names no column is refused where it
  stands. `Encode`, `Target` and `Ahead` keep the one-column door, each for a reason a test holds them to.

- **Where the features land is said once: `.DefaultFeatures(Form)`.** Written above the split, where it holds for
  every column alike: `Form.Signed` is between minus one and one, `Form.Unit` between nothing and one, and a column
  that names its own kind keeps it. It decides what a scaling that names no kind is written as and how a moment on a
  circle is written, and nothing of it reaches the file — each column is written as the kind that lands it there, so a
  pipeline that says the range and one that names the kinds are the same text.

### Changed

- **A row divided by its own size is worked out around the largest value in it.** The length of a row squares its
  values, and a small enough value squares to nothing while a large one squares to infinity: a row of 1e-160 came back
  with its length out by a relative 5.6e-6, and a row of 1e200 came back as noughts. The sizes are scaled by the
  largest value in the row first, as every library that measures a length does, so what the step declares of the
  columns it writes — between minus one and one — holds for every row there is. A row of ordinary numbers comes back
  exactly as it did.

- **A gap in a column a feature was worked out from is refused with the column it came from.** The features are added
  above the split and the fills stand below it, so a sum of two columns one of which has a gap is itself a gap, and
  filling what it was made from afterwards does not reach back into it. The handover now names the columns the feature
  was worked out from and says to settle those rows above the step that works it out, rather than saying only that a
  gap is a gap.

- **A fit says what it learned, what it decided and what it saw.** `FittedStepValues` gains `Saw` and `Decided` beside
  `Learned`: a value a replay reads is learned, a choice made from the training rows that a replay reads is decided —
  whether a column with that many gaps is filled at all — and a count nothing replays is seen. The file is unchanged
  to the byte, because it holds one object of names and values and marks none of them, so every pipeline published
  before keeps loading; what the words buy is a rule that can be tested rather than hoped for, and the tests now hold
  every fit to it: nothing a replay reads was only counted.

- **An extension method is a member of an `extension(T value)` block.** Every one in the library — 115 of them, across
  every package — is written that way now rather than with a `this` parameter, so the call site reads as a sentence
  and the thing a method is about is its receiver. Nothing about the surface changes: the methods keep their names,
  their arguments and their order, and package validation against 0.5.0 sees no difference.

- **The notebook's file, the engine's questions and the host's duties stand apart.** `NotebookHost` keeps what makes
  it a host — one notebook, one turn at a time, one publisher — and is read a duty at a time across
  `NotebookHost.Editing`, `.Running`, `.Panels`, `.Publishing` and `.Opening`. What its file last held and the writing
  of it are `NotebookFile`; the twenty questions that are about the engine rather than about an open notebook are
  asked of the engine, the notebook's scaffold, a cell or a serializer. One type, one published surface, nothing moved
  that an application can see.

- **One predicate where there were three.** Two of the host's names for "whether two of the engine's names name the
  same thing" and the comparison that asked whether a kernel is the C# one were three copies of one line; there is one
  now, asked of the name.

- **The text a pipeline file is read from is its own thing.** `SurveyedText` holds the text, where each of its parts
  stands and every fault at its line and column, and `TextPieces` reads a larger file a piece of its UTF-8 at a time;
  `PipelineDocument` keeps the pipeline's own vocabulary and holds one of them. The sibling shape in the
  deep-learning core, `NetworkText`, was already this, so the second implementation named what the first had found.
  The refusal that says a key is written twice now shows that key the way every other refusal shows a file's text.

- **`Transforms.cs` is `Scaling.cs` and `Encoding.cs`.** It held two families that share no type, under a name that
  named neither, so a reader after a scaling opens the file named for scaling. No type, name or signature changed.

- **A form's field name is asked of the parameter it belongs to.** The sixteen ways a notebook's properties panel names
  and reads its fields took a parameter's key and what the field was about as two strings side by side, which a caller
  can hand over the wrong way round; the parameter is the receiver now, so it cannot. Every way of writing a field is
  held to having a way of reading it back, found by reflection, so one added without its reader says so at once.

- **The verbs write their own schema and reference, and a model's predictions write themselves.**
  `StepCatalog.JsonSchema()` and `.VerbReference()` now ask the descriptions rather than handing them to a writer, and
  what a model predicted says `ToJson()` and reads `FromJson(…)` as every other saved thing in the library does.

### Removed

- **A guard that could not fire.** Reading what is unsaved compared the notebook with its file inside a `try` that
  answered a fallback when the comparison was caught by a run writing what a cell shows. Measured at 8.2 million
  comparisons against three kinds of writer — the same output written again, the list cleared and refilled, one added
  and taken away — it never once threw: Verso reads those outputs whole rather than as they come. The guard is gone,
  and a test now holds that measurement, so a version of Verso that read them as they came would say so here rather
  than silently telling a person there was nothing to save.

## [0.6.0]

Where the model moves into the pipeline. Pipeline-driven design is the whole course from raw data to a validated
model, declared in advance and replayed — and until now the last step of that course was written by hand after the
declaration ended. A pipeline now names the network it is prepared for, in the words TensorFlow and Keras use or the
words PyTorch uses, and running it trains exactly that network: the layers, what moves them, what judges them, when
the run stops and the engine it runs on are all in the file, so the file says what was trained behind these steps and
training it again from the file gives the same network, number for number.

### Upgrading from 0.5

- **The pipeline file is at its fifth version.** A file DeepSharp writes now names version 5, and an older DeepSharp
  refuses it whole rather than reading a word it does not know. Every file 0.5.0 wrote is read as it always was, and a
  network's file is still held to the pipeline it carries by the digest of its text, which leaves the version out — so
  a model trained behind a 0.4.0 or 0.5.0 pipeline loads and serves unchanged.

- **A catalog that reads a file naming a learner needs the package that brings the verb.** `learn.network` is brought
  by `DeepSharp.Learners.Networks`, as `read.parquet` is brought by `DeepSharp.Pipelines.Parquet`: call
  `StepCatalog.BuiltIn().WithNetworks()`, or let a host find it through `IStepContribution`. A catalog that was never
  taught it says which package brings it, rather than that the verb is misspelled. The notebook's own catalog knows
  it.

- **`IStepParameterVisitor<T>` has a fifteenth member.** A form, a schema or a reference page of your own that
  implements the interface gains `Visit(PartsParameter)`, the kind that holds the parts of a model. Nothing else about
  the interface changed, and no other kind moved.

### Added

- **A pipeline names the network it prepares its rows for: `.WithTorch(…)` and `.WithTensorflow(…)`.** Below the
  output, the chain says which network these rows train — `.Dense(16).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy()
  .Run(seed, epochs).StoppingAfter(patience: 10)` — and the whole thing is one declaration. Both doors write the same
  step and lower onto the same model; where a caller leaves a number unsaid, each door writes the number its own
  library leaves there, so the file says which run was asked for rather than which library asked for it. The words
  TensorFlow and Keras use run on libtorch and on the light engine alike: a vocabulary is a way of speaking, never an
  engine.

- **`pipeline.Train()` trains what the declaration names.** It runs the pipeline for the learner it names — leaving
  out the steps that learner does without, as `RunFor` always has — trains the network, measures it as the pipeline's
  report declares, and hands back the same `TrainedNetwork` a network written as code gives. `prepared.Train()` does
  it for a run already in hand. A pipeline that names no learner says so rather than guessing one.

- **The engine is named in the declaration, and the application says what that name stands for.** A step names an
  engine — `light` unless said — and `new Engines().Use("torch", TorchBackend.OnCpu())` hands that name an engine
  before the run. So one declared model runs on the light engine on a machine with nothing installed and on libtorch
  where libtorch is there, without a word of the file changing; a name nothing was given is refused by name, with what
  to do.

- **`learn.network`, the verb a pipeline file writes it as.** Its layers are written as a list of named parts, each
  with the settings that name takes, in the same words a saved network uses — `dense` with its `units`, `adam` with
  its `rate`, `binaryCrossEntropy` — so a declaration and the network it trains speak one language. Every setting a
  name takes is written down: a file that leaves one out is refused where it is read, rather than filled in behind the
  writer's back.

- **A declaration may name the learner it is written for: `INamesTheLearner`.** A step that implements it changes no
  column and runs nothing — the package that brings the verb trains once the pipeline has run — and says one thing the
  pipeline itself uses: what that learner needs of the features it is handed. A declaration has at most one, below the
  output it learns to answer, and `PipelineDeclaration.Learner` is where every part that needs it asks.

- **`PartsParameter`, the kind of parameter that holds the parts of a model.** A step says which names a part may give
  and what each takes, and the file, the JSON schema, the reference page and a notebook's form all describe those
  parts through the kinds every other parameter is described through. It is the first parameter whose value is not one
  value, and it is written as a list, or as itself where a step takes one.

- **The notebook holds a learner block.** Its catalog knows `learn.network`, its form draws a part's name and the
  settings that name takes, and typing a part offers the names there are. The sample notebook now declares the network
  the passenger list trains, and its C# cell trains that one with `Train()` rather than writing a network of its own.

### Changed

- **The networks sample declares the passenger list's network in the chain.** Its other two walked networks — a price
  five days on, a day's bikes hour by hour — are still written as code, so both doors stand side by side in one
  sample.

## [0.5.0]

Where it runs on libtorch and reads what others trained, and where every gap 0.4.0 left between a trained network and a
person using it is closed. A network trains, is measured and serves on the engine you hand it — the light one, or
libtorch on the processor or a graphics card — and a model saved by PyTorch, Keras or an ONNX exporter is read into the
same network. A pipeline reads Parquet files, Excel workbooks and JSON files, and is run for the learner that learns
from it, saying which steps that learner does without. The notebook's report block draws what the cell that trains a
model hands back, and a sample notebook ships.

### Upgrading from 0.4

- **`CheckpointFile.Read` holds a checkpoint's network to the pipeline the file carries, as `TrainedNetwork.FromJson`
  always has.** A checkpoint whose pipeline was taken out or replaced by hand went on whenever the pipeline handed over
  was the one its network was trained behind; it is now refused with a `NetworkFileException` at its pipeline. So is a
  checkpoint whose pipeline names a version of the pipeline file below the third, which no network's file ever carried,
  or above the one this DeepSharp reads. A checkpoint `CheckpointFile.Write` wrote is read as before.

- **`CheckpointFile.Read` reads the whole file before it holds it to the pipeline handed over.** A checkpoint whose own
  training part is wrong, handed a pipeline that names other answers than its network was trained for, is refused with
  the `NetworkFileException` that names the file's fault, where 0.4 refused it with the `ArgumentException` that names
  the pipeline. A file that is right is held to the pipeline as before.

- **An engine handed to a run is asked for the report's measures too.** The engine in `FitOptions.Backend` was asked for
  the run's training passes and its looks at the validation rows; it is now also asked to evaluate each part the report
  names, once the last epoch is over, a chunk of the part's rows at a time. An engine of your own whose arithmetic
  differs from the light one's can move the measures in their last digits from what 0.4 reported for the same run.

- **The report and a trained network serving rows ask an engine of your own a chunk of rows at a time, never all of
  them in one pass.** The report's chunks are the run's own `FitOptions.BatchSize`; a served row's are thirty-two,
  Keras's own default for a model's `predict`. Answering twenty sine rows with a report of two parts made an engine that
  counts what it is asked twelve operations under the run's default batch size, as before; under a batch size of three
  the same report makes it thirty-six, and a hundred rows served at once, which made an engine one pass regardless of
  how many rows they were, now make it four. No answer moves: on the light engine a chunked answer comes to the bit what
  one whole pass over the same rows makes, and every engine's own contract now holds that within the tolerance it is
  already held to against the light one.

- **A network's file is written with the second version of its network's part, which 0.4 refuses as newer.** The part
  now records which features held one value on every training row, and a reader that does not know the record names the
  newer version rather than a key it cannot read; a checkpoint's training part names the same version. A file 0.4.0
  wrote is read as before and records nothing of it: its `TrainedOn.Unvaried`, and the `Unfamiliar` of what it predicts,
  are null — not recorded, which is not the same as nothing unfamiliar — until the network is trained again, or goes on
  from its checkpoint, under this DeepSharp. `TrainedOn`, `Predictions`, `PartPredictions` and `PartMeasures` each gain
  a member, which the first three, being records, compare as well when asked whether two are equal.

- **A run goes on from a checkpoint only under the batch size and the early stopping it was taken under.** A checkpoint
  records them beside its seed, kept in memory or written to its file, and `Fit` refuses a run whose options differ from
  what its `ResumeFrom` records with an `ArgumentException` that names each difference, both values of it, before
  anything is put back. Such a run went on without a word, as another run than the one the checkpoint was taken of: hand
  it the batch size and the early stopping the first run had. The epochs are not among them, since going on to more of
  them is what a resume is for. A checkpoint 0.4.0 wrote records neither, and goes on under whatever it is handed, as it
  did; the checkpoints of the run that goes on record what that run was handed.

- **A run goes on from a checkpoint only on the engine it was taken on.** A checkpoint records the engine its run was
  on — its name, and, for an engine that names them, the version of what works its arithmetic out and the device it
  works it out on — and `Fit` refuses a run handed another engine than its `ResumeFrom` records with an
  `ArgumentException` that names both, before anything is put back. The light engine names DeepSharp's own version, so
  a checkpoint a run on it took is refused by the light engine of another release of DeepSharp; the libtorch engine
  names the libtorch it runs on and its device, so a checkpoint taken on libtorch is refused on another libtorch, and
  one taken on the processor on a graphics card. Hand the run that goes on the engine the first run had. A checkpoint
  0.4.0 wrote names no engine, and goes on on any, as it did; the checkpoints of the run that goes on name the engine
  that run was handed.

- **`Needs.Numbers` is the need of a learner that takes every step as declared; a tree states `Needs.NoScale`.** The
  description of `Numbers` named two needs at once, and its half about scale is now `NoScale`. `Numbers` and `OneScale`
  keep their values, 0 and 1, and a batch handed over for `Numbers` holds what it held. For a tree whose scales should be
  left out, run the pipeline with `RunFor(Needs.NoScale)` and hand the batch over with `NoScale`.

- **Test a need by what it says.** `Needs` gains `NoScale` (2) and `Categories` (3). A `switch` that named only two
  members now meets values it did not name. `NeedsOneScale()`, `DoesWithoutScaling()` and `TakesCategories()` say what
  each member means. A value no member names is refused with `ArgumentOutOfRangeException`; 0.4 handed such a batch over
  as for `Numbers`.

- **A pipeline file, and the file of saved columns beside a notebook, name version 4, which 0.4 refuses as newer.** A run
  that leaves no step out writes nothing but the new number, so its fit is the fit 0.4.0 wrote, and networks and
  checkpoints from 0.4.0 go on behind it. Every file 0.4.0 wrote is read as before. A file of an earlier version that
  names `skipped` is refused at that key.

- **The report reads only the keys and the answers of the rows it measures.** It used to go through `Batch(part)`, which
  refused words or gaps among the features although the report measures none of them.

- **Eight members 0.4.0 published with five or six parameters have a shorter form, and the long one is obsolete.** No
  method or constructor in a DeepSharp package takes more than four parameters now, and a test reads every package to
  keep it so. The long forms still compile and do what they did, with a warning that names the form to call instead; a
  build that treats warnings as errors stops at each until it is moved:
  - `new PreparedData(declaration, table, parts, fitted, evidence)` → `new PreparedData(declaration, table, parts, fitted)
    { Evidence = evidence }`.
  - `schema.Column(name, kind, optional, format, missing)` → `schema.Column(new ColumnDeclaration(name, kind, optional)
    { Format = format, Missing = missing })`: the column is its declaration.
  - `pipeline.AddIndicator(name, indicator, columns, period)` → `pipeline.Add(new AddIndicatorStep(name, indicator, columns,
    period))`: the step says all of it.
  - `new ColumnParameter(key, description, example, accepts, optional: true)` → `new ColumnParameter(key, description,
    accepts)`, a column the step can do without, which starts a new block with none; a column the step needs is
    `new ColumnParameter(key, description, example, accepts)`.
  - `new ColumnsParameter(key, description, example, accepts, optional, repeatable)` → `new ColumnsParameter(key,
    description, example, accepts) { Optional = optional, Repeatable = repeatable }`.
  - `new NumberParameter(key, description, example, above, atLeast)` → `new NumberParameter(key, description, example)
    { Above = above, AtLeast = atLeast }`.
  - `new WholeNumberParameter(key, description, example, atLeast, leftOut)` → `new WholeNumberParameter(key, description,
    example) { AtLeast = atLeast, LeftOut = leftOut }`.
  - `new FillStrategyParameter(key, description, example, allowed, what)` → `new FillStrategyParameter(key, description,
    example, allowed) { What = what }`, which a new form must say.

- **An application that references `DeepSharp.Verso.Notebooks`, or `DeepSharp.Verso.Api`, takes the three readers with
  it**, and with them Parquet.Net 6.1.0, ExcelDataReader 3.9.0 and the libraries they bring. An application that pins
  another version of either decides which one it runs, as with any package it shares.

- **A Jupyter or Markdown file Verso's converter made of a notebook of blocks is refused when it is opened**, as a file
  of another format holding steps as code cells already was; 0.4 opened it with every block turned into text that runs
  nothing.

- **`FitOptions.Backend` is never null: options hold the light engine unless they name another.** It read null when no
  engine was named, and the loop and the report each made a light engine of their own. The options now say which engine
  a run takes, the light one from the start, and null handed to them leaves them holding it. Code that read null as "the
  light engine" reads that engine.

- **A part of a pipeline read from its file is refused, saying the pipeline holds no rows.** Its file keeps what the
  steps learned, not the rows, and `Batch` on it said that the answer's column never reached the end of the pipeline —
  or, for a pipeline that names no answer, handed over a part of no rows without a word — and `Fit` behind it said "This
  pipeline predicts 'survived', and no column of that name reached the end of it", which sent a person looking for a
  column that was there. Both now say that the pipeline holds no rows and how to have them — run its declaration over
  them, `new Pipeline(declaration, rows, folder).Run()`, and train behind that run — as the report's measuring already
  said; serving rows through it is unchanged.

- **A fit whose encoder writes one column per category refuses a category that would take a column of the encoder's
  own.** Keeping a place for a category the training rows never held, the encoder names that place's column after the
  encoded one and `other`, and the column that marks a gap after it and `was_missing`. A category called `other` shared
  its column with the kept place, and one called `was_missing` with the mark, and their rows lost their category without
  a word: a training row holding `other` came out as a row with no category. Such a fit is now refused, naming `unseen: refuse`
  and `as: ordinal`, which write no such column. A pipeline file fitted before this replays as it was written.

- **Working back from a loss asks the engine only for the way back to what was asked for.** `RecordingBackend.GradientsOf`
  worked out a share for every tensor the pass read, the rows of a batch among them, though nobody asks how the rows
  moved the loss: a convolution's way back folded a gradient onto its images, and a dense layer's took a product for its
  rows. A share is now worked out only for a tensor asked for, or one the pass made from such a tensor — as libtorch
  records an operation only when one of the tensors it reads wants a gradient. Every gradient asked for is the same, to
  the bit. An engine of your own that counts what it is asked sees fewer operations: a Titanic training step's way back
  asks for 19 where it asked for 22, a convolution step's for 25 where it asked for 33, and no fold among them. On the
  light engine the Titanic step takes 19.5 µs where it took 27, and the convolution step 11.9 ms where it took 15.2 on
  .NET 10 and 11.8 where it took 14.2 on .NET 8.

- **A stack that ends in the sigmoid its loss applies itself is refused where it is compiled.** `network.Compile(optimizer,
  new BinaryCrossEntropy())` on a stack whose last layer is a `Sigmoid` — a stack within it read down to its own last
  layer — throws an `ArgumentException` that names the layer. Leave that layer out: the loss takes the logits and applies
  the sigmoid itself, so the predictions still come out as chances. A network written as code is compiled as it is
  written, since nothing here reads its forward pass. A file that holds such a pair, as 0.4.0 wrote it, is read, served and
  gone on from as it was written.

- **A description in Keras's words that ends in `.Sigmoid()` before `BinaryCrossEntropy` builds its network without
  it.** Compiling the description lifts the last sigmoid into the loss, so the network ends at the layer before it and its
  file holds no sigmoid; before any other loss the sigmoid stays. `Lower(shape, stream)` still keeps every word, so a
  description lowered by hand to be compiled with such a loss is written without its last `.Sigmoid()`. A description of
  nothing but that sigmoid is refused at `Compile`.

### Added

- **A trained network serves on the engine you hand it.** `trained.Predict(rows, engine)` works its answers out on that
  engine, for that call alone, and `trained.Predict(rows)` goes on serving on a light engine of its own. The engine is
  the caller's and nothing keeps it: the network holds none and its file names none, so a network trained on one engine
  is read back from its file and served on another, and a trained network's file is the same whichever engine did its
  arithmetic; only a checkpoint names the engine its run was on.

- **An engine of your own keeps a tensor's values where it keeps them.** `TensorStorage` is where a tensor's values live
  — how many it holds, and `CopyTo`, which copies them into memory you own. An engine that keeps its values somewhere of
  its own, memory it allocated outside .NET or a graphics card, derives one and hands back what it makes as
  `Tensor.On(shape, storage)` — a name of its own, so every call that makes a tensor from values compiles as it did;
  `tensor.Storage` says where a tensor's values live, so an engine knows its own when a tensor comes back to it and
  takes any other in. `tensor.Values` copies the values of a tensor on such a storage into
  this machine's memory the first time they are read, once — whoever reads them first, while any reading at the same
  moment waits for that copy — and keeps the copy, so reading them again copies nothing; it is never a view of the
  engine's own memory. Nothing is added to `ITensorBackend`, and a tensor made here holds its values as it did, at
  sixteen bytes more: 1.8 KB more of what a Titanic training step on the light engine allocates, in the same time.

- **What the seam refuses is public, and every engine says it first.** `TensorOperandExtensions` holds each check an
  operation of `ITensorBackend` makes of the tensors it is handed, in the words the light engine refuses them with —
  `left.RequireSameShape(right, "Add")`, `left.RequireMatrixProduct(right)`, `matrix.RequireRow(row)`,
  `values.RequireValues()` for a mean, `patches.RequirePatchesOf(images, window)` and the rest. The light engine calls
  them, and an engine of your own that calls them before its arithmetic refuses what the light one refuses, with the
  same exception, in the same words, naming the same argument — a column added to one value, or the mean of nothing, is
  refused there as it is on the light engine, rather than stretched to fit or answered with something that is not a
  number.

- **What every engine is held to is written once, and measured.** The tests of the seam, the layers, the losses, the
  optimizers, the loop and the network's file take the engine they run on, and run on the light engine and on one whose
  every value lives in memory allocated outside .NET and whose totals are single precision, as libtorch's are. An
  engine's totals lie within three roundings of the size of their terms of the light engine's — three times 2⁻²⁴ times
  the sum of the terms' sizes — and a value worked out value by value within PyTorch's float tolerance, a relative
  1.3e−6 and an absolute 1e−5. One Titanic training step is held the same way: on that engine its gradients came within
  1.07 roundings and its parameters after Adam within 1.5e−8. The promise is per step, not per run: the networks
  sample's Titanic run kept the same epoch on both engines and ended with weights 0.08 apart. What stays alive of an
  engine's memory is what something holds — the network's slots, what the optimizer remembers, the best epoch's slots
  and the checkpoints you keep, each with the best epoch's slots as they stood when it was taken: at the tenth and the
  fiftieth epoch of a Titanic run keeping every fifth checkpoint, exactly those were alive.

- **A network runs on libtorch, on the processor or a graphics card, without a line of it changing.**
  `DeepSharp.Backends.TorchSharp` is an engine on libtorch through TorchSharp 0.107.0: `TorchBackend.OnCpu()` or
  `TorchBackend.OnGpu(0)`, handed to a run as `new FitOptions(seed) { Backend = engine }` — which trains the network,
  judges it by its validation rows and takes its report's measures on it — and to a trained network as it serves,
  `trained.Predict(rows, engine)`. The network holds no engine and its file names none, so one trained on libtorch is read
  back and served on the light engine, and one trained on the light engine is served on libtorch. Every engine contract
  holds on it: on the processor a product came to 2.24 roundings of its terms' size at most, and every contract held on
  an RTX 5070 Ti as well. The networks sample's Titanic run went the same way on it as on the light engine — forty-six
  epochs, the thirty-sixth kept, 0.815 of the test passengers right — its weights 0.08 apart at the end, the promise being
  per step, not per run. What the engine makes stays in libtorch's memory, taken out of any dispose scope of TorchSharp's
  the moment it exists, so a scope of yours around a run ends without touching the network's slots or a checkpoint you
  keep; it is let go of once nothing holds it, the collector told how many bytes each tensor holds, so five hundred
  dropped tensors of four megabytes were at most eleven alive at once. The package brings TorchSharp alone; your
  application brings libtorch — `libtorch-cpu-win-x64`, `-linux-x64` or `-osx-arm64` 2.10.0 for the processor,
  `TorchSharp-cpu` for all three, or `TorchSharp-cuda-windows` or `TorchSharp-cuda-linux` for an NVIDIA card — and
  making the engine where there is none says which to add. It never sets libtorch's threads, which stay yours, nor draws
  from libtorch's generator; two hundred Titanic steps came to the same bits at 1, 4, 16 and 32 threads. Measured side by
  side on one machine, a Titanic training step took nearly nine times as long on libtorch as on the light engine, which
  stays the faster at that size; a convolution's step took under a third of the light engine's time on one thread, and
  under a fifth on sixteen. On an RTX 5070 Ti the sample's run ended 2.4e−7 from the light engine's weights; a Titanic
  step there took thirty times the light engine's time and a small convolution's about what libtorch's sixteen threads
  take on the processor, while a convolution over eight times the images with four times the filters took under a fifth of
  libtorch's time on the processor and a sixty-ninth of the light engine's. Two thousand convolution steps left the
  card's memory where the first thirty had.

- **An engine names its version and the device it works on: `INamesItsVersionAndDevice`.** An engine that implements
  it, beside `ITensorBackend`, which it extends, says `Version` — the version of what works its arithmetic out — and
  `Device` — where it works it out — and a checkpoint of a run on it records both with its name. `CpuBackend` names
  DeepSharp's own version and `cpu`; `TorchBackend` names the libtorch it runs on as TorchSharp states it, 2.10.0.0 for
  TorchSharp 0.107.0, and `cpu`, or `cuda:0` for the first graphics card. An engine of your own that does not implement
  it is recorded by its name alone, and held to that.

- **A served row that moves a feature every training row held at one value is named, and the report counts such rows.**
  A network learns nothing about a feature its training rows never varied: on the wiki's Titanic network `pclass_other`,
  `pclass_was_missing`, `sex_other` and `sex_was_missing` are nought on all 623 training rows, and the first layer's
  weights for them are still the random start's after training. A passenger written 'Male', of a fourth class or of no
  sex given was answered from that start, by the seed rather than by anything learned — across six seeds the same
  passenger was given 0.257 to 0.495 as 'Male' and 0.290 to 0.509 with no sex, against 0.162 to 0.186 as 'male', while
  test accuracy stayed between 0.785 and 0.815 — and nothing said so. `TrainedOn.Unvaried` now records each such feature
  with its value, and the network's file keeps it; `trained.Predict(rows).Unfamiliar` names, for each served row, the
  features it moves away from that value — `sex_other`, `pclass_other`, `sex_was_missing` — and an empty list for a row
  that moves none; and each part the report measures counts its rows that do, as `PartMeasures.UnfamiliarRows` beside
  its `Rows`. A learner of your own says the same of its rows through `PartPredictions.Unfamiliar`. The answer itself is
  unchanged: a pipeline that should answer no category its training rows never held declares
  `EncodeCategories(unseen: Unseen.Refuse)`.

- **A notebook's report block draws the measures of a model a C# cell trained.** `Measures.PredictionsToJson()` writes
  what a report's measures were taken from as text — the fit the predictions were made behind, and for each part the key
  of each row, what was predicted for it in the units the answers were handed over in, and what the model said it learned
  nothing about — and `PreparedData.MeasureAgain(text)` measures that text again, by the report, on another run of the
  same pipeline, and only beside the very fit the predictions were made behind: held to it as a network's file is held to
  its pipeline, predictions made behind other steps or other learned numbers — a feature or the answer scaled another way,
  another split — are refused, as are those of another output, named by its answers, and of other rows, known by their
  keys. In a notebook, the cell that trains hands them back with
  `Variables.Set("deepsharp.predictions", trained.Measures!.PredictionsToJson())`, and "Show the data here" at the
  `evidence.report` block measures them on the notebook's own run of its blocks and draws them under the grid, as the
  report says they are shown; a hand-back from before a block was edited is refused there, saying to run the cell that
  trains again, and with nothing handed back the block names its measures and says where they come from. Shown again, or
  paged, it measures nothing again while the blocks, the file and the predictions stay as they were, and the toolbar's run
  ending at the report fits the pipeline once for the fit it hands over and the measures it draws.

- **`PipelineText`, in `DeepSharp.Pipelines`.** A pipeline's text as the one writer writes it, and the rule two pipelines
  are held to one fit by, public beside the pipeline: `Text`, its `Version`, the `Digest` a network records, the
  `FitDigest` two pipelines are compared by — the digest of the text with its version left out — `IsTheFitOf`, and
  `IsComparable`, the versions of the pipeline file two texts may be compared across, from `FirstComparable`, the one the
  first network's files carry, to the one this library writes. A network's file, a checkpoint and predictions handed back
  are all held to a fit by it.

- **The report, rendered once.** `measures.Report()`, in `DeepSharp.Charts`, is the measures as their report says they
  are shown — the numbers, the charts or both — as one value whose `ToHtml()` gives the HTML. Verso shows such a value as
  HTML, so a C# cell that ends with `trained.Measures!.Report()` shows the report, and the report block draws the same
  rendering. A cell that ends with a chart, such as `trained.History!.LossCurve()`, shows the picture, as Verso shows the
  text of any SVG.

- **A pipeline is run for the learner that learns from it, leaving out the steps that learner does without and saying
  which.** `Needs` states each thing a learner can need of its features. Besides `Numbers`, every feature a number on the
  scale its steps declare, and `OneScale`, a learner indifferent to scale states `NoScale`, and one that takes categories
  itself, as a boosted tree that splits on them does, states `Categories`. `pipeline.RunFor(needs)` runs the declaration
  for that learner. A step that only scales a feature is left out for a learner indifferent to scale. Each encoder hands
  its categories over as themselves: the place of each in the list the training rows held, the place after the last for
  one they never held, and minus one for a gap. It learns the same list it learns for every learner. A step an answer's
  way back runs through, a step whose column a step below reads, and a scale that refuses what it was not fitted on are
  always taken. A step of your own that some learners do without says so by `IMeetsANeed`: `NeededBy(needs)`, whether a
  learner of that need takes it as declared, and `Instead`, what a run for the others takes in its place, or nothing.
  `PreparedData.Skipped` lists the steps a run left out. Its file writes them under `skipped`, so a replay
  leaves out the same steps. On the wiki's Titanic pipeline, a run for `Categories` leaves out `encode.categories` and the
  four `normalise` steps. A passenger is handed over as nine numbers: age 29, fare 8.4583, class as place 2 of 3, sex as
  place 1 of 2. The network's run hands over fourteen. Both runs share the split, the filled age and the categories
  learned, so the report measures both learners on the same 623, 133 and 135 rows. `Batch.Categories` and
  `ServedBatch.Categories` name each feature handed over as a place, with the categories the training rows held.
  Predictions handed back as text from such a learner are measured again on a run of every step. `Run()` takes every
  step, as before.

- **A learner is handed only a run it can take.** `Batch` and `Served` refuse a run that left out a step the learner
  needs, naming each step. A network is refused a run made for `NoScale` or `Categories`. A run of every step goes to
  every learner.

- **Parquet files, read.** `DeepSharp.Pipelines.Parquet` adds `.ReadParquet(path)` to the chain and `read.parquet` to
  the pipeline's file, a file the pipeline reads and replays as it does a comma-separated one, from the pipeline's
  folder when its path is relative. A Parquet file types its columns, so it says what each holds, as a database does,
  and the proposal of kinds takes what it says. Its values reach the pipeline by the one rule the database door already
  handed a value over in — a number in its shortest exact form, a moment or a day as ISO 8601 writes it — now public
  as `TypedValueExtensions.AsCell` and `AsColumnKind` for a reader of your own; a column of lists, of groups of fields
  or of raw bytes is refused by name. The passenger list written as a Parquet file with the cells its comma-separated
  file holds gives the same keys, the same split and the same batches.

- **Excel workbooks, read.** `DeepSharp.Pipelines.Excel` adds `.ReadExcel(path)` and `.ReadExcel(path, sheet)` to the
  chain and `read.excel` to the pipeline's file, for `.xlsx`, `.xls` and `.xlsb`: the first sheet unless one is named,
  its first row naming the columns, a sheet it does not have refused with the names of those it has. A cell is read as
  the sheet types it, never as the text a culture shows it as — a date as a moment — an empty cell is empty text, as
  the same sheet saved as comma-separated text holds it, and a formula's error is spelled as Excel spells it there. The
  library it reads with asks .NET for a Windows code page before it opens any workbook, `.xlsx` too, so the package
  offers .NET's own code pages, once for the whole program, and every encoding the program had stays as it was. A step
  of your own can take words it does not need, as the sheet is: `new TextParameter(key, description, example,
  optional: true)`. The passenger list saved by Excel as `.xlsx` and as `.xls` gives the same keys, the same split and
  the same batches as its comma-separated file.

- **JSON files, read.** `DeepSharp.Pipelines.Json` adds `.ReadJson(path)` to the chain and `read.json` to the pipeline's
  file, for an array of records, one object a row, as an API hands them back; .NET reads JSON itself, so the package
  brings nothing else. A value is read as the file writes it — `22.0` stays `22.0` — the columns are the keys in the
  order they first appear, a key a record leaves out and a null are both gaps, and a value that is itself an object or a
  list is refused by name, with its record. The passenger list written as JSON gives the same keys, the same split and
  the same batches as its comma-separated file.

- **Taking a preset over lists the columns every reader's file has.** A chain that takes saved decisions over at a
  Parquet file, a workbook or a JSON file names the file's columns the decisions never showed, as it did for rows
  handed in and for a comma-separated file; a source step of your own says its columns by `INamesItsColumns`, and one
  that does not is asked nothing, as before.

- **A notebook reads a Parquet file, an Excel workbook and a JSON file.** A notebook's first block can be `read.parquet`,
  `read.excel` or `read.json`, as a pipeline file's can: `DeepSharp.Verso.Notebooks` brings the three readers with it, as
  it brings the indicators and the charts, in every place it runs. The data at a block, the list of the columns, the
  grid's boxes, the form, the take-over, the run, the export and the report work over them as over a comma-separated
  file, and the passenger list in each of the four formats shows the same rows under the same keys in the same parts.
  The notebook parses the rows from the very bytes it fingerprints, whichever file it reads, so an export keeps a fit
  only while the file holds the bytes it was learned from, and a Parquet file's list says what the file states each
  column holds. `IReadsAFile` is how a reader of a file does that: `step.Open(bytes, file)` reads the rows from bytes
  already in hand, and `bytes.AsText()` reads them as a file is read as text; a Parquet file's columns are named from
  its schema, without reading a row. The readers add about 1.8 MB of managed code to an install of the notebook, 1.5 MB
  of it Parquet.Net and the compression libraries it reads with. A C# cell that reads the pipeline the blocks hand over
  brings the same packages and teaches its catalog their verbs:
  `StepCatalog.BuiltIn().WithIndicators().WithParquet().WithExcel().WithJson()`.

- **One evaluation for every door, in the core: `network.Predict(features, loss, engine)`.** What a network answers for
  rows is one evaluation pass on the engine handed in, through the loss's output activation. A compiled network's
  `Predict`, the report's measures and a trained network serving rows all answer through it — the last two a chunk of
  rows at a time — so no door writes the evaluation its own way.

- **Numbers trained somewhere else go into a network by the path of each slot, and `IImporter` is the seam that reads
  them.** `network.Load(entries)` puts a tensor into each slot its path names — `1.weight`, `1.running_mean` — as
  PyTorch's `load_state_dict` does with `strict=True`: every slot of the network once, none it does not have, each of
  its slot's shape and holding finite numbers. It puts in all of them or none: whatever is wrong — a slot left out, a
  path the network has no slot at, a path handed twice, another shape, a value that is not a finite number — is refused
  at once with a `SlotLoadException`, whose `Faults` name each by where its file holds it, the slot and what is wrong,
  and the network keeps what it held. From outside the library a batch normalisation's scale and shift could be set only
  by writing a network's own file and reading it back; now any reader puts them in.
  `IImporter` is what a reader of another framework's files implements: `importer.Read(stream)` hands back a
  `SavedNetwork`, as reading a network's own file does, whether the importer is handed the network the numbers belong to
  or builds it from the file's own description. A network's own file is held to the same rule, and refuses in the same
  words.

- **Models Keras saved, read: `DeepSharp.Import.Keras`.** `new KerasFile().Read(stream)` reads the `.keras` archive Keras
  3 saves a model to, or the HDF5 file, `.h5`, it saved one to before, and hands back a `SavedNetwork` — the network the
  file describes, every slot holding Keras's numbers, and the loss the model was compiled with — as any importer does.
  Each layer is built as Keras's words build it here: a batch normalisation keeping Keras's epsilon and the complement of
  its momentum, a window padded as TensorFlow's 'same', the activation a Keras layer carries a layer of its own after it,
  and a last sigmoid or softmax the loss applies itself lifted into that loss. A layer's numbers are found by the name the
  file gives them, since Keras files them under the layer's class rather than its name. The networks sample's Titanic
  network, trained in Keras 3 on the pipeline's training rows and saved both ways, answers the 135 test passengers within
  five roundings of a single-precision number of Keras's chances, and the two passengers the sample serves within two; a
  network of every kind the reader builds answers within six. What no network here is built of is refused at the layer
  that says it, every such layer at once — a stride of its own for each axis, a dilated window, channels in groups,
  pooling, any other kind of layer or activation, a layer without a bias, a model of another kind than a Sequential, in
  another precision, or saved without its loss — and numbers that do not fit its layers are refused by `network.Load` at
  their datasets' paths. The package brings PureHDF 2.2.0 (MIT), a managed HDF5 reader, and nothing else; the models the
  tests read are kept with them, with the scripts that made them.

- **Networks exported to ONNX, read: `DeepSharp.Import.Onnx`.** `new OnnxFile(loss).Read(stream)` reads the graph
  PyTorch's `torch.onnx.export` writes — by its default exporter or by the TorchScript one before it — the graph Keras
  writes with `model.export(format="onnx")`, and the graph tf2onnx converts a TensorFlow SavedModel into, and hands back a
  `SavedNetwork`: the network the graph is, every slot holding its numbers, and the loss handed in, since a graph names
  none. The graph is lowered onto the layers it is — a Gemm, or a MatMul and the Add of its bias, a dense layer; a Conv; a
  BatchNormalization with its momentum and epsilon; a LayerNormalization; a Flatten or a Reshape that flattens each
  example; a Relu, a Tanh or a Sigmoid — a Cast into single-precision numbers is nothing, and a last Sigmoid or Softmax the
  loss applies itself is lifted into that loss. The numbers are turned as each node declares them: a weight matrix Gemm
  says is written outputs by inputs turned round, a kernel written channels first laid out as the slot keeps it, and the
  numbers along an image flattened channel by channel, as ONNX flattens one, turned into this library's order, place by
  place. Images go in, and come out, with their channels last: a graph that takes them channels last, as tf2onnx's does,
  declares so with the Transpose that moves them after the batch for its convolutions, and the one moving them back before
  its flatten is nothing here; a convolution's pads written out one side at a time are read as TensorFlow's 'same' when
  they are the border 'same' gives the image reaching it. Numbers of half precision or bfloat16 are widened exactly,
  numbers of double precision rounded to the nearest single one, and those PyTorch's default exporter keeps in a file
  beside the graph are read from beside the graph's own file. Measured: the Titanic network PyTorch trained on the
  pipeline's training rows answers the README's passenger within one rounding of a single-precision number of PyTorch's
  chance, and all 135 test passengers within five, from either PyTorch exporter's file; the Titanic network Keras trained,
  exported by Keras or converted by tf2onnx 1.17.0 from TensorFlow 2.21.0's SavedModel, answers the 135 within five of
  Keras's chances and puts the numbers Keras saved into the slots its `.keras` archive does, bit for bit; a network of
  convolutions, batch normalisations and a flatten gives each image PyTorch's output within 4.5e−8, and one of 'same' and
  strided convolutions and a flatten tf2onnx converted from a SavedModel of a fixed batch gives each TensorFlow's within
  2.4e−7. What no network here is built of is refused at its node, every one at once: a branch — a node taking another
  value than the one the node before it made, or two — pooling, a stride of its own for each axis, a dilated window,
  channels in groups, a Cast into another type, a Transpose other than those two, any other operator — among them the Mul
  tf2onnx writes a batch normalisation as, and the Shape, Gather, Slice and Concat it works a flatten's target out with for
  a SavedModel of batches of any length. Numbers that do not fit are refused by `network.Load` at the initializer that
  holds each. The package brings OnnxSharp 0.3.2 (MIT), the messages of ONNX's own format for .NET, and the
  Google.Protobuf 3.29.3 (BSD-3-Clause) it reads them with; the graphs the tests read are kept with them, with the scripts
  that exported them.

- **A sample notebook, `Samples/titanic.verso`.** It reads the Titanic passenger list from the samples' data beside it,
  block by block — the columns declared, the split, the gaps filled, the categories encoded, the scales, the answer and
  the report — and its C# cell trains a network on the pipeline the blocks hand over and hands the predictions back, for
  the report block to draw. `deepsharp-serve Samples/titanic.verso`, from the repository's folder, opens it, and a test
  runs it as a person does.

- **Make blocks of the steps: a button that makes blocks again of the steps another format kept as text.** A notebook
  saved as Jupyter comes back with its blocks as raw cells, one saved as Markdown with them fenced into a cell of text,
  and a Jupyter file another program wrote may hold them as code. Verso's browser editor opens such a file without
  asking the guard that refuses it, and saving it there writes those cells into a `.verso` file as they are. The
  notebook's toolbar now has a button, on wherever a cell holds such a step, that puts a block in each step's place —
  the text around them staying text, every other cell as it was — so the notebook runs again, before it is saved or
  after. It is the same button in every editor the notebook runs in, it runs nothing, and the guard and the button
  read a cell by the one rule.

- **A window pads as TensorFlow's 'same' does.** `new Window(3, 3) { Stride = 2, PaddingMode = PaddingMode.Same }`
  works its border out from each image as TensorFlow's and Keras's `padding='same'` does: along each axis as many places
  as the stride fits into the image, the rows or columns those need beyond it split in two, the odd one after. No border
  stated for every side gives that: through a window of three at a stride of two over 28 by 28, TensorFlow pads none before
  and one after, and a stated border of one gives the same fourteen by fourteen places with values up to 4.56 away;
  through a window of two, a stated border gives 27 rows or 29 where 'same' keeps 28. The convolution comes within 1.2e−7
  of TensorFlow's rule on the light engine. `window.BordersOver(rows, columns)` says the border on each side, as a `Borders`
  of `Top`, `Bottom`, `Left` and `Right` — the one rule every engine's unfolding and folding reads, so an engine of your
  own honours 'same' by starting its first patch where it says, and the tests' native engine, written outside the
  library, does. A network's file writes such a window's padding as the word `"same"`, as Keras writes it. What a window
  covers otherwise is one stride down and across alike, a patch of neighbouring places and every channel at once: a file
  that says a stride for each axis, a dilation or groups of channels is refused at that setting, and a pooling layer at its
  kind.

- **Keras's words take Keras's settings for the normalisations.** `.BatchNorm(momentum, epsilon)` takes Keras's momentum
  — the share of the running statistics a training batch leaves as they were, 0.99 in Keras — and the layer keeps its
  complement, PyTorch's meaning of the word, with Keras's epsilon; `.LayerNorm(epsilon)` takes Keras's epsilon.
  `.BatchNorm()` and `.LayerNorm()` stay PyTorch's: a momentum of a tenth and an epsilon of a hundred-thousandth. A network
  Keras 3 described and ran — windows padded as 'same' at strides of two and one, a batch normalisation of momentum 0.9 and
  epsilon 0.001, a layer normalisation of epsilon 0.001 and a last sigmoid — written here in Keras's words, holding Keras's
  numbers in Keras's own order, answers as Keras did within 1.8e−7 on every logit and 9e−8 on every chance, on the light
  engine and on the tests' native one; with PyTorch's epsilons its chances lay up to 0.0099 away. The fixture is kept with
  the tests, with the script that made it. Trained here, such a batch normalisation still counts each batch's variance
  over one row fewer, as PyTorch does, where Keras counts it over every row: after one batch of 96 places its running mean
  came within 1.9e−9 of Keras's and its running variance lay up to 8.4e−4 above it.

- **A network PyTorch trained runs here: `DeepSharp.Import.PyTorch` reads a safetensors file into the same network.**
  `new SafetensorsFile(network, loss).Read(stream)` puts every tensor of a file PyTorch saved a network's state in into
  the slot its name names — a `LayerStack` in Keras's words or written by hand, its layers numbered as PyTorch's
  `Sequential` numbers its modules, or a network written as code, its layers named as the PyTorch module named them — and
  hands back the network and its loss as a `SavedNetwork`, to serve, to measure, to train further or to save as a
  network's own file. Each number is turned into the layout its slot keeps, as the layer
  holding the slot says, never as its shape suggests: a linear layer's weights turned round, a convolution's kernel laid out
  channels last, biases and a normalisation's numbers as they are, and the count of batches PyTorch keeps beside a batch
  normalisation left out by its name. The rows a flatten makes of images PyTorch orders channel by channel, so the numbers
  the next linear layer and any normalisation before it keep for them are put in their places here once the reader knows
  what the flatten is handed — an `Example` the network takes, which it runs through as zeros, or `Flattened`, stating it
  by the flatten's path; told neither, it refuses them. Numbers held in 16 bits are widened exactly, those in 64 bits
  rounded to the nearest 32-bit float as PyTorch rounds them, and whole numbers refused. The file is read by
  Onnxify.Safetensors 0.3.11 (MIT), a port of safetensors' own reader: every file that reader refuses — twenty written
  wrongly on purpose, eighteen of them refused — is refused here as a `FormatException`, and so is one whose note names
  another framework's layout than `"format": "pt"`; one with no note, as `safetensors.torch.save_model` writes, is read as
  PyTorch's. Every fault among the tensors — a kind of number that is no float, another shape than PyTorch keeps the slot
  in, a tensor for no slot, a slot no tensor is for, a value that is not finite — is named at once by the tensor's name, in
  one `SlotLoadException`, and the network keeps what it held. A network written as code keeps to itself only the order
  its forward pass runs its layers in, so wherever rows made of images could reach one of its linear layers or
  normalisations, what that layer reads is taken only as `Flattened` states it by the layer's path, and refused
  otherwise; a layer of a kind the reader does not know has none of its numbers read. The Titanic network PyTorch trained
  answers a man of 22 in third class within one rounding of PyTorch's chance and all 135 test passengers within five,
  written in Keras's words or as code; one with a batch normalisation within ten; convolutions flattened into a linear
  layer within 3e−8, as a stack and as code, where reading the flattened rows unturned lands 0.043 away. The files and
  what PyTorch answered are kept with the tests, with the scripts that made them.

- **…and the file `torch.save(model.state_dict(), file)` writes: `new TorchSaveFile(network, loss).Read(stream)`.** The
  `.pt` or `.pth` archive torch.save has written since PyTorch 1.6 goes into the same network, every number into the same
  slot, in the same layout and to the same bit as `SafetensorsFile` puts the same numbers there — the Titanic network in
  32-bit floats, 16-bit floats, bfloat16 and 64-bit floats, the network with a batch normalisation, the convolutions
  flattened into a linear layer, the parameters saved on their own, tensors saved as views of longer storages, and a file
  written as a big-endian machine writes it; and forty layers in a row, whose pickle is long enough to number its memo
  past 255, to the bit PyTorch holds them. Both readers are a `PyTorchFile` now, and share
  its `Example` and `Flattened`. The pickle torch.save writes the state into is a program, so no pickle library reads it:
  an interpreter written here carries out only the instructions PyTorch's own weights-only reader,
  `torch.load(weights_only=True)`, carries out, and builds nothing a file names but what a state dictionary is made of — an
  `OrderedDict`, a tensor rebuilt by `torch._utils._rebuild_tensor_v2` or `_rebuild_parameter`, and the storages of a kind
  of number. Every other name — `os.system`, `builtins.eval`, `subprocess.Popen`, a type of .NET's — is refused where the
  file names it, at its byte, before anything is looked up, built or run, and nothing in the package finds a type or a
  member by a name. Of 49 files written wrongly or with hostile intent on purpose, each handed to PyTorch's own reader
  first, every one it refuses is refused here too; of those it reads, six are refused here by name: a pickle of
  `torch.Size`, of bytes or of a `Counter`, none of which is a state dictionary; the format before PyTorch 1.6; a
  checkpoint that keeps the state under a key of its own; and a tensor saved as a negated view. What a file can make the
  reader do is bounded by the file and the network: a view that repeats its storage's numbers is refused, a dict is keyed
  by nothing deeper than a tuple of plain values, and a tensor's numbers are gathered only for a slot that takes them — a
  hundred views of a megabyte storage for no slot cost 4.6 MB to refuse, where gathering them took 109 MB. The records a
  file's archive holds together hold no more bytes than the file itself, as torch.save, storing each as it is, writes
  them: an archive of one megabyte inflating to a gigabyte is refused with 2 MB taken, before a byte of it is inflated, and
  a record whose bytes are other than its headers and its check say is refused. The archive is taken for one only when it
  begins as a zip archive does, as PyTorch tells one — bytes before it, or the format before 1.6 in front of it, are
  refused — and a name it holds twice, in the same case of its letters or another, is refused rather than read as either.
  A dict's keys are hashed as no file can aim at: eighty thousand keys chosen to share the runtime's own hash take a
  tenth of a second, where they took a minute and a half. Text is decoded as PyTorch decodes it, half of a surrogate pair
  standing alone included. A file longer than one array holds, 2,147,483,591 bytes, is refused by name — before a byte of
  it is read when its stream can say its length. Every name a file holds is shown in a refusal as `Quoted()` shows it.
  Razorvine.Pickle, the reader first chosen for these files, was measured and set aside: it learns PyTorch's names only
  in a dictionary every reader in the process shares, and it builds something for every name a file holds before
  anything could refuse it.

- **A file's text is shown in a message as words and nothing else: `text.Quoted()`.** Each character that would break
  the message's line, hide what follows it or turn it round — a control or format character, such as a bidirectional
  override, a line or paragraph separator, half of a surrogate pair standing alone — is written as a backslash and its
  code, the backslash itself doubled, and past 200 characters the text is cut short and says how long it was. The load of
  a network's slots shows every name a file holds this way, in a fault and in the place it names, and so do the PyTorch
  readers' refusals: a name written with a line break can no longer start a line of its own in a log, and a name of fifty
  million letters makes a refusal of a few hundred characters. The same rule reaches every reader that echoes a file's own
  words in a refusal: a Keras model's class, layer and activation names, a loss's name and how it reduces a batch, and
  what a setting is written as; an ONNX graph's, node's and value's names, an operator, a kernel's or a weight's name, and
  where a tensor's bytes are kept beside the graph; a network file's own key, parameter, feature and remembered-tensor
  names, and a kind's name where a file names one this library does not know; and a pipeline file's key, step verb and
  parameter names, and a value written for one. `DeepSharp.Pipelines` carries no dependency on the deep-learning core, so
  it keeps the same rule as its own internal `FileTextExtensions.Quoted()`, reached the same way.

- **A layer lists the layers it holds: `layer.HeldLayers()`.** Every layer a layer holds, and every one those hold, each as a
  `NamedLayer` with its dotted path from the layer asked — `1.hidden` — in the order `Slots()` walks them, so the layer
  holding any slot is named by the slot's path, in a network written as code as much as in a stack. A stack's own
  `Layers` property stays the list of its layers in the order they run.

- **A checkpoint read from one reading of its file: `NetworkDocument.ReadCheckpoint(json, network, training, catalog)`.**
  It reads the network a file holds under one key and what its run needs to go on under another, from the text read
  once, and hands both back as a `SavedCheckpoint` — its `Network`, as `ReadNetwork` reads it, and its `Run`, as
  `ReadTraining` reads it — refusing either part in the words and at the places those refuse it. `CheckpointFile.Read`
  reads through it, where it read the text twice, once through each of the other two.

- **A model trained somewhere else is trained further behind a pipeline, and kept with it as the one file.** An import
  is a `SavedNetwork` — a network and its loss, which records no fit of a pipeline — so it stands behind one as any
  network does: `saved.Network.Compile(new Adam(0.001), saved.Loss).Fit(prepared, options)` trains it on the rows the
  pipeline prepares, for one epoch or for as many as training it further takes, and the network behind its pipeline is
  written as the one file and read back beside that fit alone. The networks sample's Titanic network Keras trained,
  fitted one epoch further behind the wiki's pipeline, was read back from its file holding the same numbers to the bit
  and giving the passengers the same chances.

### Fixed

- **A notebook of blocks can be saved in VS Code.** Verso names its own format twice — its writer says `verso`, and the
  name its extensions are told when Verso's VS Code host saves a notebook as .verso is `verso-native` — and the guard
  that keeps a notebook of blocks in .verso knew only the first, so in VS Code it refused every save of such a notebook,
  in its own format too. It takes both names as Verso's own; measured in the host Verso's VS Code extension starts, the
  notebook is written as .verso, and saving it as Jupyter or Markdown is still refused.

- **A category the step that encodes categories turns into columns can be left out again.** Its box in the grid and the
  list stood ticked and could not be clicked — the sample notebook's `pclass` and `sex`, and any column ticked in and made
  a category — since the category never reaches the end of the pipeline itself, only the columns made of it do, and a
  column that does not reach the end was taken to be out already. A declared column no step names is now left out in the
  schema with its kind, wherever it goes after, and ticked in again as it was; the last category is not offered, since the
  step would then have none to encode. A profile's answer that leaves such a column out, which changed nothing, now
  leaves it out too.

- **A notebook converted to Jupyter by `verso convert` is refused when it is opened.** Verso's Jupyter writer turns a
  block into a raw cell, and Markdown into text that fences the block's text in, while the guard read code cells alone,
  so the file the converter wrote opened with its blocks as inert text. The guard now reads a raw cell, a code cell and a
  fenced Markdown cell alike, and a test holds it to refusing whatever each of Verso's formats makes of a block. It is
  asked in VS Code, in `deepsharp-serve` and in an application on `DeepSharp.Verso.Api`; the browser editor `verso serve`
  starts asks no extension when it opens or saves a file, so there the converted file opens with its blocks as text,
  and saving it writes a `.verso` file over the notebook it came from. There, **Make blocks of the steps** on the
  toolbar makes them blocks again, before the save or after it. A test reads that editor's code, so a Verso whose
  editor starts asking is noticed.

- **`deepsharp-serve` names every library it carries in its third-party notices.** Its package carries the libraries
  the server runs on — Verso and its engine, the C# compiler, NuGet's client, Markdig, and from this release the readers'
  Parquet.Net, its compression libraries and ExcelDataReader — while its notices named only the three scripts its page
  carries. Each is now named with its version, where it comes from, its copyright and its licence, and a test fails the
  day a library is carried that the notices do not name.

- **0.4.0's notebook drew no report.** Its changelog says the notebook draws a report shown drawn; it did not. An
  `evidence.report` block showed its card and its grid and nothing else — measured in Verso's own loader on .NET 8 and
  .NET 10 — because the notebook trains nothing, and no measures ever reached it: the drawing it had for them was reached
  by its tests alone. The block now draws what the C# cell that trains a model hands back, and says where its measures
  come from while nothing is.

- **The notebook is checked as Verso installs it.** 0.4.0 says an install of the notebook carries the charts and the
  engine with it, and nothing had installed one to see. The test that stood for an install laid out a folder from the
  suite's own build — without `System.Numerics.Tensors`, with `DeepSharp.Verso.Api`, which no install holds — and loaded
  it into a process that hands over any file such a folder lacks. Verso's own installer, run on 0.4.0's package since,
  found that its install did carry both, and loaded. Now, before a package leaves a run of the workflows, the notebook
  package just made is installed by Verso's own installer and loaded by its own loader in a host that holds nothing of
  DeepSharp, on .NET 8 as `verso serve` runs and on the newest runtime as Verso's VS Code extension runs. There the
  sample notebook's report block shows the data, its C# cell trains a network on DeepSharp's packages named by their
  NuGet ids and draws the loss, and the report block then draws the measures handed back. The files Verso lays out are a
  list the notebook's tests hold to what the build resolves for the package, so what it claims and what Verso installs
  cannot part without a test failing; everything the notebook refers to is found in the folder Verso installed it to —
  DeepSharp itself and System.Numerics.Tensors 10 as well, on both runtimes, where nothing had looked for either before —
  and every method of DeepSharp's compiles against what is there. `bash tools/verso/check.sh nupkgs` runs the same after
  a pack.

- **A network's file from 0.4.0 still loads once the pipeline file's version moves.** What a network records of the fit
  it was trained behind — the SHA-256 of its pipeline's own file — was checked against that pipeline written again by the
  DeepSharp reading the file, which writes its own version of the pipeline file into it: the first version after 0.4.0's
  would have made every network's file and checkpoint 0.4.0 wrote "trained behind another fit", with nothing changed but
  that number. The record is now checked against the pipeline's text as the file carries it, laid out as a pipeline's
  file is, so a file another writer indented otherwise, put on one line, gave other line endings or escaped otherwise
  loads too, and a network read from its file writes that text back as it was. A checkpoint goes on behind its fit run
  again under a later version: both pipelines pass through that one writer, the version each names left out. The wiki's
  Titanic network and a checkpoint of it, written by the published 0.4.0, are kept with the tests: they serve the
  passenger at a logit of −1.7700741 and go on to the eight epochs 0.4.0 went on to.

- **The report measures a network on the engine its run was handed.** With `new FitOptions(seed) { Backend = engine }`
  the engine trained the network and judged it by the validation rows, and the report's measures were still worked out
  on a light engine made for them: handed an engine that counts what it is asked, a run of twenty sine rows made as many
  operations on it with a report of two parts as without one. The report is part of the run, and its measures are now
  taken on the run's engine — the twelve operations of its two evaluation passes reach it — by the same evaluation pass
  `Predict` serves with.

- **`Scale.MidRange` with `OutOfRange.Refuse` no longer refuses the pipeline's own training extremes.** Its centre is
  the middle of the training range and its spread half that range's width, two numbers built by different sums from the
  same two training extremes, so the training maximum mapped back a shade past one rather than onto it — measured on a
  published price series, 1.0000000000000004 — and a refusal read that as a value outside the range the fit had just
  been given. A value within the rounding that arithmetic carries, of an end the training rows reached, now lands
  exactly on that end; min-max and max-abs were checked the same way and already landed there exactly, because each
  builds its centre from one training extreme alone rather than an average of both. A refusal also now names the row
  by where it stood in the file, as every other refusal already does, rather than by its place among rows a split may
  have put in another order.

- **An encoder refusing a category the training rows never held names the row where the file has it.** It named the
  row by its place among the rows it was handed, which after a split is another number than the line a person looks
  up; it now names the row as it was read, as the handover and every other refusal do.

- **A checkpoint records the batch size and the early stopping its run went under, and a run going on under others is
  refused.** It recorded the seed alone, and a run went on from it bit for bit only when handed the same batch size and
  early stopping, which nothing held it to: the batch size decides which rows each step takes, and a dropout's draws are
  counted by the step, while the early stopping decides where the run ends and which weights it ends holding. On the
  networks sample's Titanic run — seed 20260929, early stopping of patience 10 that restores the best, 46 epochs keeping
  the 36th — its checkpoint after the 20th epoch went on in batches of 16 to 35 epochs keeping the 25th, with a
  patience of 3 to 31 epochs, with a patience of 30 through all 100, and without early stopping to the network of the
  100th epoch rather than the best, each accepted without a word; only another seed was refused. The file's training
  part now holds `batchSize` and `earlyStopping` — its `patience`, `minDelta` and `restoreBest`, or null for a run that
  had none — and a refusal names every difference at once, the seed's among them.

- **A checkpoint records the engine its run was on, and a run going on on another is refused.** A checkpoint named no
  engine, and a run went on from it on any engine it was handed without a word, as another run: the networks sample's
  Titanic run on the light engine, gone on from its file on libtorch or on the tests' native engine after its 1st, 5th,
  10th or 20th epoch alike, kept the same epoch and ended with weights up to 3.0e−7 from those of the run the checkpoint
  was taken of, which the light engine goes on to the bit — under the same seed, with nothing to tell the two runs
  apart. The file's training part now holds `engine` — its `name`, and its `version` and `device` where the engine
  names them — and a refusal names both engines, with every other difference at once.

- **A served row refused for a column it lacks is told what its rows offer.** The refusal listed the answers the
  pipeline awaits among the columns the rows offered — `survived` for the Titanic passengers, the twenty-four hours for
  a day of bikes — though nobody handed them in; it lists the columns handed in.

- **What `ServedBatch.Keys` says of a served row's key.** It said that the answers a row awaits count as gaps in its
  key. A row handed in with its answer is keyed by the cell it carries, so the same passenger handed in with and without
  `survived` has two keys, and is served alike under both.

- **A network's file and a checkpoint end every line with a line feed, on every system.** The network's part was
  written with the line endings of the machine it ran on — a carriage return before every line feed on Windows — beside
  the pipeline's part, written with line feeds everywhere. Every such file is read as it stands, and one written again
  comes out with line feeds throughout.

- **0.4.0's changelog says that what a network is handed lies between minus one and one; it is declared to land
  there.** A scale fitted on the training rows lands them between minus one and one, and the handover refuses a network
  any feature not declared to land there. A row that arrives later and lies beyond what the training rows held passes
  outside, unless its scale is declared to clip it or refuse it: on the Apple prices, four closes and one volume of the
  152 days after the training rows do.

- **The README says where its example's file comes from.** It read `btceur-1d.csv`, which ships nowhere, without a word
  about it; it now says that it is a file of your own and names the columns it holds, and a test holds those to the
  columns the example declares.

- **The wiki's code compiles as it stands, and a test holds it to that.** Every C# block on its pages is compiled
  against the packages as built, and fails on a warning, an obsolete form among them; a notebook's cell is run in
  Verso's own engine. The Networks page's network written as code trained and could not be saved: it now says the name
  its file knows it by and how it is built again, and the page names the catalog it is read back with. The pages use the
  forms that take four parameters at most.

- **A network in Keras's words that ends in `.Sigmoid()` before `BinaryCrossEntropy` answers through one sigmoid.** The
  loss takes the logits and applies the sigmoid itself, and 0.4.0's words kept the sigmoid layer too, so every prediction
  went through it twice and the training read the sigmoid's outputs as logits: logits of −20, −1, 1 and 20 came out
  0.5000, 0.5668, 0.6750 and 0.7311, every one a half or more, so every measure that counts classes counted every row as a
  one. The words now leave that sigmoid to the loss, and the same logits come out 0.0000, 0.2689, 0.7311 and 1.0000.

- **Reading a network's file takes a twelfth of the memory it took, a checkpoint's a third, in under half the time.** To
  place a fault at its line and column, the reader kept where every number of every slot stood in the file, and counted
  every line: a network of four million parameters — one dense layer of 2000 by 2000, a file of 95.6 MB — took 430 bytes
  a parameter to read. Where a number stands is now worked out only when a fault is placed there, and the lines are
  counted only then; and a tensor's numbers are read from the text where they stand, rather than parsed with the rest of
  it into a document that keeps a row for each, taken from .NET's pool of arrays at the size of the whole text. The same
  read takes 36 bytes a parameter: the file's own text once more, as UTF-8, and the numbers. The one file of a network of
  four million parameters behind the Titanic pipeline took 501 bytes a parameter: its top and the pipeline it carries
  were each read from a copy of the whole file as well, each counting its every line. Both are now read a piece of the
  file at a time, keeping only the pipeline's own bytes, and the one file takes 36 bytes a parameter. A checkpoint of the
  same network under Adam, whose file holds three numbers a parameter, took 372 bytes a parameter to read, and 432 with
  the best epoch's slots besides, since its network and its run were each read from the whole text; read once, it takes
  96 and 128. Every fault is placed where it was, and every file reads as it did.

- **The largest network the one file holds is the same on every system: about 45 million parameters.** The file is one
  text, .NET makes no text longer than 1,073,741,791 characters, and a network's numbers take between 23.6 and 24
  characters a parameter; a network of 45 million parameters is refused with the `OutOfMemoryException` .NET refuses so
  long a text with. On Windows the writer ended every line with a carriage return and the text was made before they were
  taken out, a character longer for every line, so a network of 43.7 million parameters could not be written there,
  though its file fits; the carriage returns are now left out before the text is made, and that network is written and
  read back on Windows too. A checkpoint holds, beside every parameter, what Adam remembers of it, and the best epoch's
  number when its run keeps it — 76 and 104 characters a parameter — so it holds a network of about 14 million
  parameters, or about 10 million: one of ten million keeping the best epoch's was written and read back, and one of 10.6
  million refused.

- **The build proves what a person runs, not only what every suite built.** `dotnet pack` now checks each of the sixteen
  packages' public surface against the one nuget.org already carries, for the nine 0.4.0 already shipped there, so a
  change nobody meant to make fails the build instead of somebody's upgrade — measured with a property taken from public
  to internal: the pack refuses it, naming the member and both builds. The seven packages new this release take the same
  check with nothing to compare against yet. An application referencing `DeepSharp.Backends.TorchSharp` from the package just made,
  and none of this repository's source, is started from it twice, the way `deepsharp-serve` and the notebook already are:
  brought no libtorch, it is refused, naming every package that brings one; brought the processor's, it fits a step of
  the networks sample's Titanic pipeline on it. And the engine's own suite, proven so far only on the platform it was
  built on, now runs on Windows in CI as well as on Linux, on both frameworks it ships for, so the native interop under
  it is not left untried on one of the two platforms a person runs it on.

## [0.4.0]

Where it learns. Layers, losses, optimizers and learning-rate schedules; a training loop that stops once the validation
rows no longer improve, with checkpoints a run goes on from bit for bit; a network described in Keras's words or written
as code, trained on the rows a pipeline prepares and kept with that pipeline as one file; the pipeline's report measuring
it in the answer's own units, and the charts drawing it. Beneath them, a gradient worked out automatically for the
arithmetic a network does, and data that says what it holds as it is read: each column is proposed a kind — a date, a
category, a number — from what every cell says, a person accepts or changes it, and what should not be there is named
with the step that deals with it. What a network is handed lies between minus one and one.

### Upgrading from 0.3

- **A backend implements the new operations.** `ITensorBackend` gains `Subtract`, `MatMul`, `Transpose`,
  `AddRow`, `SumRows`, `Mean`, `Scale` and `Fill`, and the operations the layers, the losses and the convolution are
  written in — `Relu`, `Positive`, `Tanh`, `Sigmoid`, `Exp`, `Log`, `Sqrt`, `Softplus`, `Divide`, `LogSoftmax`,
  `Reshape`, `Unfold` and `Fold` — so a backend written against 0.3 has to implement them before it compiles again.

- **A visitor of evidence visits measures too.** `IEvidenceVisitor<TResult>` gains `Visit(Measures)`, so a visitor
  written outside the library implements it before it compiles again.

- **A batch says which part it came from and which row each is.** `Batch` gains `Part` and `Keys`, set by
  `Batch(part)` and nothing otherwise, so a batch made by hand is no longer equal to one the pipeline handed over.
  `INamesTheAnswer` gains `AnswersCanBeClasses` and `Ones`, each with a default, so an output from another package
  compiles as it is.

- **The notebook package brings the charts.** `DeepSharp.Verso.Notebooks` draws through `DeepSharp.Charts`, which
  brings `DeepSharp` with it, so an install of the notebook carries both; the correlation it draws is the picture it
  drew.

- **Cells that cannot be read are refused all at once, at the schema.** Run through a pipeline, they come as a
  `DeclarationException` with a fault at the schema's step, where a `FormatException` named only the first cell;
  `SchemaBinding.Bind` on its own still throws a `FormatException`, now naming every column at once.

- **A timestamp column reads its moments as ISO 8601 writes them, or as its format says.** A column that names no
  format reads `2015-02-18`, `2015-02-18 09:30` and `2015-02-18T09:30:15`, with a fraction of a second, a zone or an
  offset or without; any other writing is refused at the schema, with its row, its column and the cell. It was read
  by a reading that filled in whatever the text left out, so `7.25` became the twenty-fifth of July of whichever year
  it ran in, `12:30` that time on the day it ran, and `02/03/2015` the third of February on every machine. A column
  written another way says how: `schema.Column("when", ColumnKind.Timestamp, format: "dd/MM/yyyy")`, or
  `"format": "dd/MM/yyyy"` on the column in the file.

- **Files are written against version 3.** A file this version writes names version 3, which 0.3 refuses as newer
  than itself; every file 0.3 wrote is read as it was.

- **An alert names its columns and how it is answered.** `ProfileAlert` holds `Columns` — the one it was found in,
  and for a column that repeats another or restates the answer, that other — and an `Answer`: a step for the column,
  the column left out, or a value the schema says stands for a gap, in place of a verb. `Column` is still the one it
  was found in. A column that never changes is answered by leaving it out, where `drop.columns` was named.

- **A scale does what is said of a value outside its range, or is refused where it is written.** Standard,
  robust and power land their rows in no range, so they only pass: `clip` and `refuse` with them are refused, where
  they did nothing. A quantile scale ranks a value among the training rows, so it holds one beyond them at the edge
  or refuses it: `pass` with it is refused, where it held the value at the edge whatever was said — write `clip` for
  what it did. A new `normalise` block starts with `midrange`.

- **A column ticked in takes the kind its cells propose.** From the notebook's list or its grid, a column the schema
  does not name is taken in as the file saved beside the notebook declares it, else as its cells propose, else as
  text — one rule for both boxes — where the grid took every one in as text.

- **A step that writes a column says where its values land.** `ColumnState.With(name, kind)` writes the column
  landing nowhere said, so a step from another package that lands a column in a range says so with
  `With(name, kind, lands)`, and one that makes a family with `WithFamily(start, lands)`; until it does, a learner
  that takes every feature on one scale refuses its columns.

- **A file read through the data frame keeps its own text.** `ReadCsvFrame` reads every column as text, as the file
  writes it, where the frame guessed each column's kind from its first ten rows and handed back its own spelling of
  what it read: `133.1285` for `133.1284878`, in 3,661 of the price series' 4,554 numbers. Its rows are now the rows
  `ReadCsv` makes of the same file, known by the same keys, so a pipeline fitted through it before divides its rows
  otherwise when it is fitted again.

### Added

- **Layers.** `Dense`, `Relu`, `Tanh`, `Sigmoid`, `Dropout`, `BatchNorm`, `LayerNorm`, `Conv2D` over a `Window`,
  `Flatten` and `Reshape`, each starting as PyTorch's starts — weights uniform by their fan-in, biases within one over
  its square root — and each matched against PyTorch on the walked rows, forward and back. A layer is code: its numbers
  are slots with dotted paths, `0.weight` and `2.running_mean`, a `Parameter` that an optimizer moves or a
  `RunningStatistic` that only a training pass moves, and a `Pass` says whether it trains or evaluates, so no layer
  carries a mode. `LayerStack` names its layers by their place, and a network written as code names its own.

- **Losses that say what a network's numbers mean.** `MeanSquaredError`, `CrossEntropy` over shares through a softmax,
  and `BinaryCrossEntropy` over chances through a sigmoid, each taking the raw numbers — binary cross-entropy is exact at
  a logit of nought, where one built from rectified values sent back nought or minus one — and each refusing, by name,
  a row whose answers it could not have meant.

- **Optimizers and learning-rate schedules.** `Sgd`, with momentum, and `Adam`, as PyTorch writes them, matched step by
  step; `ConstantRate`, `StepDecay`, `ExponentialDecay` and `CosineDecay`, worked out from the epoch and asked once an
  epoch.

- **A training loop.** `network.Compile(optimizer, loss, schedule)` and `Fit(train, validation, new FitOptions(seed))`:
  Keras's defaults — one epoch, batches of thirty-two — the rows shuffled afresh every epoch, the validation rows judged
  once an epoch and never trained on, `EarlyStopping` as Keras's to the letter, restoring every slot of the best epoch
  when asked, and `Checkpoints` every epoch or only the best, from which a run goes on bit for bit. A loss that is not a
  finite number is refused with the batch and the epoch it came from. `History` holds every epoch's losses and rate,
  the seed, the best epoch and why the run stopped.

- **Every draw counted from one seed.** `RandomStream` keeps nothing but its seed and counts every draw from it — what
  a layer starts at, an epoch's order, what a dropout leaves out — so a run resumed at an epoch draws what it would
  have, and a layer added moves no other's start.

- **A network in Keras's words.** `new Sequential().Dense(16).Relu().Dense(1)` is read once and lowered onto the stack
  a network written as code would be, every width worked out from the rows — `Lower(shape, stream)` by hand, or at the
  first fit of `Compile`, from the rows' shape and the run's seed — so the two give the same network bit for bit and the
  same file. `Input(shape)` states an example's shape, so every word is checked where it is compiled.

- **A network written down.** `NetworkDocument` writes a network as the kinds it is made of, the numbers it learned
  and its loss, and reads it back through a `NetworkCatalog` to the last bit; a checkpoint adds what the run needs to
  go on. A kind nobody registered, a setting or a slot that is missing, extra or wrong, and a value that is not a finite
  number are refused at their line and column, all at once. A network written as code is registered with
  `Register<T>()` and read back by its name.

- **A network trained behind a pipeline.** `DeepSharp.Learners.Networks`: `compiled.Fit(prepared, options)` trains on
  the training rows the pipeline hands over, judged by its validation rows, never showing it the test rows, and gives
  a `TrainedNetwork` — its history, its measures, and `Predict(rows)`, whose `Predictions` come back in the answer's own
  units with where each row was handed in. `ToJson()` writes the network and its pipeline as one file, which
  `TrainedNetwork.FromJson` reads back only beside the very fit the network was trained behind; `CheckpointFile` writes
  and reads a checkpoint as the same file.

- **A report of what a trained model is measured by.** `evidence.report`, or `.Report(report => report.Measure(...)
  .On(...).As(...))` on the chain: RMSE, MAE, R², accuracy, precision, recall and the confusion matrix, on the
  training, validation and test parts, shown as numbers, drawn or both. `PreparedData.Measure` measures a model's
  predictions in the answer's own units, each beside predicting the training rows' average, by scikit-learn's
  definitions, and only for predictions of the part's own rows in the part's order. New in the file's third version.

- **Charts.** `DeepSharp.Charts` draws, as the text of an SVG, a run's loss curve and learning rate, a report's
  measures as bars beside the average, each confusion matrix as a heatmap of counts, what was predicted against what
  was there, what was left over, and a correlation as a heatmap. The notebook draws a report shown drawn with it.

- **A pipeline read from inside a larger file.** `PreparedData.FromJson(json, catalog, property)` reads the pipeline
  under one of the file's keys, every fault at its line in that file.

- **The word a file writes a choice as.** `Word()` names a part, a measure or any other choice as a pipeline's file
  writes it, by the one rule its reader and writer use.

- **Two years of bike sharing, and a networks sample.** `Samples/data/bikes.csv`: a row a day, its weather and season,
  and how its rentals spread over its twenty-four hours (Fanaee-T and Gama, CC BY 4.0). `Samples/DeepSharp.Sample.Networks`
  trains a network on each of the three datasets, through both doors, measures, saves, reads back and serves each.
  `Samples/DeepSharp.Sample.Pipelines` now asks the passenger list what each of its columns holds before anything is
  declared and has a profile say what should not be there and how each is answered, prints its numbers the same on every
  machine, and runs on .NET 8 as well; a test runs each sample as it stands.

- **The operations a backward pass needs.** A matrix product, its transpose, a row added to every row of a matrix
  — the way a bias reaches every example of a batch — and the rows summed, a subtraction, a mean that is one value
  with no axes, as a loss is, a scale by one value, and a tensor filled with one value. `Add` still takes two tensors
  of one shape. .NET's vector primitives hold no matrix product, so the one that ships is written here, and every
  total is kept in double precision: ten million tenths average to a tenth, where a single-precision total gave
  0.1087937. And the operations the layers, the losses and a convolution are written in, each with its rule for the
  way back: rectified and stepped values, tanh, the sigmoid, exp, log, a square root, softplus, a division, a row's
  log-shares, a reshape, and an image's windows unfolded into rows and folded back, channels last. Softplus is worked
  out in double and exact at both ends — .NET's `LogP1` gave nought for the softplus of −40, which is 4.2e−18 — and a
  row's log-shares are shifted by its largest value, so logits of a thousand are as good as logits of one. A `Window`
  says how a convolution walks: its height and width, its stride and its border.

- **Gradients, worked out automatically.** `RecordingBackend` wraps any backend for one pass and writes down each
  operation as it runs it; `GradientsOf(loss, parameters)` then works back from a loss of one value to how much each
  tensor moved it, each gradient of its tensor's own shape. A tensor read twice gets the sum of both readings, and one
  the pass never read is refused rather than given a gradient of nothing. The way back runs on the wrapped backend's
  own operations, and every operation's rule is checked against the loss nudged a little either way.

- **A timestamp column says how its moments are written.** `format` on a declared timestamp column, as .NET writes a
  date format, from the chain, in the file and in the notebook's form, where a taken timestamp column has a field for
  it; left empty, the moments are read as ISO 8601 writes them. A column with a format also reads the round-trip form a
  database or a typed data frame hands its moments over in. A take-over lists a change of format like any other change
  to a column, and a file holds a format only where one is said. `WithFormat(column, format)` on a declaration hands
  back its steps with the format said, as the other column operations do, and `WithColumnFormat` on the schema.

- **The notebook's list shows what each column's cells propose.** Beside a column the schema and the saved file leave
  to its cells, the list says what they propose — `category, proposed, 3 values`, with a category offered beside whole
  numbers and the format moments are written in — and its box takes the column in so. A profile's alert whose answer
  is a change to the columns carries a box that gives it: leaving the column out, or saying on the schema that a value
  stands for a gap.

- **A learner that takes every feature on one scale says so where they are handed over.** `Batch(part,
  Needs.OneScale)` and `Served(rows, Needs.OneScale)` refuse every feature not declared to land between minus one and
  one, all at once. Where each column lands is followed down the steps with the columns — a scale's range, a moment's
  form on its circle, an encoder's noughts and ones, true or false, a row divided by its size — as
  `KnownColumn.Lands` and `ColumnState.LandsOf`, so the check reads the declaration and holds a pipeline loaded from
  its file as it held the one that was fitted.

- **A quantile bound at nothing is the training extremes.** `outliers.clip` with `bounds: quantile` takes `at: 0`, the
  smallest and largest value the training rows hold, so `refuse` stops the run at a later row beyond anything the fit
  saw; by spread or by the middle half, nothing is still refused. `NumberParameter` takes a least value, `atLeast`.

- **Features between minus one and one.** `Scale.MidRange` centres a column on the middle of its training range
  and divides it by half its width, so the training rows land between minus one and one, as a network takes them —
  scikit-learn's `MinMaxScaler` with a feature range of minus one to one — and comes back the way every linear scale
  does. On the price series the second day's close of 128.720001 becomes 0.7993437, and four closes of the 152
  days after training land outside, which `clip` holds at the edge and `refuse` refuses. `Lands()` says where each
  scale lands the training rows, and a quantile scale can now refuse a value beyond them.

- **What should not be there, found by the profile.** A column that goes with the answer value for value hands a
  model the answer, and one that goes with a column before it says again what that column says — each named with its
  values and how many training rows hold them, and answered by leaving it out; a column of words no two rows share, or
  a running number, only names its row; and a number far from every other, held by more than one row and written as
  files write that nothing is known — 0, −1, a run of nines — is answered by the schema saying it stands for a gap.
  `AlertAnswer.AppliedTo` gives an answer that changes the columns, and a view carries the pipeline's `Answers` and the
  columns its rows are `OrderedBy`. An extreme raises no alert: clipping or refusing one is a step you declare.

- **A value that stands for a gap, said on the column.** `missing` on a declared column — `0`, where a file writes 0
  for a fare nobody knows — makes every cell holding it the gap it is, before the kind reads it, so a fill counts it
  and marks it. It is compared as the column's kind reads it when it reads as that kind, so `0` and `0.0` are one
  value, and as it is written otherwise, so a column of numbers can say that `?` is a gap. From the chain
  (`schema.Column("fare", ColumnKind.Number, missing: "0")`), in the file, and in the notebook's form, where every
  taken column has a field for it; a take-over lists a change of it, and `WithMissing(column, value)` on a declaration
  and `WithColumnMissing` on the schema say it. The row is still known by what the file wrote.

- **Each column's kind, proposed from every cell.** `KindProposal.Of(source)`, or `ProposedKinds()` on the chain
  before the schema, reads every cell of every row with the schema's own reading and proposes a kind for each column —
  true or false in any spelling, whole numbers, numbers, moments as ISO 8601 writes them or by the one usual format
  that reads them all, a category for few words, and text — with how many cells hold a value, how many are gaps and how
  many different values there are. Whole numbers few enough to be groups are offered as a category beside what is
  proposed, and moments whose order of day and month the cells cannot settle are offered with every format that reads
  them, never proposed as one. It is a value to decide from: nothing takes it in, and the schema is still written by a
  person.

- **A source can say what its columns hold.** `IStatesKinds`, beside `IRowSource`: a query read by `ReadDbAsync`
  states each column's kind as the database types it, and the proposal takes that over what the cells look like, so a
  code the database holds as words stays words however it is written.

### Fixed

- **A column the training rows hold one value of is centred, not blown up.** The standard and power scales took the
  rounding of their own arithmetic for a spread: two hundred of 0.1 came out with a spread of 6.9e-17, so every training
  row became −1 and a later 0.2 became 1441151880758557.8. A spread no larger than that rounding is now nothing, for every
  scale, as scikit-learn decides a feature is constant. A pipeline already fitted on such a column keeps the spread it
  learned until it is fitted again.

- **The power scale keeps the shaping its training values are most normal under.** Its search added a twentieth at a
  time and reached nought and two as 1.2e-15 and 2.000000000000002, so what it measured there was not the shaping it
  kept: on seventy small values either side of nought it kept 2, where scipy's `yeojohnson_normmax` puts the best at
  0.71, and now keeps 0.7. Fitted again, a pipeline may keep another shaping than it did.

- **Opening a notebook in `deepsharp-serve` runs nothing its file carries.** An output a cell showed when it last ran is
  kept in the notebook's file, and the page placed its HTML as it was written, as Verso's editor does: a handler on an
  element in it ran as the page itself, which holds the notebook's socket and can run its cells. Every piece of HTML the
  page places — what a cell shows, what a layout draws, a button's icon — now passes DOMPurify first, carried in the page
  like Mermaid and KaTeX, and what would run is taken out. A widget, which runs code on purpose in a frame of its own,
  takes its look only from the page that made it.

- **Two different rows are never counted as one repeated row.** The profile joined each row's cells into one string,
  so two rows whose cells held the characters it joined with could come out alike; a row is now known there by the same
  digest a row key is made by, each cell with its length.

- **A kind is read from its word alone.** A control that carried `1`, or two kinds joined by a comma, was read as a
  kind in the notebook, where a file refuses both.

- **A gap is answered by a step the rules keep for its column.** The profile named `fill.missing` for a gap in words,
  in true and false, or in moments, which no fill takes: a gap among words is answered by the encoder, which makes it
  no category and marks it, and among true and false or moments by `drop.gaps`.

- **The pipeline file's JSON Schema refuses what the reader refuses of a column.** An editor checking a file against
  it passed a `was` on a column that is not a category, and a category that says it was a category; the reader
  refuses both, and so does the schema now, as it refuses a `format` on a column that holds no moments.

## [0.3.0]

A pipeline you can look at: every step written as a block of a notebook, and the data at any block shown on
request. Underneath it, the pipeline became strict where it was only polite. Every way of writing one keeps
the same rules, a row is known by what it says rather than where it stands, and a file says everything that is
wrong with it at once, each at its line and column. A model can be asked for more than one answer — the shares
of a whole, labels, a value some rows on — and its predictions come back in their own units; and the columns a
pipeline decides about can be changed from the notebook, saved beside it and taken over again. An application of
your own can host the notebook, too, and DeepSharp's own server shows it in a browser with nothing else installed.

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
  ticking again write the steps that do it and take them back, a column's kind kept. A column that is not in is
  black, its values left off the page, and a category is one colour, darker for a value the training rows never
  held. "Choose the columns", on the
  schema's block, lists every column of the source with its first values and a box that takes it in or leaves it
  out, marking new the columns the saved file never showed; each row's kind select commits the kind a keyboard walk
  ends on, as one pick from the state the list was drawn in, so a category walked away from and back to remembers
  the kind it was. Above the rows a select picks the kind of output, and each row's output box puts its column into
  that output or takes it out, with a box of its own that takes the output away; the output's own values — rows
  ahead, a return, how many ones, what the shares are shares of — are selects offering only the values the rules keep.
  The list ticks one column or a range: two ticks take in seventy columns, or make them the answer, in one change
  with one kind picked for the range. What the blocks decide about
  their columns is saved beside the notebook, as `<notebook>.columns.json`, after every change they accept and
  every run of the whole pipeline. An output block is a stage of
  its own, and its form swaps it for any other kind of output. A profile block names, for
  every problem it finds, the step that answers it, and a correlation block is drawn as a heatmap over the
  complete training rows. The toolbar runs the whole pipeline, fitting every step on the training rows, and
  exports it as the same pipeline file the chain writes. Its "Take over the saved columns" lists, at the schema's
  block, every column whose decision taking the saved ones over would change — whether the source may lack it too, and
  what becomes of the columns the schema does not name — with every saved drop the blocks cannot make and why, and
  changes nothing until the list's own box is ticked. Every change to a notebook takes its turn — a gesture on the
  grid or a list, a change in the form, the toolbar's run, export and take-over — so a host that hands them over side
  by side still changes the blocks one at a time, and a change still on its way to a block the change before it
  rewrote is not made. A block added, taken away or moved, a cell turned into another kind, or text typed into a
  block — none of which Verso's editors tell a part of — is caught up with at the next gesture: a grid the blocks no
  longer make is cleared — before it is forgotten, so one a stop leaves on the screen is cleared by the next catch-up —
  and the pipeline handed to C# cells is taken back unless they still make it, while a cell turned into another kind
  keeps what it shows. An application that changes cells itself tells the notebook at once,
  with `StepCellType.BlocksChangedAsync`, and tells it of a stop with `StepCellType.StoppedAsync`, which gives the
  notebook back at once and ends once every write let through before the stop has landed; from then on what the
  stopped run asks for writes nothing, and a change whose run was stopped before it began does nothing at all. What a
  block's run writes — its card, a grid and what it measured, the list of the columns, the columns saved beside the
  notebook, what is handed to C# cells — is written whole or not at all, so a stop, Verso's own editors' Stop among
  them, never splits one. C# cells in the same notebook are handed the pipeline as text, under `deepsharp.pipeline`,
  with the notebook's folder under `deepsharp.folder`. Saving the
  notebook keeps the steps and leaves out what the blocks show, the data included. It is saved as `.verso`, the one
  format that keeps a block a block: saving it as Jupyter or Markdown, which keep a cell's text and lose its kind, is
  refused, and so is opening a file of another format whose code cells are steps. It is
  installed from Verso's Extensions panel and runs in Verso's VS Code extension, in the browser editor
  `verso serve` opens, and inside an application that takes Verso's engine as a dependency, DeepSharp's own server
  among them.

- **`DeepSharp.Verso.Api` — the notebook in an application of your own.** It opens a notebook the way Verso's own
  editors open one — through the serializer for its format, past the guards that run after reading, with the cells
  that are only ever shown rendered drawn — and registers the notebook's parts itself, so a program published as a
  single file, with nothing beside it for Verso to find, still has them. One file is one notebook, however many views
  show it: the same path twice, or two spellings of one path, hand back the same host, and a notebook that cannot be
  opened leaves nothing open behind it. An extension a notebook asks for is refused rather than fetched. Nothing the
  engine falls back on — a kernel, a layout, a theme — is written into a notebook, as Verso's browser editor writes
  none: a file that names none is saved naming none, and a layout the engine lacks is shown in the notebook's own and
  kept by its name. What the layouts arranged, such as the dashboard's tiles, and what the parts keep as settings are
  handed back at open and taken back at every save. Its cells, and what they show, are handed over as plain values.
  Typing a cell's text, running a cell and a click on a control a block drew take their turn one at a time, in the
  order they came; typing or running a cell a change before it rewrote is refused, while a click from its card still
  reaches the part, which knows the block it became, and what a part answers a click with is what the cell shows next.
  A cell is added after another or at the end, of any kind the engine has — code in a language it runs, Markdown, a
  pipeline block and the rest, each saying whether a person writes its text and whether it is shown rendered once it
  has run — and starts empty; a kind named with no language takes the one Verso's editors give it. It is taken away,
  moved past the neighbour it passes, or turned into another kind in one step, keeping its text. Each goes through the
  port the notebook's layout guards, so a layout that does not allow it refuses it, typing included, and every version
  says which layout the notebook is shown in, what it allows and whether it has a properties panel: the dashboard and
  the presentation have none, and there a cell's panel and its fields are refused. As a cell's text is typed, its
  kernel offers what may come next and says what a word means, as Verso's editors ask it: a block's verbs, keys and
  columns, C#'s members. A new notebook is made as one block that reads a CSV file, naming C# as its kernel and when it
  was made, as a `.verso` file written whole before it appears, and never over a file that is there already. Typing
  into a block tells the notebook at once, so what was worked out from the block as it was is taken back. A run exists
  from the moment it is asked, numbered, and a stop names the run it means. A run that never ends is stopped by a
  fresh kernel — of what ran at the stop, found as the engine finds it: the kernel of the cell's type, else of its
  language, else the notebook's default, and none for a cell that only draws, which takes no C# turn either; for code a
  button runs in no cell, its language, else the default. A stop while nothing runs, between two cells, starts none. A
  stop is one step: unless the run already ended by itself, it takes the run's end, marks the run, tells the notebook —
  what the run left behind writes nothing more, no grid and no pipeline handed to C# cells, and the notebook takes its
  next change at once — and only then takes what runs as the kernel to start afresh, which happens once everything let
  through before the stop has landed; so whether a kernel starts afresh never depends on when a thread happens to run.
  C# runs take their turn across the whole process, so none prints
  into another; a C# run waiting for its turn is told on every version, and a stop ends the wait, so it never runs.
  Every version carries the toolbar: every button the engine has, each asked at that version whether it can be
  pressed — a button on a cell's toolbar for each cell, as Verso's editors ask it, so a page draws a cell's buttons
  where they can be pressed — and a button whose part cannot say is not pressable, and says why. A press is one run,
  and stopped, the cell under way is left behind and no cell the button would still run begins; a file a button hands
  over goes to whoever pressed it, unless the press was stopped, and nothing is written beside the notebook. A cell's
  properties panel is a section
  from every part that has one for it, and a changed field is made by that part. The panel, what a kernel offers and
  what a word means are read beside the notebook's turn, so they answer while a run is under way and through a stop's
  fresh kernel; one that a start afresh overlapped says nothing, even when the kernel failed as it was put away, since the
  kernel it asked is gone. A look — a button asked whether it can be pressed, a panel asked to draw its section — is refused every
  verb, so what is only looked at does nothing to the notebook. A change — a click on a control a cell drew, a changed
  field — runs DeepSharp's blocks as its own; anything else it asks to run is a run of its own, told and stopped as
  any run is, and once that run is stopped nothing else the change asks is done, and what it would answer or hand over
  is dropped. The layouts and the themes the engine
  has are listed, and each is switched by its id in the notebook's turn and saved as the notebook's choice
  (`SwitchLayoutAsync`, `SwitchThemeAsync`). What a person does to what a layout drew — a tile moved, resized or run —
  goes to the layout's own part as a change (`InteractAsync`): a move runs nothing and waits for no C# run, a tile's
  run is a run told and stopped as any, and a file the part hands over goes to whoever acted. The title is changed as
  Verso's editors change it (`RetitleAsync`): a change every view is told, unsaved until saved, and the name of what
  the notebook is exported as. A notebook is saved the way Verso's editors save it — what a block shows left out, when
  it was last saved stamped in, written whole before it takes the file's place — and saved under another name it is
  that file from then on, what DeepSharp names after it following; a name another open notebook holds is refused.
  Every view of a notebook is told what changed: the notebook as it stands when the view begins, then each change after
  it, numbered — the cells that came or changed as they now stand, and every id in order when cells came, went or
  moved, whatever made the change, a clear and a form included, of which the engine says nothing. A turn tells every
  view what it does while it does it, and one version is made at a time, so no view loses a change. A cell carries
  what Verso's editors draw it with: how it is shown, how many times it ran, how its last run ended and how long it
  took; what it shows carries a failure's name and where it happened, and whether a text came on standard output or
  standard error. What a C# cell displays while it runs reaches a view before the run ends, and every version says
  which run is under way, from its start — the cell that runs or waits, none between a button's cells, since when, and
  the number a stop names. Every version says, too, what the engine runs that no run owns — a block a change runs, and
  what a stop left behind — each until the engine says it ended; whether the notebook differs from its file, worked
  out at open too, so a save is told as a version; what became of its kernels — how many times one was started afresh,
  by a stop or by Verso's Restart Kernel, whether one is being started now, and why the last start failed; the theme
  the notebook chose; what the layout draws of its own — the dashboard's tiles, the presentation's column — as HTML
  with a slot for each cell it shows, or why it could not; and what the notebook says of itself: its title, its
  default kernel, when it was made and last saved, and the version of its format. A view that reads slowly holds one
  change, the latest look at each cell, and never holds the notebook up; it takes the change waiting, or waits for the
  next, whenever it is ready for one. A request about a cell that is gone says from which version on. A file that
  repeats a cell's id gives each repeat an id of its own, as Jupyter's own reader does. Given a grace, a notebook
  closes by itself once no view has shown it for that long, nothing runs or waits in it — nothing a stop left behind
  either — and nothing in it differs from the file it was last saved to — what a block shows never counts, as it is
  never saved; one with changes not yet saved stays open until it is saved. An application can close one notebook at
  once: its run under way is stopped, never waited for, a change under way finishes, and whatever still waits its turn
  is refused; a closed notebook tells its views nothing more, and closing them all stops every notebook's run before
  any of them closes.

- **`DeepSharp.Verso.Serve` — DeepSharp's own server.** The tool `deepsharp-serve`, installed with
  `dotnet tool install --global DeepSharp.Verso.Serve` and run beside a notebook or a folder of them, built on
  `DeepSharp.Verso.Api` and needing nothing else installed. The notebook it serves runs code, so it listens on this
  computer alone, answers only a request carrying the token it said when it started — in the address, or in the cookie
  its first page sets — opens a notebook's connection and makes a change only for its own page, since a browser carries
  that cookie for a page from any port of this computer, answers only under a name of this computer, and serves its own
  page and nothing from the folder it runs in. It says where it is once, then nothing; `--port` names the port,
  `--no-browser` opens none, a port already taken stops it before it says anything, and Ctrl+C ends it within moments
  whatever a notebook runs: every run is stopped as a close stops it, and a thread a cell left going does not keep the
  tool alive. It carries a build for .NET 8 and one for .NET 10, and keeps starting on the newer runtime once .NET 8
  is gone.

  A page holds one connection to a notebook, a WebSocket, as Verso's editor holds one for each tab: the notebook as it
  stands first, then each change, numbered. On it the page asks for what a person does — typing, sent as it is typed;
  a run, and a stop naming the run it means, answered at once, past everything asked before it; a click on a block's
  controls; a cell added, taken away, moved or turned into another kind; what a cell's kernel offers and what a word
  means; a toolbar button; a cell's properties, and a field changed, its value handed on in the form Verso's browser
  editor hands it; the layout, the theme and the title; an act on what a layout drew; a save and a close — and each is
  answered to that page alone, never before the change it made. A dropped connection opens again by itself, and the
  notebook is drawn onto what the page shows, so the cell being written keeps its text and its cursor; text typed while
  it is down is kept, marked as not sent yet, and sent once it is back. A notebook the server does not serve is refused
  in words, and a cell another change took away is said to be gone in plain words. A file a button hands over arrives
  as a download in the page that pressed it, under its own name, and nothing is written beside the notebook; in a
  folder, the page makes a new notebook under the bare name of a `.verso` file, never over a file that is there.

  The page draws the notebook as Verso's editor draws it, and carries everything it draws with, so it fetches nothing
  from anywhere: a block's output; a failure with its name and where it happened, and standard error labelled; JSON as
  a tree, CSV as a table, progress as a bar; Mermaid diagrams, and formulas in Markdown, from Mermaid 11.17.2 and KaTeX
  0.18.9 carried inside it, their notices in the package; a widget in a sandboxed frame of its own, shown as the state
  it saved. A browser that has the page is told it is unchanged rather than sent it again. The dashboard and the
  presentation are drawn as the engine arranges them, the dashboard's tiles moved, resized and run where they stand.
  The panels are Verso's: Metadata, where the title is changed; the chosen cell's Properties, where the layout has
  them, each field described under its label; and View, which switches the layout and the theme, the page drawn in the
  notebook's theme once it chose one. A cell is chosen as in Verso's editor, and a cell shown rendered shows its text
  while it is chosen; cells are added, taken away once you say yes, moved and turned into another kind, each only where
  the notebook's layout allows it, and a cell's bar and the export menu hold only the engine's buttons that can be
  pressed there. A cell's text takes Verso's keys — Shift+Enter, Ctrl+Enter, Alt+Enter, Ctrl+Alt+Enter, Tab and Escape
  — lists what its kernel offers as it is typed, on Ctrl+Space and after a dot or a quote, says what the word at the
  cursor means, and, in a code cell, folds to its first lines. Outputs set hidden say so, and outputs cut short keep to
  the cell's lines. The parameters form draws itself, and its controls reach Verso's parameters part as Verso's own
  script sends them. Beside the notebook's name stand its kernel and how many cells it has, a status of what its
  kernels do, and a dot on Save while anything is unsaved. A run is offered only while none is under way, and its Stop
  stands where Run All stood; a run that waits for another notebook's C# run says so, and a cell a change runs, or a run
  a stop left behind, shows running with nothing to stop. The page says things as Verso's editor does: a notice that
  goes after three seconds — "Saved to titanic.verso", "Kernel restarted" — a sentence that stands while what it says
  holds, and an error in a banner with Dismiss; when the connection goes while something is unsaved, it says that is
  lost if the server has stopped. Close goes at once, even while a run is under way, and asks first only when something
  would be lost; a folder's list, shown even when it holds one notebook, makes a new one.

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

- **What can be done to a column is said once.** `Including`, `Excluding` and `WithKind` on a declaration hand back
  the steps a change to one column makes — `Including` takes several in at once too, with one kind: a column taken
  back in as it was, or out of the drop that left it out; a column no step reads excluded in the schema with its
  kind, and any other dropped below the last step that reads it and the step that makes it; a category that
  remembers the kind it came from. Asked for what already is, each hands back the steps it was given.
  `ChoicesFor(columns)` says how each column stands — taking part, excluded, dropped, made by a step and which one,
  kept with the rest, or not declared, and whether the source may lack it — and what can be done to it without
  breaking a rule, and `KindsFor(column, header)` the kinds it can be given, so the answer is never offered to be
  left out and a column a later step scales as a number is never offered to become a category. The schema's own
  `WithColumn`, `WithColumnExcluded` and `WithColumnKind` change one block alone. `WithOutput` places an output, in
  the place of the one standing or at the end, and a return directly after the split; `WithoutOutput` takes it
  away; each row says what its column is to the output — an answer comes back to it, or the way back reads it — and
  `ChoicesFor` names the output it asked about.

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
  `MeasuredCategories(column)` names the categories those rows hold, by the rule an encoder learns them by.

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
