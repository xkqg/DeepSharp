# Architecture

This file records the decisions that shape DeepSharp, and why each one was taken. A decision that is not
written down here is not a decision, it is a habit.

## The layout

```
Src/DeepSharp/Tensors/            the engine side: Shape, Tensor and the TensorStorage its values live on, Window and
                                  the border it pads an image with, ITensorBackend and what it refuses
                                  (TensorOperandExtensions), the version and the device an engine names
                                  (INamesItsVersionAndDevice), CpuBackend, and the RecordingBackend gradients are worked
                                  out through
Src/DeepSharp/Networks/           the model side
    Layer.cs Slots.cs Pass.cs Network.cs
                                    a layer, the numbers it learns and measures, one pass, a stack of layers
    Dense.cs Activations.cs Dropout.cs Normalisations.cs Convolution.cs
                                    the layers
    Initialisers.cs RandomStream.cs Philox.cs Draws.cs
                                    what a layer starts at, and the one source every draw is counted from
    Losses.cs Optimizers.cs LearningRateSchedules.cs
                                    what is brought down, what moves the parameters, how the rate changes
    Sequential.cs                   a network in Keras's words, lowered onto a stack
    CompiledNetwork.cs FitOptions.cs TrainingData.cs History.cs Checkpoint.cs
                                    the training loop, early stopping, checkpoints, and what a run did
    NetworkDocument.cs NetworkCatalog.cs Rebuilding.cs PartReader.cs NetworkText.cs TrainedOn.cs
                                    a network written down as its kinds and its numbers, and read back
    IImporter.cs SlotEntry.cs SlotLoad.cs SlotLoadException.cs
                                    the seam a reader of another framework's files implements, and the one load
                                    every number read for a slot goes through, all of them or none
    FileTextExtensions.cs           a file's own words shown in a message as words and nothing else
Src/DeepSharp.Pipelines/          the data side
    IPipelineStep.cs                what a step is: a verb, how it writes itself, what it reads, what it does
    StepParameters.cs ParameterKinds.cs
                                    a step's parameters, said once; the closed set of kinds they hold
    StepCatalog.cs                  the verbs a file may hold, and the one door a step is read through
    PipelineDeclaration.cs DeclarationRules.cs
                                    the steps in order, the rules every declaration keeps, the keys of its prefixes
    PipelineDocument.cs PipelineFileException.cs PipelineFileSchema.cs VerbReference.cs
                                    the file: its envelope, the schema, the reference
    SurveyedText.cs TextPieces.cs   the text a file is read from: where each of its parts stands, every fault at its line
                                    and column, and a larger file read a piece of its UTF-8 at a time
    PipelineText.cs                 a pipeline's text as one writer writes it, and whether two pipelines are one fit
    PipelinePreset.cs               what a pipeline decided about its columns, saved on its own
    ColumnState.cs                  which columns there are at each step, followed from the schema down
    ColumnChoices.cs                what can be done to a column, and how each stands
    Steps.cs Schema.cs Splits.cs RowOrder.cs
                                    reading, declaring, putting in order, dividing
    Features.cs TimeParts.cs Maths.cs
                                    what is worked out from a single row
    FillStrategy.cs FillNaN.cs Scaling.cs Encoding.cs Outliers.cs DropColumns.cs DropWarmUp.cs
                                    what learns, and what takes rows or columns away
    Settling.cs Marking.cs          settling a gap with a value no row decided, above the split; and the one column
                                    that says where the gaps were, named and written in one place
    Evidence.cs Report.cs Measures.cs PredictionsDocument.cs
                                    the profile and the correlation a run is declared to produce, the report of what a
                                    trained model is held to, how it measured, and the predictions it measured as text
    RowSource.cs SourceFolder.cs Binding.cs Table.cs RowIdentity.cs TrainingValues.cs
                                    rows, where a path is read from, typed columns, who a row is, what a fit sees
    FileBytesExtensions.cs          a file's bytes, once in hand, read as a file is read as text
    CellTextExtensions.cs TypedValueExtensions.cs KindProposal.cs
                                    the one reading of a cell as each kind, the one writing of a typed value as a cell,
                                    and the kinds a source's cells propose
    Walk.cs Course.cs Execution.cs Views.cs Fitting.cs Handover.cs NeedsExtensions.cs
                                    the one walk every run is, the steps a run for a learner takes and leaves out, the
                                    data after any step, what a fit learned, the handover, and what each need says
    Outputs.cs WayBack.cs           what a model is asked to predict, and how its answers come back into their units
    FileTextExtensions.cs           a file's own words shown in a message as words, the core's rule kept here too
Src/DeepSharp.Pipelines.DataFrame/   a reader of Microsoft's DataFrame, through MatPlotLibNet.DataFrame
Src/DeepSharp.Pipelines.Parquet/     a reader of Apache Parquet files, through Parquet.Net
Src/DeepSharp.Pipelines.Excel/       a reader of Excel workbooks, through ExcelDataReader
Src/DeepSharp.Pipelines.Json/        a reader of JSON files holding an array of records
Src/DeepSharp.Pipelines.Indicators/  indicators over a series, as verbs
Src/DeepSharp.Import.PyTorch/        a reader of what PyTorch saved — a safetensors file, through Onnxify.Safetensors,
                                     and torch.save's .pt, its pickle read by a weights-only interpreter of its own
Src/DeepSharp.Import.Keras/          a reader of the models Keras saved, .keras and .h5, through PureHDF
Src/DeepSharp.Import.Onnx/           a reader of ONNX graphs — PyTorch's, Keras's and tf2onnx's — through OnnxSharp
Src/DeepSharp.Learners.Networks/     where a network meets a pipeline: trained behind it, serving, the one file
Src/DeepSharp.Backends.TorchSharp/   an engine on libtorch, through TorchSharp, on the processor or a graphics card
Src/DeepSharp.Charts/                every chart, as the text of an SVG, drawn with MatPlotLibNet; the report, as HTML
Src/DeepSharp.Verso.Notebooks/       a pipeline written as a notebook in Verso
Src/DeepSharp.Verso.Api/             an application of your own that hosts the notebook
    NotebookHost.cs                 one open notebook: its state, its turn, and the read every look takes
    NotebookHost.Editing.cs NotebookHost.Running.cs NotebookHost.Panels.cs NotebookHost.Publishing.cs
    NotebookHost.Opening.cs         the same host, read a duty at a time: the cells, what runs, the parts and the layout,
                                    the versions every view is told, and opening, saving and closing
    NotebookFile.cs                 the notebook's file: what it last held, whether the notebook differs from it, writing it
    EngineExtensions.cs             what the engine is asked about itself, each question asked of the thing it is about
Src/DeepSharp.Verso.Serve/           DeepSharp's own server, the notebook in a browser
Samples/                          runnable programs, a sample notebook, and the published data they read
tools/                            the coverage check, the script that draws the icon, and the checks that start what a
                                  person installs from the packages just made: deepsharp-serve, the notebook as Verso
                                  installs it, and the libtorch engine in an application of its own
Tst/DeepSharp/                    the tests of the libraries
Tst/DeepSharp.Import.TestParts/   importers those tests read files with, written outside the library's internals, as a
                                  reader of another framework's files is
Tst/DeepSharp.Backends.TestParts/  engines those tests hand a network, written outside the library's internals: one on a
                                  storage that counts its copies, one on memory allocated outside .NET
Tst/DeepSharp.Backends.Contract/  what every engine is held to, written once against what the library publishes
Tst/DeepSharp.Backends.TorchSharp/  the libtorch engine's tests: every engine contract, on the processor and a card
Tst/DeepSharp.Verso.Notebooks/    the notebook's tests, run in the host an application of your own uses
Tst/DeepSharp.Verso.Api/          the host's tests, run on Verso's own engine the way an application runs it
Tst/DeepSharp.Verso.Serve/        the server's tests, run against the server started for real
Tst/DeepSharp.Verso.Serve.TestParts/  parts the server's tests carry beside it, found as another's parts are found
```

The tensor library and the pipeline library do not reference each other, and a test reads their assembly
references to keep it that way. They meet in `DeepSharp.Learners.Networks`, which references both and which neither
references, and the charts reference both and MatPlotLibNet; what each of those packages references is a closed list a
test reads.

The notebook's tests, the host's and the server's are suites of their own because Verso's engine and the
validator the core's tests hold the pipeline schema to each need a different version of the C# compiler, and one
test program can load only one. The libtorch engine's tests are a suite of their own because they carry libtorch,
which the core's suite never does: the processor's build for the platform they run on, as an application brings it,
or — built with `-p:Libtorch=cuda`, into a folder of its own — the one for an NVIDIA card, whose tests run only where
libtorch finds a card and are skipped everywhere else. The coverage check and the release both run every suite they
find, by the name every suite has, on every runtime the suite is built for. The check measures on the newest and runs the others:
the two builds of one assembly, measured together, merge as one module and most of its branches lose their counts,
so every class read as fully covered. The code is one code on both runtimes, so one measurement covers it. The
suites' measurements are pooled by the check itself rather than merged by the coverage tool, whose merge of the
same reports came out differently from one run to the next — once keeping every branch of a class, once losing
most of them: a line counts as reached when any suite reached it, and its branches as the most any one suite
covered, since a report says how many of a line's branches were taken but never which.

The server's tests carry parts of their own — a kernel slow to start afresh and one that fails to, a layout that fails
to draw and one that draws a slot for every cell, a part that runs the next cell on a click, one that answers a click
with what its control carried, and one that writes down the form of each value its fields are handed — in a project
that is no suite, measured by nothing and shipped by nothing. It is an assembly beside the server, because that is where
Verso's engine finds an application's parts, and it is found there as a third party's part would be. Its name does not end
in `.Tests`, since the engine passes over an assembly named so.

## The design language: PDD, pipeline-driven design

A design is named after whatever drives it, and here that is **the pipeline**.

Every piece of work follows one sequence: collect the data, choose its rows and settle the gaps that would travel into a
feature, add the features, split into train, validation and test, fill what is still missing and normalise — those two
below the split, since they learn — build the model, and check it against data it has never seen. The sequence is not the interesting part — everyone does those steps. What matters is that the
whole of it is **declared in advance as one artefact and then replayed**, instead of being assembled again
at each stage by whoever happens to be writing that stage.

That is the difference between a pipeline and a script. A script does the steps; a pipeline is a thing you
can hand to somebody, save beside a model, and run again a year later on data that did not exist yet.

### The declaration is the enforcement, and the chain is where you meet it

Everything is reached through a factory and built with a fluent chain, and the stages of that chain are
different types. A pipeline under construction offers `Normalise` and `FillMissing` **only after** it has
been told how to split, because those are the operations that learn from the data.

That is where this started, and it was not enough. A council found the rule held for the *verbs* and not
for the *steps*: the extension point every other package uses took a step that learns, and a hand-edited
file could put one anywhere at all. Both were executed, not argued. So the rule moved to the one place every
door passes through — the declaration's own constructor — and the type system now carries the concept
rather than the arrangement of methods: a step that learns from the data says so, by implementing
`IFittedStep`, and a step that divides the rows says so with `ISplitStep`.

Saying so is doing so. `IFittedStep` carries the fit and the apply, and `ISplitStep` the division itself, so
a step cannot claim to learn and learn nothing, or divide the rows by some other means than the one the
declaration can see. There used to be a marker for each and a second interface for the work, and a split
that implemented one without the other left every row in training.

A package adding a verb inherits the rule by saying which kind its step is. It cannot forget to, because
these are the only way to be either kind.

The constructor keeps the other rules a declaration has to keep, whichever door it came through — the chain,
the extension point, a hand-written file, a notebook. One source, one schema, one split, one order and one
output: a second one used to be ignored, so a file said one thing and the numbers came from another. The
schema comes directly after the source, since everything else works on columns. Rows are dropped and put in
order before the split, never after it, because the split divides the rows it is given once. A step that reads
the rows in their order stands below the step that says what that order is. Every step does something the run
acts on. Each rule is a small type of its own, the constructor refuses with every fault at once, each with the
step it is at, and `PipelineDeclaration.FaultsIn` gives the same faults without refusing — so something that
writes a declaration a piece at a time, as a notebook does, can put each one where it belongs.

Which split is a separate question, and the library does not answer it: random for independent rows, by
time when you predict forward, stratified when a class is rare, by group when several rows belong to one
entity, explicit when the data already carries the answer. Each is offered at the same point and none is
the default, because a default there is a guess about somebody else's data. What is fixed is the ordering
alone.

What every split shares is that it divides rows by what they say, not by where they stand. Each row is ranked
by a digest of its own contents and the seed, so the same rows are dealt the same way whatever order a file
lists them in, and every copy of a repeated row lands where its first copy does. Split by position, 31 groups
of repeated rows in the Titanic data had copies on both sides of the line. A split in time never divides a
moment: every row of one moment lands on the side of the line its first row does.

A split in time can also keep a gap: the last moments of every part — before each line, and at the end — are
set apart, fitted on by nothing and handed to nothing, and the fit writes down how many rows the gap held. It is
there for an answer read from later rows. Without it, the last training rows learn their answers from the rows a
model is measured on: in the published price series, five days ahead, the last five training days read their
answers from validation. The gap counts moments rather than rows, so it never divides a moment either, and a gap
that would take every row a model learns from or is measured on is refused.

Reading ahead is held to that. Only an output reads rows after its own — a feature that knows the future scores
well on every row it is measured on and on none it is asked about — and an output that does stands below a split in
time whose gap is at least as wide as how far it reads, the rows ordered by the column that split divides by and
nothing else, so the rows after a row are the ones that came after it. A return stands above every step that
changes the column it is made from, since it comes back by that column as it was read. An output that acts at all
acts by making its answer.

So fitting on the whole set is not a mistake a caller can make and be warned about later: it is a method
that does not exist yet at that point in the chain. A rule in a document is advice; a rule expressed as
which methods are in scope is the only kind that cannot be skipped in a hurry.

### A reader is an extension method, shipped by the package that owns the format

`Pdd.Create().ReadCsv(path)`, `.ReadParquet(path)`, `.ReadExcel(path)`, `.ReadJson(path)`, `await ….ReadDbAsync(reader)`
— one verb per source, and every one of them an **extension method defined in the package that brings the
dependency**.

The obvious alternative is a method per format on the pipeline type itself. It reads the same and costs
the whole architecture: the core would have to reference Parquet.Net, ExcelDataReader and a data frame,
every project would carry all of them, and adding a format would mean editing the core.

As extensions, a verb exists exactly when its package is referenced. Reference `DeepSharp.Pipelines.Parquet`
and `.ReadParquet` appears; do not, and it is not in the list. Nothing is carried that is not asked for,
and a new format is a new package rather than a change here. A reader of a file is a verb of the pipeline's file as
well — `read.parquet`, `read.excel` and `read.json` beside `read.csv` — which its package teaches a catalog,
`.WithParquet()` or its `IStepContribution` among a host's services; a file naming one in a catalog never taught it is
refused with the name of the package it needs. A reader of a file also opens its rows from the file's bytes once they
are in hand, through `IReadsAFile`, so whoever fingerprints a file parses the very bytes it hashed rather than reading
the file a second time.

The escape hatch is one method wide: `.Read(IRowSource)` takes rows and a declared schema, so a format
nobody shipped is still a few lines away rather than a fork.

### Five readers ship, and a funnel carries the rest

Reading data is a solved problem with a long tail, and reproducing that tail is a library in itself —
pandas exposes nineteen readers. What ships here is the short head, chosen by where data actually arrives
and by costing a thin adapter rather than an implementation:

- **CSV** and **SQL** cost nothing at all: the pipeline reads a comma-separated file itself, and Microsoft's
  DataFrame — `Microsoft.Data.Analysis`, from the ML.NET family — loads a query through whichever ADO.NET provider the
  caller brings, awaited once, as `ReadDbAsync(reader)`.
- **Parquet** is where data of any size lives, **Excel** is how data arrives from people rather than
  systems, and **JSON** is what an API hands back. Each is an existing .NET library plus a few lines — Parquet.Net,
  ExcelDataReader, and the System.Text.Json that comes with .NET, so the JSON package brings its reader and nothing
  else.
- **Live sources** are their own family, fetched and landed rather than read during training. None is built yet;
  each will be a package of its own, and `.ReadBinance(symbol, interval, from, to)` is the shape one takes.

Everything else stays one `IRowSource` implementation away — rows plus a declared schema, which is the one
door the DataFrame opens for anything enumerable. Not shipping a reader is not the same as refusing a
format, and that distinction is what keeps the list short.

A file a reader reads is read as the pipeline's own comma-separated one is: named in the declaration, opened when the
pipeline runs, from the pipeline's folder when its path is relative, and read again whenever the pipeline is, so a
pipeline file written with any of them replays. Every value reaches the pipeline as text and a row is known by that
text, so each reader hands a cell over as its file holds it. JSON is text, like a comma-separated file, and a number
keeps its spelling there: `22.0` stays `22.0`. A Parquet file and a workbook hold values rather than text, and hand them
over by one rule the database door shares, `TypedValueExtensions.AsCell`: a number in its shortest exact form, a moment
or a day as ISO 8601 writes it, true and false as `True` and `False`. The passenger list's cells held as its
comma-separated file writes them are therefore the same rows in every format — the same keys, the same split and the
same batches, in Parquet, in `.xlsx`, in `.xls` and in JSON — while a file that types an age as the number 22 holds
another record of that passenger than one that writes `22.0`, since a row is what its file says.

Each format has its own few rules. A workbook is read from its first sheet unless the pipeline names another, and the
sheet's first row names its columns; a cell is read as the sheet types it, never as the text a culture shows it as —
a date as a moment — an empty cell is empty text, as the same sheet saved as comma-separated text holds it, and a
formula's error is spelled as Excel spells it there. The library that reads workbooks asks .NET for a Windows code page
before it opens any of them, one that keeps no text in a code page as much as one that does, and .NET carries those
code pages without offering them; the Excel package offers them, once for the whole program, which leaves every
encoding the program already had exactly as it was, rather than hand every caller an error about code page 1252. A
JSON file is an array of records, one object a row: its columns are the keys in the order they first appear, a key a
record leaves out and a null are both gaps, and a value that is itself an object or a list is refused by name with its
record rather than flattened by a guess. A Parquet file is read row group by row group in the order it holds them, and
a column of lists, of groups of fields or of raw bytes is refused by name. Each column is read into arrays of the
reader's own, its values beside the marks of where its gaps are. Parquet.Net's typed read takes both from the pool of
arrays every reader in the process shares and, for a column that can hold no gap, leaves the marks unwritten, so
whatever the pool held there read as gaps and moved the values along: with four hundred readers beside four hundred
writers of other files, the passenger list came back with other rows in three runs of six, and read into arrays of its
own it came back right in twenty of twenty.

Read through the frame, a file is read as text, every column as the file writes it. Left to guess, the frame took
each column's kind from its first ten rows and handed back its own spelling of what it read — `133.1285` where the
file says `133.1284878`, in 3,661 of the price series' 4,554 numbers — and since a row is known by what it says, the
same file made other rows, split otherwise, than the pipeline's own reader made of it. A query's columns come as the
database types them, and that typing is kept as the database's statement of what they hold; a Parquet file's columns
are kept as its statement the same way.

### A live source is fetched once, then read as a file

Reading straight from an API is the obvious convenience and it quietly removes the one property a pipeline
exists to have. An endpoint answers differently every time it is called, so a pipeline that fetched during
training would train on different numbers tomorrow while claiming to be the same pipeline.

Fetching and reading are therefore separate. A fetch pages the endpoint and lands the result as a file,
recording what it asked for and when; the pipeline then reads that file like any other, by fingerprint. A
convenience method may do both on first use and read the landing thereafter — the declaration names the
window, the landing names the bytes.

Two consequences, neither optional: the window is closed (a `from` and a `to`, never "the most recent
thousand", which is a moving target dressed as a source), and each live source lives in its own package,
since it brings HTTP, retries and a rate limiter that a project serving a trained model has no use for.

### The evidence is part of the declaration

A run also declares what it must produce as proof: which measures are computed — root-mean-square error,
mean absolute error, R², accuracy, a confusion matrix — on which splits, and whether they are shown as a
grid of numbers or drawn. Naming them before the numbers exist is the point: every run then produces the
same evidence, two runs are comparable without anyone remembering what was shown last time, and the report
cannot quietly shrink to whatever happened to look good.

Three kinds are built. Two are about the data before anything learns from it: a profile of the columns, which
says for every problem it finds how it is answered, and the rows a correlation is drawn from. Each is
measured on the rows the split trains on — the split below it as much as one above — because a profile over
every row lets the rows a model will be measured on shape what it is shown. The third is the report: which
measures a trained model is held to, on which parts, and how they are shown. What they produce is output: it is
kept with the run — in `PreparedData.Evidence`, or in the measures a trained model's predictions were given — and
never written into the pipeline's file, since the file is what is replayed and a replay learns nothing.

A report, `evidence.report`, is declared before any number exists, and the run acts on nothing for it, as it acts on
nothing for an output that only names its answer. There is one at most, below the output whose answers it measures;
a pipeline that names no answer, or never divides its rows, has nothing to measure and is refused where the report is
written. It measures the rows a model learns from, is chosen on and is tested on, and no others. A measure that counts
classes — accuracy, precision, recall, the confusion matrix — is refused where it is written against an output whose
answers are amounts, a distribution's shares or a return; against one whose answers can be classes, a row whose answer
is neither nought nor one is refused when it is measured, naming the row as it was read. `PreparedData.Measure` takes
what a model predicted for each part and gives the measures in the answer's own units — the predictions and the
answers both come back through the way back, because in normalised units every error is small and every model looks
excellent — each beside the same measure of predicting, for every row, the average of the training rows' answers, and
each by scikit-learn's definition, checked against it to a millionth of a millionth. Predictions are measured only when
they are for the part's rows in the order the part hands them over, which the rows' keys are how to tell: answered in
any other order, 346 of 349 price rows and 510 of 512 days of bikes were measured against another row's answer, and
nothing in the numbers said so.

What a profile finds is what should not be there, with the columns it is about and how it is answered — a step the
column rules keep for the column's kind, the column left out as those rules leave one out, or a value the schema says
stands for a gap. A gap among numbers is answered by a fill, among true and false or moments by dropping the rows it is
in, and among words by the encoder, which marks a gap and makes it no category: no column set when one-hot, no place
(minus one) when handed to a learner that takes categories; written as places, it takes the first place and only its
mark tells it apart. Naming a fill for a column no fill takes named a step the rules refuse. A column that goes with the answer value for value hands a model the answer; one
that goes with a column before it says again what that column says; one whose values are each a row's own — words no
two rows share, or a running number the rows are neither ordered nor divided by — only names its row. Each is
answered by leaving it out. Two columns are compared only where their values repeat, each held by two rows or more on
average, since columns that never repeat a value go with each other by chance; a profile therefore knows the
pipeline's answers and the columns its rows are ordered by, which a view carries. A number far from every other, held
by more than one row and written as files write that nothing is known — 0, −1, a run of nines — is answered by the
schema saying so. An extreme is not something that should not be there: it is measured, and clipping or refusing one
is a step somebody declares.

Every figure is drawn by one package, `DeepSharp.Charts`, with **MatPlotLibNet**: a correlation as a heatmap, a
report's measures as bars beside the average, each confusion matrix as a heatmap of counts, what was predicted against
what was there, and what was left over. The pipeline holds only the declaration and the numbers, and the notebook draws
through the same package, so a figure is drawn one way wherever it is shown.

The report itself is rendered there too, and only there: `measures.Report()` is the measures as the report says they
are shown — the numbers, the charts or both — as one value whose public `ToHtml()` gives its HTML. Verso shows any value
with such a method as HTML, finding the method by its name, so it does not matter which assembly the value's type was
loaded from; a C# cell that ends with the report shows it, and the notebook's report block draws the same rendering once
the cell that trained the model hands its predictions back. A formatter of DeepSharp's own was the other way to show it,
and it does not work: Verso picks a formatter by the types it names, and a cell's types are not the notebook's.

### The rule that gives it meaning

**Anything that learns from the data is fitted on the training split alone, and then replayed unchanged.**
A mean and a standard deviation, the value a fill learns for a gap, the categories an encoder knows, the bounds
of a clip — each is a parameter, each is learned once, and each is applied identically to validation, to test, and to
data arriving in production long afterwards. A value no row decided — a nought, a number you chose — is not one of
them, and that is what lets `settle.gaps` stand above the split.

Break it and nothing goes red. A mean computed over the whole set gives a model that scores beautifully in
validation and disappoints the day it meets real data, because the validation rows had already been allowed
to influence what the model saw. The same failure wears a second costume at serving time: a feature computed
one way while training and another way live, which is the same leak running backwards.

### Three words for what a fit writes down

A fit does not only learn. It **learns** a value — a mean, a bound, the categories it found — and a replay reads it. It
**decides** something from what it saw, such as whether a column with that many gaps is filled at all, and a replay reads
that too, so rows served long afterwards keep meeting the pipeline the model was trained with. And it **sees**: how many
gaps there were, how many values were no number, how many lay outside its bounds. The third is evidence, written down so a
person can see what the fit was looking at, and no replay may depend on it.

The three are said apart in one bag, because that is what the file holds: one object of names and values, in one order,
marking none of them. So a pipeline published before the three words existed still loads, and a bag read back from a file
counts nothing — the words are the fit's own, where they can be held to. What the distinction buys is a rule that can be
tested rather than hoped for: nothing a replay reads was only counted.

### What that makes the carrier

The pipeline is the artefact: the ordered steps plus what they learned while being fitted. It is saved
beside the model, because a model without it cannot be used — the numbers reaching it would not be the
numbers it was trained on.

The intermediate datasets are deliberately *not* the carrier. Naming and storing every state between steps
multiplies the things that can drift apart and answers a question nobody asks; a replayable pipeline
produces any of those states on demand, and it produces them the same way every time.

### A gap is filled by a named strategy, never by a flag

Two things get thrown onto one heap elsewhere and are kept apart here. A value is **missing** when it was
never there — an empty field, a NULL from a database — and that is data. A value is **not a number** when
arithmetic produced no number, almost always a division by zero in a derived column, and that is a fault
further upstream. They deserve different verbs and different defaults:

```csharp
.FillMissing(fill => fill             // fitted on train, a kind a group of columns:
    .Mean("trades")                   // Mean, Median, Zero, Constant(0), Previous, Refuse
    .Previous("volume"))
.FillNaN("range", With.Refuse)        // the default: a NaN stops the run, because it should not be there
```

Dropping the row is not among the strategies, though it was once meant to be. A strategy stands below the split,
and dropping rows there would quietly change how many each part holds — the shares the split promised. So it is
a verb of its own, `DropGaps`, that stands above the split beside `DropWarmUp`: the rows are chosen first and divided
afterwards. `SettleGaps` stands up there as well and keeps the row instead, putting in a value no row decided. Leaving a
column out is a verb too, `Drop`, anywhere below the schema.

The strategy is a named value rather than a flag. A boolean parameter says nothing at the place it is
written — `Fill("trades", true, false)` has to be read with the signature open beside it — whereas the
name is the sentence.

Marking where a gap was is not a parameter either: a `trades_was_missing` column is emitted alongside, and
always. Filling destroys the distinction between "absent" and "the value happened to be that", and it
destroys it irreversibly; a caller who does not want the column drops it like any other column, which is a
verb they already have.

A value that is not a finite number is refused by every fit, not only by `FillNaN`: one not-a-number among the
training values became the centre a scale was built around. The fit names `fill.nan` as the step that deals
with it, and the handover refuses one as well.

A column worked out from another carries that column's gaps, and filling what it was made from does not reach back
into it: the features are added above the split and the fills stand below it, because what a fill puts in a gap is
learned from the training rows. So a sum of two columns, one of which has a gap, is itself a gap — unless the gap is
settled where the features are worked out, which is what `settle.gaps` is for. It puts in a value no row decided, so
nothing of validation or test taught it anything; it writes nothing into the fitted half, because it learned nothing;
and a feature worked out after it is worked out from settled columns. `DropGaps` and `DropWarmUp` stand there too and
take the rows out instead. The handover says all of this when it meets a gap it cannot hand over, naming the columns
the feature was worked out from rather than only that a gap is a gap.

### The declaration is a file, and the chain can write it

The pipeline is saved as one file with two blocks, because they have different authors. The
**declaration** — the steps in order with their arguments — is written by a person. The **fitted** part —
the means, the fill values, the category lists, the boundary the split landed on — is written by the fit.
Keeping them apart is what allows the same declaration to be re-fitted on fresh data, two runs to be
compared by diffing the declaration alone, and a serving process to load the file without ever knowing the
builder existed.

```
{
  "version": 6,
  "declaration": [ { "step": "read.csv", "path": "titanic.csv" }, … ],
  "fitted": [
    { "step": "split.stratified", "prefix": "9e27e6…",
      "learned": { "rows.train": 623, "rows.validation": 133, "rows.test": 135, "rows.predict": 0, "digest": "b457b5…" } },
    { "step": "fill.missing", "prefix": "c1a1de…", "learned": { "gaps": 133, "value": 29 } }
  ]
}
```

Every fitted entry is tied to the steps it was fitted behind. Its `prefix` is a key made from its own step
and every step above it — a chain of SHA-256 digests, each over the key before it and the step as the step
writes itself, so the spacing, the order of the keys and the spelling of a number in the file make no
difference. A fit is never used under steps that changed after it was learned: read back by position, a fit
spliced under another declaration served a price of 135.7 where 0.9048 was meant. The split writes an entry of
its own, how many rows went to each part and a digest of the rows it divided, which does not depend on the
order they came in: what the fit saw travels with what it learned. A run made for a learner that does without some
steps writes a third block, `skipped`: each step it left out, by its verb and its prefix, so the file says what that
learner was fed and a replay leaves out the same steps.

The format is deliberately not the pipeline. One internal declaration model is the truth, and every format
is a front end that produces it: JSON for machines, because it diffs and travels; YAML for people, because
it carries comments and loses the punctuation; a spreadsheet or a generated form, for the same reason and at
the same cost, which is a parser rather than a redesign. The same seam as the readers. JSON is written today,
and a notebook — block by block, or field by field in a generated form — is the second front end.

What a pipeline decided about its columns can also be saved on its own, as a preset: the schema, the columns
dropped after the steps that read them, the output, and the source's columns as they were last shown. It holds no
fit and no rows, only decisions, so a pipeline written again over new rows can take it over. It goes through the
same door as the pipeline file — read with a catalog, every fault at its line and column, a newer version refused
whole — because a second reader of the same steps would be a second set of rules. A preset names its version
without exception: none was written before the second, and its output may be a word only the second has.

Taking a preset over lists before it applies. `TakeOver` makes the steps — the schema whole, in place of the one
there or directly after the source; the drops made the preset's; the output the preset's, placed whole, or none — and
lists every column whose decision that changes, from how it stood to how it stands after: every part the schema
writes of it — how it stands, its kind, the kind a category was, whether the source may lack it, how a timestamp's
moments are written, which value stands for a gap. What a column offers and what it is to the output follow from
those, so they are not listed for themselves; the output is listed once,
before and after, the schema's order when the columns both schemas name stand in another, and what the schema does
with the columns it does not name when that changes. A column a step makes is never said to be missing from the
source. The source's columns the preset never showed are named new. A drop the preset saved can only be made of a
column that reaches the end of the pipeline: one that nothing makes any more, that the schema leaves out, or that a
step below takes away cannot be dropped, and it is listed as not made, with how it stands, rather than refused —
refusing would hold back every other decision the preset saved, and a notebook's saved columns are written again from
the blocks at their next change anyway. A take-over whose steps would break a rule is refused with every fault and
applies nothing, and the same pipeline, preset and header give the same answer whoever asks: a notebook's list, its
Apply, and code. A source's header is read on its own, by the same reading as the whole file, so a large file costs
its first line; a Parquet file, a workbook or a JSON file says its columns as its reader opens it, and every door asks a
source its columns the one way, `INamesItsColumns`. A chain written in code takes a preset over in two places, because it is written in order: the
schema where it declares its columns, directly after the source, and the drops and the output where it names its
answer, once the steps that read and make those columns stand. Each place lists what it decides before anything
runs, the second is the one take-over every door makes, and together they list each decision once — there is no
third, merged listing, since no door ever stands where the chain did before its schema.

One thing does not survive being written down, and it decides the shape of the rest: an inline lambda. A
custom step is therefore registered under a name and looked up while parsing, and a file naming a step that
nobody registered **refuses to load** instead of quietly skipping it. That pushes features towards a named
vocabulary, which is exactly what makes the file portable.

### One validator, and a template that falls out of the same model

The template and the validator are both generated from the declaration model rather than written beside
it. Anything hand-maintained drifts from the code it describes, and a validator that drifts is worse than
none: a file can then be legal in a way the fluent chain is not.

Validation has two halves and they run at different moments. **Shape** — does the step exist, are its
parameters the right kind, do the shares fit inside a whole, is every column read where a column of that name
and kind exists — needs no data at all. The columns are followed from the schema down, each step saying what
it leaves behind, so a column the schema left out or a step above took away is refused at the step that reads
it, not a whole run later as a column nobody could find. **Binding** — are the declared columns in the source,
and do the steps still find theirs among the columns it actually bound — needs the source open, and runs
before the first step does. Both report every fault at once with its position in the file. Someone who is
handed one error at a time, five times over, stops using the thing. A cell that cannot be read as its column's kind
is named with every other such column, at the schema's step: one cell with its row and value, more with how many
and the first three. Every cell is read by one reading for each kind, the same whoever asks what a column holds, so
two parts of the library never disagree about whether a cell is a number.

```
btceur.pipeline.json(9,5): Step 4: 'feature.indicator' is a step from DeepSharp.Pipelines.Indicators, which is not registered here. Reference the package and register its steps with the catalog that reads this file.
btceur.pipeline.json(10,5): Step 5: The step 'split.byTime' cannot be read: The shares add up to 1.3 and a split has to use every row.
btceur.pipeline.json(11,5): Step 6: 'normalize' is not a step anything here knows. The nearest one it knows is 'normalise'.
```

The first message is deliberately not the same as the last. "I do not know this step" and "I know it, but you
have not installed it" are two different problems for the reader, and collapsing them costs an afternoon. A
refusal a step raises in the language of a C# argument is reported in the file's words, without the name of a
parameter the file never had.

The file also names the version it was written against, and each verb the version from which it means what it
says now. A pipeline from a year ago either loads, or says precisely which step changed underneath it: when the
splits began to divide rows by what they hold, a file from before names a split that no longer does what it was
written to do, and it is refused by name rather than run the new way. A file newer than the library is refused
whole, since it may hold words this one does not know. The number goes up for that reason too: once a timestamp
column could say how its moments are written, a file became able to say something an older library would not
understand, and that library now names the newer version instead of stumbling over the word. The report is such a
word: new in the third version, so a file that says it was written against the second and names one was written by no
library that meant it, and it is refused by name, as the schema refuses it.

### The share you never write down

Three numbers that have to add to one is a rule a caller can break, and the way it breaks is quiet: shares
of 0.70, 0.15 and 0.10 leave a twentieth of the data in no split at all, the run works, and every number
after it is computed over less data than anyone thinks. So the share to be measured on is not a parameter.
You write what training takes and what validation takes; test is whatever is left, and nothing can add up
to more than there is. Writing them as percentages or as fractions says the same thing — `80, 10` and
`0.80, 0.10` — and mixing the two in one call is refused rather than read.

`Predict(10)` holds a fourth part back, outside the three: nothing is fitted on it and nothing is measured
on it, so running a trained network over it is the closest thing to running it tomorrow. Training,
validation and test are then the ninety that remain. It is written once, before the split, and the split is
what carries it — a pipeline that holds rows back and never divides anything is refused at the point it is
built, because that share would otherwise simply disappear.

### One description drives both doors

A pipeline can be written in C# or written as a file, and both have to reach exactly as far. That is only
true if neither is the description: the step types are, and everything else is derived from them. Each step
lists its parameters once, each of one kind from a closed set — a column, a list of columns, a number, a share,
a word from a set — and from that list the step is written and read, a key nobody defined is refused, the JSON
schema of the file and the template a new step starts from are generated, the reference of every verb is
written, and the notebook builds a step's form. Add a step, and every one of those gains it without anyone
remembering to go and edit it; the schema and the reference are committed files a test compares with what the
steps say, so neither can fall behind in silence. A parameter may hold only some of its set — a report measures the
parts a model is trained, chosen and tested on, and no other rows — and then the schema, the reference and the form
offer only those, and the reader and the step refuse the rest where it is written. The word a choice is written as is
one rule too, `Word()`, which the reader, the writer, the notebook and the charts all name a choice by.

The property that keeps it honest is round-tripping. Build a pipeline in code, write it out, read it back,
and the two declarations must be equal — a test that fails the moment one door learns something the other
cannot express. Without it, "both work" is a claim that decays silently, one step at a time.

### A form is how a signed value is written down

Three ways to write the same signed number, offered wherever a step produces one: `Form.Signed` keeps it as
it is, `Form.Unit` shifts it to 0..1, and `Form.SplitSign` writes it as two non-negative columns,
`max(x,0)` and `max(-x,0)`.

The first two are the same feature up to a shift, so any layer with a bias absorbs the difference; the
choice between them is about having one range across the whole feature block, not about the model. The
third is the only one that adds anything: each half carries its own weight, so a rise and a fall can be
answered differently instead of being forced into mirror images. It costs a column and stays exactly
reversible, `x = pos - neg`.

Two consequences worth writing down. `Form.Unit` moves zero to 0.5, so any later step that treats zero as
special now means "the middle". And `Form.SplitSign` has to be the last form applied — a per-column
normalisation after it would scale the two halves by their own extremes and give one quantity two different
slopes, so a step that would do that is refused rather than run.

### An indicator is a feature with a memory

A moving average, a relative strength index, an average true range: arithmetic over the rows that came
before, learning nothing from the set as a whole, and therefore a feature, above the split.

Two properties are not negotiable. The window looks only backwards — a centred or forward-reaching window
is a leak in mathematical dress, because the row would carry what had not happened yet. And the warm-up is
a gap with a known cause: an indicator of period N has no value for the first N rows, so those rows are
marked missing rather than filled with a zero that reads like a measurement, and what happens to them is a
written step.

"Backwards" presumes an order, and a file does not promise one: an export or a query without an ordering hands
rows over in whatever order it happened to hold them, and the same prices listed newest first gave a five-day
average of 98.352 where the right one is 99.74. So the order is declared, with `order.by`, above every step
that reads the rows in their order — an indicator, the warm-up drop, a fill that carries the previous value
forward — and such a step without one is refused. The order is taken from values compared in their own kind,
and two rows whose keys are equal are refused, because nothing then says which came first.

The implementations are borrowed, not written. Where a list of them already exists it is adapted rather
than reproduced — with one wrapping requirement measured on a real one: an indicator that returns a shorter
array than it was given has dropped its warm-up rows, so the adapter re-aligns the result by the known
offset. Silently accepting the short array shifts every value onto the wrong row, which changes nothing
visible and corrupts everything afterwards. Indicators live in their own package, since a project that is
not looking at market data has no use for the list.

### Normalising is a family, and two of its choices are declared

Standard, min-max, max-abs, robust, quantile, power and midrange each learn something different — a mean
and a spread, two extremes, a magnitude, a median and its quartiles, a whole distribution, a shaping
parameter — which is why the kind is named in the declaration and what it learned is stored apart from it.
Standard and min-max are both moved by a single extreme value, so on prices and volumes the robust and
quantile forms are the ones that describe the data rather than the spike.

Midrange is min-max centred: the middle of the training range becomes nothing and its ends minus one and one,
which is where a network takes its features — scikit-learn's `MinMaxScaler` with a feature range of minus one
to one. It learns the two extremes min-max learns and stores them as the same middle and spread, so its way
back is the one every linear scale already has. Where each scale lands the training rows is one rule: between
nothing and one for min-max and quantile, between minus one and one for max-abs and midrange, in no range for
standard, robust and power, which centre a column and leave its extremes where they fall. A value within the
rounding that its own centre and spread carry, of an end the training rows reached, lands exactly on that end
rather than a shade past it — measured on a published price series, the training maximum otherwise came back as
1.0000000000000004, refused by the very fit it came from.

A spread no larger than the rounding its own arithmetic carries — how many values, times the step between one double
and the next, times the largest of them — is nothing, and the column is divided by one, as scikit-learn decides a
feature is constant. Two hundred of 0.1 average to 0.10000000000000007, and the standard deviation around that came out
as 6.9e-17: taken for a spread, it made every training row −1 and a later 0.2 more than a quadrillion.

Row-wise normalisation is a different verb, not a member of this family: it works across a row, learns
nothing, and is fitted nowhere. Where it lands its columns is a fact of its arithmetic rather than a choice — a value
divided by its row's size is between minus one and one for every norm there is — so it is the one landing nothing
declares. Its sizes are worked out around the largest value in the row, as every library that measures a length does:
squaring the values themselves lost a row of 1e-160 to a length out by a relative 5.6e-6 and sent a row of 1e200 to
infinity, which made every value of it nought.

**One line scales many columns, and the kind is the method.** `.Normalise("age", "fare")` scales both the way a caller
who names no kind means it, and `.Normalise(scale => scale.MidRange("age", "fare").Robust("volume"))` names a kind per
group. What reaches the declaration is one step a column either way, so the file, a notebook's blocks and every fit
keyed by the steps above it are what they always were: the line is a door, not a shape. The same door is on
`FillMissing`, `FillNaN`, `Reshape`, `Cyclical`, `ClipOutliers` and `TimeParts`, written out by one fan-out
(`IDeclaresSteps`, `LineExtensions.Declared`) rather than by a copy of it per verb, and a line that names no column is
refused where it stands. What a column needs beyond the kind stays on the verb that takes one column — the share above
which a fill is refused, the column a reshaping writes into, what happens to a value outside a clip's bounds — because
it belongs to that column and not to the group. Three verbs keep the one-column door and no line, each for a reason
written down in the test that checks it: `Encode`, because `EncodeCategories()` already writes every category at once;
`Target` and `Ahead`, because a pipeline names one answer.

**Where the features land is said once, for the whole pipeline.** `.DefaultFeatures(Form.Signed)` is between minus one
and one, which is what a pipeline that says nothing means; `.DefaultFeatures(Form.Unit)` is between nothing and one. It
is written above the split, where it holds for every column alike, and it decides two things: what a scaling that names
no kind is written as — midrange for signed, min-max for unit, by the one rule that also says where each scale lands —
and how a moment on a circle is written. A column that names its own kind keeps it. Nothing of it reaches the file:
each column is written as the kind that lands it there, so a pipeline that says the range and one that names the kinds
are the same text. Said twice, or said below a step it would have governed, it is refused where the second is written;
`Form.SplitSign` is refused outright, since writing a value as two columns is no answer to where one lands.

The scale a caller who names none means is written down once, as `NormaliseStep.DefaultScale`, and the range it lands
in as `NormaliseStep.DefaultFeatures`. A default is a compile-time constant in C#, so a signature cannot read it from
the parameter that governs the file; the parameter reads it from there instead, and a test holds every door that lets
the scale be left out to that one value. It used to be written twice and read a third way — the chain and the step's
own constructor each said standard, while the value a new block starts with said midrange — so the same omission meant
two different things depending on which door it came through.

Two decisions are made here rather than discovered later. What happens outside the learned range while the
model is running — clip, pass through, or refuse — because a price meets a new high and min-max has no
answer of its own. Every pair does what it says or is refused where it is written: a scale that lands its rows
in no range has nothing to hold a value in or refuse one outside, so it only passes, and a quantile scale ranks
a value among the training rows and has no place beyond them to pass one to, so it holds it at the edge or
refuses it. A refusal written into a pipeline and then ignored is worse than none: it reads as a guard that is
not there. And whether the answer is normalised, because if it is, the way back is part of the
saved pipeline; without it every error is reported in normalised units and every model looks excellent.

### The pipeline ends at the data, and the learner is a plug

Everything up to and including normalising is declared once, whatever is going to learn from the result, so that
is where the pipeline stops: a prepared, split dataset plus the declaration of what the run has to prove.
What learns from it is chosen at that seam — a network built here, a trainer from an established .NET
machine-learning library such as ML.NET, Microsoft's own, or something a caller wrote — and each plugs in at the
same point.

That is worth more than the convenience. Two learners compared on the same prepared data and the same
declared measures can honestly be compared; two learners each fed by their own preparation cannot, and that
is the usual way a comparison between models is quietly meaningless. For tabular data a gradient-boosted
tree regularly beats a small network, so this is a real branch rather than a courtesy.

One thing keeps it honest. The same prepared data is not the same *representation* for every learner: a
network needs everything numeric, expanded and on one scale, while a tree is indifferent to scale and is
actively harmed by a wide one-hot expansion of a high-cardinality column. So a learner states what it needs
— numbers only, categories handled natively, scale-insensitive — and the steps it does not need are skipped
**by declaration and recorded as skipped**, never dropped in silence. The artefact then still says exactly
what each run saw, which is the only reason the comparison means anything.

A learner says which of four needs it has — `Numbers`, `OneScale`, `NoScale`, `Categories` — and `RunFor(needs)` runs
the declaration for it: a step that only scales a feature is left out for a learner indifferent to scale, and an encoder
hands each category over as itself, its place in the list the training rows held, to a learner that takes categories. A
step says which learners need it, and what stands in its place for the others, by `IMeetsANeed` — only a step that
learns can, since a source, a split, a drop or an added column decides what the data says for every learner alike. A
step stays whenever leaving it out would change more than what the learner does without: an answer's way back runs
through it, a step below reads what it wrote, or it refuses what it was not fitted on. What is left out is decided from
the declaration and the stated need alone, and written into the run's file under `skipped`; a run that leaves nothing
out writes the file it always wrote. A learner is handed a run only when it does without every step the run left out.

And the preparation happens once, here. A learner that brings its own normalisers does not get to use them:
running them again would leave the saved pipeline describing something other than what the model was
actually fed.

### That seam is the first package boundary

The pipeline and the things that learn are separate packages, and the reference only runs one way: a
learner package knows the pipeline, the pipeline knows no learner. It is the first cut in the library
because it is the one that decides what a project has to carry — a service that prepares data and hands it
to an established .NET trainer should never drag a tensor engine along, and a network that trains on data
somebody else prepared should not drag a CSV reader.

The contract between them lives on the pipeline side: prepared splits, the declared evidence, what a learner
says it needs, the step a declaration names its learner in — which carries that need and nothing else the pipeline
reads — and which steps a run left out for it, and what a model says of the rows it predicted — which of them it learned nothing about, for the
report to count — together with the text those predictions cross in, to be measured again wherever the same pipeline
runs. Everything else about a learner is its own package's business.

A cut like this is not kept by intention, so it is pinned: a test reads the assembly references and fails
the moment the pipeline acquires one it is not allowed to have. Direction is easy to state and easy to
break, and by the time it is broken the fix is no longer a line.

The first learner is a network, and it meets the pipeline in a package of its own, `DeepSharp.Learners.Networks` —
the one place both are referenced. The tensors never learn what a pipeline is and the pipeline never learns what a
network is: the measures of a trained model are the pipeline's, since they are held to its parts and its way back, and
the history of a run is the network's.

### A package is named after the role it plays, never after the vendor it brings

The segment after `DeepSharp.` says which side of a seam a package sits on; the segment after that says
which outside thing it carries.

```
DeepSharp                   tensors, the backend seam, the light engine, the layers and networks, the training loop
DeepSharp.Pipelines               the pipeline, ending at prepared splits and the declared evidence
DeepSharp.Pipelines.<Format>      a reader: Parquet, Excel, Json
DeepSharp.Pipelines.<Source>      a live source: it fetches and lands, it does not read during training
DeepSharp.Learners.<Name>   something that learns from prepared data, behind the learner seam
DeepSharp.Backends.<Name>   an engine behind ITensorBackend
DeepSharp.Import.<Name>     reading weights or a model trained somewhere else
DeepSharp.Charts            drawing, and the report rendered once, from what the loop and the measures already keep
DeepSharp.<Host>.<Part>     a front end inside a host: DeepSharp.Verso.Notebooks, .Serve, .Api
```

Naming by role rather than by vendor is not tidiness. A package called after a framework implies that the
framework is *in* there as itself, and for one of them that would quietly contradict the design: the
declarative vocabulary — stack the layers, compile, fit — is a **way of speaking**, not an engine. It lives
in the core and lowers onto the same model the imperative door produces, so there is nothing to put in a
package named after it. What deserves a package of its own is reading what that framework *saved*, and
`DeepSharp.Import.<Name>` says exactly that and nothing more.

The same test applies to the other two. An engine belongs under `Backends` because a model cannot tell
which one is underneath; a trainer from an established .NET library — ML.NET's, say — belongs under `Learners`
because it sits beside the model rather than below it. Both distinctions disappear the moment a package is named
after the logo instead.

A front end is the one exception, by decision: it is named after its host first and after what it is there
second — `DeepSharp.Verso.Notebooks`. It carries nothing of the host inside it, it plugs into it, and a person
looks for it by the host's name, in the host's own list of extensions. Another package for the same host takes the
same prefix: `DeepSharp.Verso.Api`, with which an application of your own hosts the notebook — it carries Verso's
engine rather than plugging into it — and `DeepSharp.Verso.Serve`, DeepSharp's own server that shows the notebook in
a browser, built on it. The packages are not Verso's; they are DeepSharp's way into it.

A project is created when there is code to put in it. Six empty assemblies laid out in advance are a
diagram that has to be maintained; the layout above is the decision, and each package appears the day its
first type does.

### Four seams, one interface each, and a package implements exactly one

What makes the layout above a structure rather than a filing habit is that every satellite package exists
to implement **one** interface, and the four are not interchangeable:

| seam | the question it answers | who implements it |
|---|---|---|
| `ITensorBackend` | where does the arithmetic run | the light engine, and anything heavier |
| `IRowSource` | where do the rows come from | a file format, a database reader, a landed fetch |
| a learner | what learns from prepared data | a network here, a trainer from ML.NET or elsewhere |
| `IImporter` | what does a model trained elsewhere look like here | a reader of one framework's saved models or weight files |

A package that implements two of them is doing two jobs and should be two packages; a package that
implements none is a convenience and belongs in whatever it is convenient for. A front end is the one
deliberate exception: the notebook implements none of the four, because it is not a part of the pipeline but a
way of writing one and looking at it, and it sits beside the seams the way the charts do.

A seam is only real when two implementations of it differ in kind, so each one is held to that: the backend
has a pure-managed engine beside a native one, rows arrive from a file and from a database reader, a
learner is a network or a decision tree, an importer reads numbers alone into a network it is handed or builds
the network its file describes as well. An interface
shaped around a single implementation is not a seam, it is that implementation with a longer name — and the
cost is paid later, by the second implementation that turns out not to fit.

### A column nobody declared is not carried, and that is a safety rule

The split protects against learning from the wrong **rows**. It does nothing at all about the wrong
**column**, and that gap is not theoretical: walking one row of a public dataset through this design found
a column holding the answer in words. In the Titanic set as it is published, `alive` maps one-to-one onto
`survived` — measured, 549 rows of `0`/`no` and 342 of `1`/`yes` — and `class` maps one-to-one onto
`pclass`. Predict `survived`, carry everything the file happens to contain, and the model is handed the
answer as an input. It scores beautifully on every split, and nothing goes red, because no fit was
corrupted: the leak arrived as a column.

So `Declare` names which columns take part, and what happens to the rest is declared rather than assumed:
dropped, passed through, or handled. Dropping is the default, because a column nobody thought about is a
column nobody checked. A profile also names a column that goes with the answer value for value, as it names
`alive` beside `survived`, but that is the second line of defence; the first is that unnamed means absent.

A column can also be named and left out. `excluded` keeps it in the schema with its kind while nothing reads it,
the source is not asked for it and the rest of the file does not reach it, so bringing it back is taking a word
away rather than remembering what the column was. A category says which kind it `was` for the same reason:
without it, making a column a category overwrites the only record of what it held before. Leaving a column out
this way is a decision about the columns, so it lives where the columns are declared; and since neither word is
written where it says nothing, a schema that uses neither is the same bytes, and the same key, as before.

What can be done to a column is said once, for every door that changes the columns. Taking one in, leaving it
out and changing its kind each hand back the steps they would make, and never judge them: the rules every
declaration keeps do that, so what is offered and what is allowed cannot drift apart. A declared column no step names is
left out in the schema, wherever it goes after — a category among them that the step encoding every category turns
into columns of its own, which never reaches the end itself, since its columns do, and which a box once could not
untick because a column that does not reach the end was taken to be out already. The last such category is not
offered, since the step would then have none to encode. A column a step names, or a step made, is dropped after the
last step that reads it, since excluding it from the schema would leave that step reading nothing. How a column stands is said without where a
drop happens to stand, because that follows from the steps rather than from anything a person decided. Asked for
what already is, an operation hands back the steps it was given, and that is how the same gesture twice does the
same thing once.

### A fill has a point beyond which it is invention

Measured on the same dataset: `deck` is empty in 688 of 891 rows, 77 per cent, and 77.7 per cent within
the training split alone. Filling it manufactures 688 values out of 203, after which the marking column
carries every scrap of information the original had.

The design knew two answers — fill it, or drop the row — and needed a third. A declared threshold above
which filling is refused, leaving the marking column to speak for itself, so that a decision this large is
made once, in the open, rather than by a mean quietly copied into three quarters of a column. It is
`refuseAbove` on the fill, a share of the training rows, and it has no default: published practice puts the
line anywhere between two fifths and four fifths, and a number nobody chose would be the same quiet decision
in another place.

What a fill counts, it counts on the training rows, like everything else it learns: the Titanic ages have 177
gaps in the file and 133 among the rows a stratified split trains on.

### Reading a value is reading a dialect

Two columns in that file hold `True` and `False`, capitalised, which is how one popular tool writes a
boolean and is not how any of the others do. A parser expecting `true`/`false` or `1`/`0` does not fail on
them: it reads text, and the column silently becomes categorical.

Every scalar kind therefore has its accepted forms written down and parsed under the invariant culture —
which also settles the decimal point, the thousands separator and the date order before they can settle
themselves differently on somebody else's machine.

A moment is the kind whose forms are too many to write down, so a timestamp column reads the forms ISO 8601 writes —
a date, a time to the minute, the second or a fraction of it, a zone or an offset or none — or exactly the format its
declaration says, as .NET writes one: `dd/MM/yyyy`. Nothing in between, but for the round-trip form a database or a
typed frame hands its moments over in, which is ISO 8601 too and which no format mistakes for another moment, so a
column with a format reads it as well. A reading that tries every form it knows
fills in what the text leaves out: it read `7.25` as the twenty-fifth of July of whichever year it ran in, `12:30`
as that time on the day it ran, and `02/03/2015` as the third of February whoever wrote the file. The first two
change with the day the pipeline runs, which is the one thing a replayed pipeline cannot do, and the third is a
guess about the order of day and month that the data cannot settle. A cell the column's forms do not read is refused
at the schema, with its row, its column and the form it was expected in, so the format is said once, in the
declaration, rather than guessed on every run.

A file also has its own way of writing that nothing is known: 0 for a fare nobody wrote down, -999 from a sensor that
was off, a question mark in a column of numbers. Read as it is written, 0 is a fare like any other — the cheapest one —
and no step can tell it from a real one: it lies inside the bounds an outlier rule learns from the training rows. So the schema says it,
once, on the column: `missing`, a value that stands for a gap, turns every cell that holds it into the gap it is before
the kind reads it, and every step downstream treats it as one — a fill counts and marks it. It is compared as the
column's kind reads it when it reads as that kind, since a frame hands over 0 where the file writes 0.0, and as it is
written otherwise, so a column of numbers can say that `?` is a gap without every question mark being refused. A row is
still known by what the file wrote, so saying it moves no row to another part of a split. It is part of the schema
rather than a step, because it is a fact about how the file writes, like its format, and nothing is learned from it.

### What a column holds is proposed from every cell, and said by a person

Nobody should have to type fifteen kinds a file already shows, and nobody should find a kind decided for them. So a
source's cells are read into a proposal — `KindProposal.Of(source)`, or `ProposedKinds()` on the chain before the
schema — and the schema is still what a person writes. The proposal is a value to decide from: nothing in the chain
takes it in, no file holds it, and a notebook keeps it beside the rows it was worked out from, once for each state of
the source's bytes. A pipeline that took a proposal in as its schema would work its schema out again from whatever the
file held on the day it ran, which is the one thing a declaration exists to stop.

It reads every cell of every row, before anything divides them. The schema comes before the split, and a kind is a
declaration rather than something learned, so there are no training rows to read it from yet — and a verdict drawn
from some rows is broken by a cell among the others. It reads them with the schema's own reading, so a kind proposed
is a kind every row binds as. The first kind every value reads as is proposed: true or false in any of its spellings —
but not noughts and ones alone, which are whole numbers, since a gap can be filled among whole numbers and not among
true and false — then whole numbers, then numbers, then moments. Moments are read as ISO 8601 writes them, else by the
one format among the usual ones that reads every cell. Every one of those names a year, a month and a day, so nothing
is filled in from the day the proposal is made; cells that read day first and month first alike are not proposed as
moments at all, but offered as moments with both formats named, for a person to say which. Words are proposed as a
category when there are at most 64 different ones and each stands for twenty rows or more: libraries that make this
call disagree sevenfold on the share, so the rule is this library's own, measured on the samples. Whole numbers that
few are offered as a category and never proposed as one, since nothing in the cells tells a class from a count of
siblings.

A source may say what its columns hold, beside the text of its cells: `IStatesKinds`, asked for and never required. A
database does, and so does a Parquet file, and what either states outranks what the cells look like — a code written in
digits that the database holds as words stays words, where read as a whole number `007` would be `7`. A data frame
handed in states nothing, since its kinds may be the guess of whatever loaded it, and neither does a workbook, whose
cells are typed one by one rather than a column at a time, or a JSON file. Whatever is stated, the text stays what a row
is known by.

What depends on how values are spread — which categories there are, where the extremes lie, what should not be there
— is not in the proposal. That is measured on the training rows, by the steps and the profile that learn it.

### The samples read real files, landed in the repository

`Samples/data` holds three published datasets, fetched once and committed: one with gaps and categories and
no time column at all, one that is a price series with a date, and two years of a bike-sharing scheme, a row a day,
whose answer is how the day's rentals spread over its twenty-four hours. Invented numbers demonstrate the happy path
and nothing else, while these three between them force the design to answer for a column that is three
quarters empty, a boolean dialect, a category that is sometimes absent, a split that cannot be made by
time because there is no time in the file, and a whole divided among its parts that comes back as counts by the day's
own total. The pipelines sample asks the passenger list what each of its columns holds and has a profile say what
should not be there; the networks sample trains a network on each of the three, through both doors; the sample notebook
writes the Titanic pipeline block by block, trains a network in its C# cell on what the blocks hand over, and draws the
report at its last block; and a test runs each as it stands, the notebook in the host an application of your own uses,
while the check of the notebook package as Verso installs it opens that same notebook. Landed rather than downloaded at run time, for the same reason a live source is fetched and
landed: a sample that reaches the network is a sample that behaves differently on the day the network does.

### The pipeline ends at a handover, and the same one serves

`Batch(part)` is where the pipeline stops: rows of numbers, their column names in a fixed order, and the
answers handed over separately when the pipeline names an output — as many numbers a row as the output names,
in the order it names them, seventy for a histogram of weights. An output of one answer hands it over as one
label a row as well. A network built here, a trainer from ML.NET or another established .NET library and a
caller's own learner all take that same handover, which is the only reason two of them can honestly be compared.

Which answers there are is the output's to say, and each kind of output is a verb of its own. `target` names one
column. `target.distribution` names the columns a whole is divided among — a flock weighed in seventy bands of
fifty grams — whose shares are at least nought and sum to one on every row; named with the column saying how many
there were, its shares come back as how many fell in each band, since a served flock knows how many birds it has
and not how they fall. `target.labels` names columns that are each nought or one on every row — one of them when a
row is exactly one of its things, any number when it is not. `target.ahead` makes its answer from a column rows
later in the declared order — the price five days on, or the return on today's price by then — so it is an output
that acts: it makes the answer where it stands when the pipeline is fitted, and a served row, which has no later
rows, awaits nothing from the rows handed in. A return comes back as a price by the row's own price as it was read.
A package adds a kind the way it adds any verb, by implementing `INamesTheAnswer`, or `IMakesTheAnswer` for an
answer the rows do not bring.

An output is made under its verb in one place, `StepCatalog.Make`: the verb's template, every value the output it
replaces holds under a key the verb takes, and on top every value picked, which wins — read as a file's step is, so
the verb's own rules hold. A form that swaps one kind for another and a list that picks a kind and its columns both
make it there, and keep the same values. Placing it is a separate thing that decides nothing about it: an output
takes the place of the one standing, or goes at the end, and a return goes directly after the split, above every step
that changes its column. Whether it is the same output as the one standing is decided by what it writes, not by its
own equality, so a kind from another package that compares a list by reference is still the same when it writes the
same. What each column is to the output is not stored anywhere: it is read off the way back, where the column an
answer comes back to is an answer and every other column the way back reads — how many birds a flock has — scales it.

Four things are refused there rather than passed on. A column still holding words, because turning one into a
number quietly is how a category becomes an order nobody meant. A gap, because a model cannot be handed an
absence, and a value that is not a finite number, which a model learns nothing from and says nothing about. An
answer that no longer exists, which is what encoding the answer column does to a pipeline that also predicts
it. And a row whose answers its output could not have meant: each kind of output says what its answers must
be — a distribution that sums to one, labels that are nought or one — and the handover asks it of every row it
hands over.

A learner also says what it needs of its features, where they are handed to it: `Batch(part, Needs.OneScale)` is a
network's, every feature declared to land between minus one and one. Where each feature lands is followed down the steps with the
columns — a scale's range, a moment's form on its circle, an encoder's noughts and ones, true or false, a row divided
by its size — and a step that writes a column without saying where leaves it landing nowhere said. The handover then
refuses every feature that is not declared to land between minus one and one, all of them at once. It reads the
declaration, not the rows: a feature that happens to lie in range on these rows and is declared to land nowhere
would not on the next ones, and a pipeline loaded from its file, which holds no rows, is held to the same answer as
the one that was fitted. A value outside the range on a later row is what each scale's own choice decides — pass,
clip or refuse — since that is where the range was declared: the training rows land between minus one and one, and a
later row that moves on passes outside unless its scale clips or refuses it.

A pipeline read from its file holds no rows — its file keeps what the steps learned, not the rows they learned it from —
so it has no part to hand over and nothing for the report to measure. Asked for either, it says so, one refusal for
both, naming the run over the rows that has them; a row that arrives later is served through it as through any fit.

A batch also says which part it was handed over from and which row each is, by its key, so what a model predicts for
it is measured against the rows it was made for. And a pipeline can be read in place from a larger file, as the value
of one of its keys — `PreparedData.FromJson(json, catalog, "pipeline")` — read exactly as its own file is, with every
fault placed at its line in the larger file: a trained network and the pipeline it was trained behind are one file.

`Replay` is the same declaration over rows nobody had seen, with the numbers the training rows produced and
nothing fitted again. That is what serving is, and `PreparedData.FromJson` loads both halves back from the
saved file so a host with no data at all can do it — with the catalog of the verbs it may hold, because a
reader that knew only this package's own could not read a file that holds anybody else's. It is also why a
model without its pipeline cannot be used: the numbers reaching it would not be the numbers it was trained on.

A replay walks the steps exactly as the run did, so it puts the rows in the declared order and drops the
warm-up rows too; the same sixty rows used to come out as fifty-six from a run and sixty from a replay.
`Served` hands the result over the way `Batch` does, without an answer — a served row is the question — and
with which of the handed-in rows each served row is, since the replay may have dropped some and reordered the
rest, and a prediction has to find its way back to the row it was made for.

The answers a served row lacks arrive as gaps, one for every answer the output names, and a step that drops rows
does not see them: dropping the rows without an answer is how training rows without one are left out, and a
served row is exactly such a row. Judged by its answer, a pipeline that left out its warm-up refused every row
it was asked about, and one that dropped the rows without an answer served none of them without a word. A fit
still judges every column.

Predictions come back into the units each answer was read in by the steps walked backwards, and the way back
follows a column up them: a step that undoes it is undone, and the way back goes on with the column that step
made it from. A logarithm taken into a new column from a scaled one comes back through the scaling too; undone
alone, it came back in the scaled units, and the run's check could not tell, because it compared with the answer
as read and an answer a step made was never read. A step whose way back needs more than the number reads the row
the number belongs to, as it was read: a share of the birds in a row comes back as a count only by those birds.
So predictions for a part come back with its rows, and predictions for served rows with the rows handed in, each
found again by where it was handed in and checked by its key — rows handed in again in another order would
otherwise be answered with another row's numbers. What is kept as it was read is what a way back names: the column
each answer comes back to, and every column a step on the way reads. Against those values the run checks every
answer's way back before anything is handed over.

### A step says what it does by what it implements

`IPipelineStep` stays narrow — a verb, how to write itself down, and which columns it reads, which it says
through its parameters — because most of what a step might do applies to only some steps. What a step can
*do* is said by the one capability it implements: `IOpensRows` for a source, `IBindsColumns` for a schema,
`IOrdersRows` for an order, `IAddsColumns` for arithmetic on a row, `IDropsRows` and `IDropsColumns` for taking
something away, `ISplitStep` for dividing the rows, `IFittedStep` for learning and replaying — and `IMeetsANeed`
for a step that learns and that some learners do without — and `IProducesEvidence` for proof. An output that only names the answer acts on nothing, because it names the answer
rather than changing the data; `INamesTheAnswer` says it is an output, whichever kind, and a pipeline has one. A
report acts on nothing either, and `INamesTheMeasures` says so: it names what a trained model's answers are measured
by. An
output whose answer the rows do not bring makes it, and `IMakesTheAnswer` is that act: made from the rows when
fitting, a gap in every row when replaying.
What every step has belongs to its type — its name, its purpose, its parameters — and a step without one of
those does not compile.

The run does not ask each step what it can do. Every capability says what doing it means, in terms of what
the walk holds — the rows, the table, the parts, what was learned — and the walk hands each step to itself.
A walk that asked was a list every new capability had to be added to, and a list that had to agree with the
rules about which steps act. A step with two capabilities would be one step the run could not place, and
outside this library it does not compile; a step with none is refused as a step that does nothing.

Widening the step interface instead would force a split step to answer for column effects and a report step
to answer for fitting, which is the interface-segregation complaint in its usual disguise. As capabilities,
a package adds a verb that does something new by implementing one more interface, and nothing existing
changes.

The order the run uses is the order the steps were written in. Everything before the split runs before the
rows are divided, so a feature is what the split divides rather than something added to one part of it; then
the rows are divided; then each step below is fitted on the training rows and replayed over all of them. A run,
a replay and the data shown under one block of a notebook are the same walk, and differ only in whether the
steps that learn are fitted there or replay what was fitted before. The run used to lift every feature above
every dropped row; a pipeline that dropped its warm-up rows between two indicators then gave fifty-one rows
where the steps as written give fifty-six.

### A row is known by what it says

Every row carries who it is: where it stood among the rows as they were read, and a key made from what it says
— every cell with the name of its column, whatever order the columns came in, digested with SHA-256 so the key
is the same on every machine. The place is how one run finds a row again: the way back for an answer, which
handed-in row a served one is. The key is how the same row is known across runs, when the same file arrives in
another order or served rows are handed in again to put their predictions back, and it is what a split ranks
rows by. It identifies a row and nothing more; no value reaches a fit
or a model through it, so the rule that an undeclared column is not carried still holds.

The same digest finds the rows that are there more than once, over two different things. A split ranks rows by the
record they were read from, so every copy of one record lands in the same part. A profile counts the rows that hold the
same where it stands — its columns, as they are there — because a repeated row in training and in test is one a model
meets again after learning it, which reads as skill and is not, and two records that differ only in a column the schema
left out are that to a model. Both are the same digest of cells with their column names, never a string the cells are
joined into, which two different rows can share.

### The data after any step, standing where the split puts it

`ViewAt` gives the data as it stands after any number of steps, and it is what the grid under a notebook's
block shows and what evidence at that place measures. Every row in it says where it stands: in the part the
split puts it in, in the gap it keeps apart, dropped before the split reaches it, or undivided when nothing
divides it. The split is found
wherever it is declared, below the view as much as above it, because a range or a profile drawn above the split
over every row would let the rows a model is measured on shape what it is shown — a view with no split read all
891 Titanic rows as training rows.

So a view above the split walks on down to the split, and it checks what it walks. A step below it that reads a
column the rows lack is refused at its own view and at the run, and never at a view above it: the rows there do
not depend on it. A view is therefore identified by the steps it walks and the place it stands —
`ViewKeyAt` — together with the bytes it was read from, and two views with the same key over the same bytes are
the same view.

What a view's measured rows hold is read by the rule a fit reads them by, never by one of the view's own:
`MeasuredValues` is what a fit of numbers sees, and `MeasuredCategories` the categories an encoder learns — every word
those rows hold, once, a gap none, in one order. The grid colours by them, so it never calls a value known that a fit
there would not know.

### A notebook is one more front end of the declaration

`DeepSharp.Verso.Notebooks` writes a pipeline as a Verso notebook: one block per step, each block the step's
own JSON, and the blocks, in the order they stand, are the steps in the order they run. The notebook is the
declaration; saving it saves the steps, and what a block shows is never saved, because it is worked out from
somebody's own data each time it is asked for. The block type says so to Verso — its outputs are no part of the
document — and the serializer each of Verso's editors saves with leaves them out: the one the engine holds for the
format, or one handed the host's cell types. A serializer made without them cannot tell a block from any other cell,
and keeps what it shows. The steps are read through a catalog as a file's are, and
assembled through the same constructor, afresh at every gesture from the blocks as they are, so a block
inserted, moved, deleted or edited is always seen. The longest run of blocks from the top that makes a
declaration is the pipeline, and a block below it says which block stops it.

The catalog is the package's own and the same wherever it runs: the library's verbs, the indicators, and the readers
of a Parquet file, an Excel workbook and a JSON file, which the package brings with it as it brings the charts, so a
first block reads any of the four files a pipeline file can name. No host adds a verb to it: an extension Verso installs
is loaded apart from every other, so another package's steps are not the notebook's steps even under the same names,
and one vocabulary everywhere keeps a notebook that reads in one host from being refused in another.

A box on the grid changes the declaration, never the data. Every column has a box saying whether it is in, and
every column the schema takes a box saying whether it is a category, and both go through the one set of column
rules every door that changes the columns uses. Unticking a declared column no step names excludes it in the schema,
which keeps its kind — a category the step encoding every category turns into columns among them; unticking one a step
reads, or one a step made, writes a `drop.columns` block below the last step
that reads it and below the step that made it — never higher, even under a schema that keeps the rest of the file,
where any name may be read from the schema down. Ticking either brings it back as it was, and a column the schema
does not name comes in by the rule the list's boxes keep, below, where the source has it — which only the source's own
header says, so a tick made before anything read the source shows the source first and takes nothing in. Ticking a category remembers the kind
the column was, and unticking gives that kind back. A box the rules would not let change is drawn but cannot be
clicked: the answer cannot be left out, nor the last column a schema takes, nor the last category the step encoding
every category has, and a category that does not say what it was keeps its box ticked. What the boxes say is what the grid draws: a column that is not in is black, and its values
are not written into the page at all, so a column left out stays out of sight until its box is ticked again; a
category is one colour, darker for a value the training rows never held, which an encoder fitted on them would not
know. A box carries no payload, so Verso's router sends the state it is in — on a click, on a change and on every key — and
the same state twice does the same thing once; a change the rules refuse, sent by a grid drawn before the blocks
changed, is refused at the block with the rule it breaks, above data that is as it was. A block the notebook
rewrites is written as a new block in the old one's place, and run, because Verso tells a front end nothing about a
block whose text a part changed, and the next keystroke there would put the old text back; under a layout that
cannot add a block the change is refused rather than half made. A gesture that can change the blocks first waits three
tenths of a second and reads them only then: VS Code sends a keystroke a quarter of a second after it lands, nothing
tells the host one is on its way, and a change written before it arrives would lose it. A view or a pick changes
nothing and does not wait.

The schema's block also lists every column of the source as one row — its name, its first values, whether it is in,
and its kind — so nothing left out ever disappears from sight. "Choose the columns" draws it from the rows as the
source reads them; no step runs. A row's box is the grid's box, asked of the same column rules, and a column the
schema does not name comes in by one rule both boxes keep: as the file saved beside the notebook declares it, else as
its cells propose — a timestamp read by the format they are written in — else as text. Where the cells decide, the row
says so beside its kind, with how many different values they hold and what else is offered, so a proposal is never
taken in unseen; the rule is worked out where a box is drawn, from the saved file and the rows that view read, and the
box carries it, since a gesture is handed neither. A list finds its
block by what it carries, never by the block it was drawn on, because a change writes that block anew while the next
key of the same walk still names the old one. It also carries what it was drawn from — the key of the blocks and the
fingerprint of the source's bytes — so a list drawn before either changed is drawn again rather than acted on. A
gesture is handed no file, so the bytes it compares are the ones this session read; in a session that read none, the
list is drawn again and says so. A list the blocks no longer match is cleared, as a grid is. It marks new the source's
columns the saved file never showed and then writes into the file that it showed them, and a change made from it saves
the header it showed.

Each row also has a kind select, and a select is harder than a box. Verso's router sends a select's value on every key
and every change, with no end to a keyboard walk, so a select commits the value it ends on as one pick of that value
from the state its list was drawn in: each value picks again from that state and replaces what the walk committed
before. The session keeps one record — the last change a select made, the steps its walk started from, and the blocks
and bytes that change left — and a send is fresh while its list still says what holds, goes on from that record while
nothing else changed, and is stale otherwise, drawing the list again and changing nothing, an echo included. So a
category walked away from and back to still remembers the kind it was, and what the grid or another select changed in
between is never written over. A redraw ends a walk: the keys after it land nowhere, and the select it draws is a new
one. The options are the kinds the rules let the column take, so no value a walk passes is refused; a column the
schema does not name starts at none, and picking a kind takes it in with that kind.

Above the rows the list says what the model is asked to predict. A select picks the kind of output its boxes make —
every kind the catalog knows that names its answer in a column — and picking commits nothing; a kind no row can take
is drawn disabled, with the words of the rule that stops it. Each row's output box puts its column into the output of
that kind or takes it out, through the same `StepCatalog.Make` a form's swap uses, so the rest of what the output holds
stays; a box of its own takes the output away. A column of an output of many goes in after the nearest column the
output holds before it in the source's order, first when none does, and comes out where it stands, so a tick never
moves a column it does not touch. A tick on a column the schema does not take takes it in first, in the same change.
A box is ticked when an answer comes back to its column — for an answer made from a column, the column it is made
from — and, like the include box, it can be clicked only when the click changes the steps and the rules keep what it
makes. So the answer of an output of one is moved or taken away rather than unticked, and a column an answer is only
made from cannot be unticked either, since the output does not name it. The output is changed only while the blocks
make one pipeline, and an output of another kind than the one picked is not changed from the list.

The output's own values beside its answer — how many rows ahead, whether a return, how many ones a row holds, what the
shares are shares of — are selects under the type select, drawn while the list makes the kind of output that stands.
Nothing is typed: a select offers the form's own choices for the value, or every whole number from the least it may
be, and keeps only those the output, placed as the column rules place one, leaves a pipeline that keeps every rule —
so rows ahead run to the split's gap. A value a file may leave out is offered as not said, and not said is not written.
The value is read into the output by the form's own rule, so the list and the form never differ in what it means; one
neither bounds is set in the output block's form. A select sends every value a keyboard walk passes, so it commits by
the kind select's rule: each value is one pick from the state its list was drawn in, and a send made stale by anything
else draws the list again — which is why a return picked and taken back leaves the output where it stood, rather than
where the return put it.

Seventy bands of a flock are two ticks, not seventy. The list ticks one column at a time, or a range — a pick that
commits nothing — and in a range the first tick says where the range starts and changes nothing, and the second takes
in every column between, in the source's order and whichever end comes first, in one change. A range carries one
kind, picked beside it and nothing inferred from the values: a range taken in offers every kind, text first, as the
source holds it; a range made the answer offers only the kinds the output reads. A column the schema already declares
keeps its kind when taken in, but not when made the answer: bands taken in as text and then made a distribution
become the range's kind in the same change, since the range asks for its answers as that kind and a distribution of
text would otherwise need a kind picked for each band. A kind of output whose answer is many columns is always offered
by the type select, because no single tick can start one; a range the rules refuse says so in their words, and keeps
its start so another end can be ticked. A list drawn again because one of its controls was stale forgets where a
range started, since the start was said on the list as it was: the echo of a range's second tick finds the blocks
changed, and would otherwise bring the start back for the next tick to end. Picks are carried by the controls, not
remembered, so a tick sent before a pick's redraw has arrived acts with the picks its list was drawn with. On a flock
of twenty thousand rows either range is one change well under a second, and seventy single ticks take seventy
changes.

What the blocks decide about their columns is saved beside the notebook, in a file named after it
(`<notebook>.columns.json`): the schema with the columns it excludes and their kinds, the columns dropped, and the
output — the same shape a preset has, written and read through the same door as a pipeline file. It is written after
every change the blocks accept and every run of the whole pipeline, and only while every block is in the pipeline;
showing a block writes nothing. It is written whole under a name of its own for each write and then moved into place,
so nothing ever meets half a file and no two writes share a name, and the same decisions again leave it untouched. A
file that cannot be read is never written over, since it may hold what this notebook cannot see; a file that cannot be
written says so at the block, and the blocks keep the decisions. The source's columns it holds belong to the list of
columns, and a write keeps them.

The columns saved beside a notebook are taken over in two presses, and the first changes nothing. The toolbar's
"Take over the saved columns" reads the file and the columns the source names (a comma-separated file's first line, a
Parquet file's schema, a workbook's first row, every key a JSON file's records use), works the take-over out against
the blocks as they stand, and lists it at the schema's block — at the source's, for blocks without a schema, which the saved one
would follow: every column whose decision would change, from how it stands to how it would stand, a column a step
makes named by that step; the output, the schema's order and what the schema does with the columns it does not name,
when they would change; every saved drop the blocks cannot make, with why, since the file forgets it the next time it
is written; and the source's columns the file never showed. When the blocks already hold everything else, the list
says so, names those drops and the new columns, and offers nothing to apply. Otherwise under the list is one box, and
ticking it applies what the list showed. The box carries it — the file's text as it was read, and the key of the
blocks it was listed for — so nothing is remembered between the two presses and what is applied is what was shown. Ticked over other blocks it is refused, with the words to take over
again; while the blocks make no pipeline it is refused, naming the block that stops them; and a tick that finds the
blocks holding it already, such as the echo of the click, does nothing. A take-over whose blocks would break a rule is
listed with every rule and offers no box, and a file that cannot be read says so at the block. The button is offered
only for a saved notebook whose blocks make one pipeline, with columns saved beside it.

The schema block's form changes the schema alone, through the schema's own operations. A column set "not taken"
stays in the schema, excluded with its kind, and a step below that reads it says so at its own block rather than
being rewritten; picking a kind for a column not taken takes it in with that kind, where the source has it. The grid
changes the pipeline, the form one block: that is the whole difference between them, and both keep a column's kind.

A profile drawn under its block gives the answer to what it finds where that answer is a change to the columns: an
alert whose column hands a model the answer, says again what another says or only names its row carries a box that
leaves it out, and one whose value is how the file writes that nothing is known a box that says so on the schema — each
through the column rules, as the grid's boxes are, drawn unticked and asking only for the answer. An alert answered by
a step names the step and carries no box, since where a step belongs is a person's to say.

Whatever changes the blocks — one gesture, or a change to several at once — goes through one way of writing them,
and it writes only the blocks whose steps changed. The steps both lists start and end with are left alone; between
them a step is matched with the one of the same verb in the same order, so it is written again where it stood and
keeps what the notebook holds about that block, or left alone, view and all, when it did not change. A step added
goes after the block before it, and one taken away is removed. So a change to the schema and to the output leaves
every step between them as it was, rather than writing them all again and clearing what they showed.

What lives between gestures is a session per notebook: which block shows which view, what a gesture asked a
block for, one view kept under its key and the bytes it was read from, the report a report's block drew last, kept
under the steps, the bytes and the predictions it was measured from so another page of its grid measures nothing again —
and one show runs the whole pipeline once at most, for the fit it hands over and the report it measures alike — and the
rows the source opened last, keyed by the read step, the path and a SHA-256 of the bytes, which are read and
hashed on every use. The read step parses the rows from those very bytes, whichever of the four files it reads, so the
rows kept are always the ones the fingerprint names, and a Parquet file's list says what the file states each column
holds. A workbook's fingerprint covers every sheet in it, so a change to a sheet the step does not read opens the file
again and withdraws a fit learned from it, the safe way round. Keeping
the parsed rows took close to half off every show on files of five and eleven megabytes, measured inside
Verso's own engine. The session is held by the block type Verso loaded, the one object every part of the
notebook reaches — the block type's own kernel directly, every other part through the host that loaded it —
and one change on it runs at a time: a gesture, a change in a block's form, a press of any of the toolbar's buttons
each wait for the one before them, in the order they came, since one host hands them over one at a time and another
side by side. A stop gives the notebook back at once: the change it stopped goes on by itself, and what that change
asks for from then on writes nothing — no grid, and nothing handed to C# cells — because a change's turn begins with
the number of stops so far and the mark of the run it belongs to, and what it asks a block for carries its turn. A turn
whose run is already stopped is born stopped and does nothing. What a change asks a block for is taken only by a run of
that block begun under the same number, and is gone once the ask ends, so a stopped change's request is never run by
the block's next run. Every write is let through only in the step that reads that number, and counted until it has
landed: a stop counts the writes let through before it, and whoever stopped the run waits for those, never for one let
through after it — which is how the stop and the writes follow one another by the order of their steps, and never by
when a thread happens to run. What a block's run writes — its card, a grid with what it measured and the record that
the block shows it, the list of the columns, the columns saved beside the notebook, what is handed to C# cells — is let
through whole, once it is worked out, under the ticket of the block's run and the turn of the change that asked, so a
stop never splits one and never waits for a fit; the same holds in Verso's own editors when their Stop marks the
block's run. What Verso's engine itself writes as it begins a cell a stop then refuses — the cell cleared, its count, its
last status cancelled — and what a C# cell a stop left behind still prints are the engine's own, and no stop of
DeepSharp's reaches them. A change still on its way to a block the change before it rewrote or took away is not made, as a host that
looks a block up by its id and finds none makes none. It is never static and never shared with another notebook. A
block run by hand reaches the session outside that line, so what the session knows is one value, and each change makes
the next value from the one there is and puts it in place whole: a reader never sees half a change, and no change is
lost to another made beside it — without a lock, which would keep each piece safe and the fact they make together
unsafe. A grid whose view, or whose boxes, the blocks no longer match is cleared, not worked out again: work runs when
somebody asks for it.

C# cells in the same notebook are handed the pipeline as text — the declaration, with what the whole pipeline's
run learned while it is the run of the steps declared now over the bytes there now — because the notebook's
types and a cell's are loaded apart and are not the same types even when their names are. The key is one no C#
variable can have, so a cell reads it afresh every time rather than the value it saw first. It is taken back
whenever the blocks may no longer make what it holds, and only a gesture or the toolbar's run hands it over
again. What changes the blocks without a gesture — a block added, taken away or moved, a cell turned into another
kind, text typed into a block — reaches the notebook through one rule, as soon as it hears of it: a grid the blocks no
longer make is cleared, and the pipeline is taken back unless they still make it, while a cell turned into another
kind keeps what it shows, since that is its own. A grid is cleared before the notebook forgets it, in the one write the
change's turn lets through, so a grid a stop leaves on the screen is still known, and the next catch-up clears it.
Verso's editors tell a part nothing of such a change, so the notebook hears of it at the next gesture; a host that
changes cells itself tells it at once.

The way back is text too. A report's block names what a trained model is held to, and the notebook trains none, so its
measures come from the C# cell that trains one: the cell hands back the predictions its report measured, under
`deepsharp.predictions`, as `Measures.PredictionsToJson()` writes them — the fit they were made behind, and for each
part the report names the key of each row, what was predicted for it in the units the answers were handed over in, and
what the model said it learned nothing about. "Show the data here" at the report's block measures them with the
pipeline's own measuring, `MeasureAgain`, on the notebook's own run of its blocks, and draws the measures under the grid
through the evidence's view, which shows DeepSharp.Charts' rendering of the report. Only what the model gave crosses;
what it is measured against and the way back to the answers' units are the notebook's run's. With nothing handed back,
the block names its measures and says where they come from, rather than showing nothing.

Predictions are measured only beside the very fit they were made behind, and held to it by the writer that holds a
network's file to its pipeline. The text names that fit as two pipelines are compared: the digest of the one writer's text
with the version of the pipeline file left out, and the version it names. The notebook's own run is passed through the same
writer, and the text is refused unless the two are one fit and the version is one in which every step means what it means
now. Rows' keys alone could not do it: scale a feature or the answer another way in a block after the cell ran, and the
rows, the answer and its numbers stay what they were while the fit does not — a model trained on other numbers, measured
through a way back its predictions were not made behind. So a hand-back from before a block was edited is refused at the
report, saying to run the cell that trains again; the answers' names and the rows' keys are checked beside it.

A notebook of blocks is saved as a `.verso` file, the one format of Verso's that keeps a block a block. Saving one
in any other is refused: Jupyter and Markdown keep a cell's text and lose what kind of cell it is — Jupyter brings a
block back as a raw cell, Markdown as text that fences the block's text in — and `.dib` is read by Verso and never
written. The guard takes every format but Verso's own, so one it does not know is refused a block rather than trusted
with one, and a test writes a block through every format the engine has and reads it back, so a Verso whose other
formats learn to keep a cell's kind is noticed. Opening a file of another format whose cells turn out to be steps is
refused too, in each form that test finds a block coming back as, and as a code cell, which is how a Jupyter file
another program wrote holds one; the same test holds the guard to refusing whatever each format made of the block.
Both refusals are the host's to ask for: Verso's VS Code host and DeepSharp's own server ask every extension before
they write a file and after they read one. Three ways around them are written down rather than trusted. `verso
convert` writes Jupyter without asking any extension, next to the `.verso` file by default. The browser editor `verso
serve` starts asks no extension when it opens a file or saves one, so neither refusal holds there: the Jupyter file
`verso convert` wrote opens in it with every block as a raw cell, and saving it writes a `.verso` file of that name —
over the notebook it came from, whose blocks were measured gone. And a package installed from Verso's Extensions panel
is loaded only after a file of another format has been opened, so the refusal on the way in holds only for an install
at the top of Verso's extensions folder. Tests pin the order Verso opens things in and read the browser editor's own
code for what it asks when it opens and saves, so a Verso that changes either is noticed.

Where the guard is not asked, the way back is DeepSharp's own and asks nothing of Verso. A toolbar button, **Make
blocks of the steps**, is drawn wherever the notebook runs, since every editor of Verso's draws an extension's buttons
and hands each the notebook's cells as they stand; it is on wherever a cell holds a step a format kept the text of, and
it puts a block in each such step's place — a raw cell or a code cell whose whole text reads as a step becomes that
block, and a cell of text is cut at each fence that holds a step, the text between them staying text. It writes the
change as any change to the blocks is written, in the notebook's turn, and takes back what the blocks no longer make;
it runs nothing. Measured in the browser editor: a Jupyter copy `verso convert` made, opened there, made blocks again
and saved, left a `.verso` file holding every block of the notebook it came from, where the same save without the
button left none. The guard and the button read a cell by one rule, kept in one place, so what one takes for a step a
format forgot the other does too. Verso has been asked, in its own tracker, to let its browser editor run the guards
its other host runs ([#113](https://github.com/DataficationSDK/Verso/issues/113)) and to say so before a save writes
over a file of the same name ([#114](https://github.com/DataficationSDK/Verso/issues/114)); the button stays either way,
since a file can reach the notebook as text from anywhere.

The same package runs wherever Verso does, and it is the same code in each: Verso's VS Code extension, the
browser editor `verso serve` starts, and an application that takes Verso's engine as an ordinary dependency,
DeepSharp's own server among them. They
drive a part through the same interfaces and differ in what surrounds it — how the package is loaded, and who
draws the output and passes a click on. What does differ is the runtime, and not the way one would guess: the
VS Code host takes the newest .NET installed, while the browser host stays on .NET 8 for as long as .NET 8 is
there. A package built for .NET 10 alone failed to load in the browser — "System.Runtime, Version=10.0.0.0"
not found — so every package ships a build for .NET 8 and one for .NET 10, and a host takes the one that
matches the runtime it is on. The Extensions panel keeps one install per runtime, in a folder named after it,
and loads the newest one the runtime it is on can run, so two hosts on one machine each find their own. It loads a
package it installed for a notebook that names it among the extensions it needs, and for no other: measured in the
host Verso's VS Code extension starts, the sample notebook named nothing and opened with none of DeepSharp's parts, and
installing the package from the panel into an open notebook writes the notebook's name for it. DeepSharp writes that
name into no notebook itself. Measured, a notebook naming it opens in VS Code with the package loaded whichever way it
was installed; but the browser editor `verso serve` starts asks for consent while it opens a notebook that names a
package it has not installed at the newest version the feeds offer, before any page exists to answer, and never
finishes opening it — so a name DeepSharp wrote would leave that editor unable to open the notebook wherever the
package was loaded from a folder of the person's choosing, or a newer version had come out. Verso names its own format
twice — its writer's name, `verso`, and
the name its extensions are told when its VS Code host saves a notebook, `verso-native` — and every part of the
package that asks takes both as Verso's own: the guard took only the first, and refused every save in VS Code. The
code is one code for both: where .NET 10 had a shorter way to say something, the way both runtimes have is the
one used — a JSON writer's line ending among them, so a pipeline file is written with a line feed on every
machine and runtime — and every suite runs on both runtimes.

What Verso installs is checked the way Verso installs it, because no suite can check it. The Extensions panel installs
the notebook package into a folder of its own, with every package it depends on, and loads it apart from everything
else, so what the package does not carry the notebook does not have; every suite runs the build, where all of it is
found. So before a package leaves a run of the workflows, `tools/verso/check.sh` has Verso's own installer install the
notebook package just made — from those packages and nuget.org, with a download cache of its own — and Verso's own
loader load it, in a host of Verso's engine that holds nothing of DeepSharp, built for .NET 8 as the VS Code
extension's host is, and started as each of Verso's hosts starts: on .NET 8, and on the newest runtime. The files Verso
lays out must be those `tools/verso/installed.<runtime>.txt` names, and the notebook's suite holds that same list to what
the build resolves for the package: the same files, but for a library the runtime carries itself, which the build
leaves to the runtime and Verso lays out all the same. So what the package claims and what Verso installs are one list,
and neither changes without a test failing. In that host the sample notebook is opened through Verso's serializer: its
report block shows the data, its C# cell trains a network on DeepSharp's packages named by their NuGet ids and draws the
loss as Verso draws a picture, and the report block draws the measures handed back. Last, the notebook's own load
context is asked where it finds each assembly the notebook refers to — the folder Verso installed it to, the host for
Verso's abstractions, the runtime for the rest — and every method of DeepSharp's is compiled against what it found. That
is what a missing file cannot hide from: in the session that installed it, Verso finds such a file among what it
downloaded, and the notebook fails only the next time Verso starts. With its charts left out of its dependencies the
package failed there four ways: Verso laid out neither the charts nor the engine, the report block could not draw, the
notebook took the charts the C# cell had downloaded, and two of its methods did not compile. And it is what would see a
dependency at another version than the build's: on .NET 10 Verso lays out the immutable collections at the version a
dependency names, 9.0, and the notebook's context takes them where its build refers to the runtime's 10.0, which every
method still compiles against. The suite's own tests lay the package out from the same list and load it as Verso loads
a package, but in a process that holds every one of its files, where a missing one would be found all the same.

Two packages follow the notebook: `DeepSharp.Verso.Api`, with which an application of your own hosts it, and
`DeepSharp.Verso.Serve`, DeepSharp's own server that shows it in a browser, built on Api. The notebook package
stays the same code in every place, and a host's work lives in the host's package. What two of them would share is
kept in one package beside them rather than written into each, and that package appears with the first type both
sides compile against; none does yet.

The block's kernel answers Verso's question for the faults in a text, but no Verso editor asks it: they ask for
completions and hover texts only. A fault is therefore shown where a block runs, on its card, each at its line.

An application that embeds the engine can get the notebook's parts from Verso's own discovery, which reads every
assembly beside the application that references Verso's abstractions. A program published as a single file has no
assemblies beside it, and discovery then finds none, without a word. `DeepSharp.Verso.Api` therefore registers the parts itself, before discovery runs — a part discovery meets
again is passed over, while one registered after it would be refused as a second — and opens a notebook the way
Verso's own editors do: through the serializer for its format, past the guards that run after reading, with the
cells that are only ever shown rendered drawn. An extension a notebook asks for is refused rather than fetched. One
file is one notebook however many views show it: the file is known by its full path, compared as the file system
compares names, it is opened once however many callers ask at the same moment, and an open that fails is forgotten
and closes what it built. What is done to an open notebook — typing a cell's text, running a cell, a click on a control
a block drew — takes its turn at the host, one at a time and in the order it came, because the engine serves one
caller at a time: two callers at once broke its count of runs, and a change made while another was under way acted
on blocks that were going away. The host's turns are its own and never the notebook's session's, since a change
handed on from inside that one would wait for the one it is inside and never run. Typing, running or opening the
panel of a cell a change before it rewrote or took away is refused, because nothing it meant still stands; a click from
its card is handed on all the same, as Verso's own editors hand every click on, because a page sends clicks from the
card it shows until it draws again — a walk down a select sends several — and the part knows what became of the block
the card was drawn for. What a part answers a click with is what the cell shows next. A cell is added after another
or at the end, of a kind the engine has — Verso's editors' list with each language folded in, code in the blocks' own
language left out, since such a cell is no block — and starts empty, as the engine inserts every kind; it is taken
away, moved past the neighbour it passes, which no other view can have moved the way it can move a place counted by
index, or turned into another kind in one step, keeping its text and losing what it showed. Each goes through the port
the notebook's layout guards, the engine's own, so the dashboard, which lets cells be run and resized and nothing
else, refuses them, and typing and a change of kind ask the same layout whether a cell may be edited. Every version
says which layout the notebook is shown in and what it allows, so a page offers what Verso's editors offer there, and
each change tells the notebook, as typing does. As a person types, the cell's kernel is asked what may come next and
what a word means, as Verso's editors ask it, in the notebook's turn like anything else; a kernel not yet started is
started then, as those editors start it, and never in the background. A new notebook is made as one block, the step
that reads a CSV file — the one place a block starts with text, since Verso's engine and editors never ask a cell type
for it — written whole under a name of its own and moved into its place by the file system in one step that never
replaces a file, so nothing meets half a notebook and a file already there is left as it was. It is a `.verso` file,
the one format that keeps a block, and a name an open notebook holds is refused, even when its file has gone.

A run exists from the moment it is asked, numbered for each open notebook, and a stop names the run it means, so a stop
sent again after that run ended stops no other. A run that never ends is stopped the only way the engine can stop one,
with a fresh kernel — its token cancelled, a C# loop that waits goes on — so stopping clears the notebook's variables,
the pipeline handed to C# cells among them, and the run itself goes on in the background until the application ends.
Verso's C# kernel puts the process's console back as it found it whenever a run ends, so a run left behind that ends
later can take from another notebook's C# run what that run prints at that moment. The kernel started afresh is the one
that runs what runs at the stop, found as the engine finds it when it runs a cell — the kernel of the cell's type, or none
when the type only draws; else the kernel the cell's language names; else none when a renderer claims its type; else the
notebook's default kernel — which also decides whether the run takes the C# turn, so a cell that only draws takes none;
for code a button runs in no cell, it is the code's language, else the default kernel. And only that: a stop between two
cells, before the first, while Run All resets the kernels, or while a cell only draws starts none, so what the kernels
hold stays, save what Run All's own reset cleared — and a stop during that reset ends the run's turn only once the reset
has ended. What runs now is what the engine says began and has not yet said ended, each word taken as the word of the
run whose work it came from: a run left behind that ends while its cell runs again ends nothing of the run under way,
and that run's stop still starts the cell's kernel afresh.
A stop is one step, whoever makes it — Stop, a close, a run that meets a closed notebook — and a run is one value, each
change of which is one exchange: unless the run already ended by itself, the stop takes the run's end, marks the run,
tells the notebook, and only then takes what runs now as the kernel to start afresh, so whatever the engine lets begin
after the stop began before it, and code a button runs is told as what runs before it is let run. The run's own flow is
handed that decision whole and never looks again at what runs, which may have moved on by then: it waits for the writes
the notebook let through before the stop, and for the verbs the run let through before it — clearing, adding, moving,
Run All's reset — and only then starts that kernel afresh. So what the run left behind asks for from then on writes
nothing, the notebook takes its next change at once, and whether a kernel starts afresh never depends on when a thread
happens to run. Typing tells the notebook too, at once, so what was worked out
from a block as it was is taken back before anything else is asked of it. C# runs take their turn across the whole
process, because a C# kernel takes over the process's console while it runs: two at once printed into each other, in two
notebooks and under two separate holders of notebooks alike. A run that waits for that turn is the run under way all the
same: every view is told it waits, and a stop ends the wait, so the turn, when it comes, starts nothing. The toolbar is
every button the engine has, Verso's own and DeepSharp's, each saying whether it can be pressed, and a button on a
cell's toolbar is asked for every cell, as Verso's editors ask it for the cell it is drawn on — asked with no cell
chosen, running a cell and clearing one could never be pressed; a press takes its turn like anything else, and the C#
turn too, since a button may run cells. A press is one run, and the button acts on the notebook through it: stopped, the
cell under way is left behind, no cell the button would still run begins, and nothing else it asks of the notebook is
done — as the Stop of Verso's browser editor ends a Run All between its cells. A file a button hands over goes to
whoever pressed it, unless the press was stopped, and nothing is written beside the notebook. A cell's properties panel is a section from every part
that has one for the cell — DeepSharp's form for a block, Verso's own for how any cell is shown — and a change to a
field is made by the part its section came from. Saving is Verso's own: the serializer for the notebook's format, which
leaves out what a block shows and keeps what a C# cell printed, past the guards that run before writing, and the file
written whole under a name of its own before it takes the old one's place. Saved under another name, the notebook is
that file from then on, and what DeepSharp names after it — the columns saved beside it, an exported pipeline — follows;
the holder of the notebooks saves it so, since only it knows which files are open, and a name another open notebook
holds is refused. Verso's editor itself is not published for applications to reuse, so such an application does what the
editor does: it draws a block's output, which is HTML; it hands a control's `data-action` and `data-extension-id` to the
part the control names, with its `data-payload` — or, for a control that carries none, its state, as Verso's own router
sends it: `true` or `false` for a box, the value it is at for a select — and with the notebook's variables and
operations; it draws again when a block's output is updated or a gesture says it changed the blocks; it gives the
toolbar's buttons, every one of DeepSharp's, a context of its own; and it saves through a serializer that knows
the cell types, so what the blocks show stays out of the file. An application that only wants the pipeline reads each
block with `StepCatalog.ReadStep` and builds the declaration through its constructor, or has the command line write the
file with `verso export --format "Export the pipeline" --extensions <the published package>`. That file holds the steps
and no fit, because nothing ran them there. The passing on, the toolbar's context and the rest are what
`DeepSharp.Verso.Api` is named for, written once beside the notebook rather than by every application: it opens the
notebook, passes a click on, gives the toolbar and the properties panel their context, saves the notebook, and tells
every view of it what changed. What a part may do through that context is the host's to say, by what it asks the part
for: a press acts through its run, and a look — a button asked whether it can be pressed, a panel asked to draw its
section — through operations that refuse every verb, whoever wrote the part, so what is only looked at does nothing to
the notebook. A change — a click on a control a cell drew, a changed field — runs DeepSharp's blocks as its own;
anything else it asks to run is a run of its own, told and stopped as any run is, and once that run is stopped nothing
else the change asks is done, and what the stopped change would answer or hand over is dropped. The drawing stays the
application's own, since only the application knows what it draws with, and it is
handed every cell and what the cell shows as plain values to draw from. Each change makes the notebook's
next version, and a view is told the cells that came or changed, as they now stand, with every id in order whenever
cells came, went or moved; a click that rewrites a block puts a new cell, under a new id, in the old one's place, since
a cell is known by its id and by nothing else. The engine says nothing when a cell is cleared, inserted or taken away,
nor when a form changes a block's text, so the host does not pass on what the engine says: it compares the notebook with
its last version at the end of everything done to it. What the engine does say — a cell began, ended or showed something
— only wakes it to compare, because the engine says it from inside the run, and a view doing its own work there held a
click twice as long. What a C# cell displays while it runs reaches a view before the run ends, gathered for a moment
first, as Verso's browser editor gathers it; what it prints comes whole at the end, as the engine hands it over. Every
version says which run is under way, from its start — the cell that runs or waits for the C# turn, none for a button's
run between its cells, since when, and the number a stop names — so a view can say so and offer to stop it. A view never
holds the notebook up: one that reads slower than the notebook changes is kept one change behind, the latest look at
each cell, and never holds more than the notebook itself. The outputs a run is adding to are copied whole or not at all,
as the engine copies them itself: a copy taken while the list grows its storage fails, and one can hold a place the list
has counted and not yet filled, so a cell caught that way keeps what it showed until the run says more. A file that
repeats a cell's id gives each repeat an id of its own when it opens, as Jupyter's own reader repairs repeated ids,
because nothing else tells such cells apart. Given a grace, the holder of the notebooks closes one by itself once no
view has shown it for that long, nothing runs or waits in it, and nothing in it differs from the file it was last saved
to — as Verso's own comparison of two notebooks finds it, told which cells never save what they show, so a block's view
never counts. The views are the only count: a page that stops reading ends its view, and nothing else says a notebook is
in use. A notebook with changes not yet saved stays open until it is saved, or until the application closes it, which it
can do for one notebook at once; without a grace, notebooks stay open until they are closed, so an application that uses
one without a view is never left holding a closed one. No close waits for a run: a close stops the run under way as a
stop does — one that waits for the C# turn never runs, one that runs is left behind — and waits only for a change under
way, whose runs of code a person wrote are runs the close stops. Whatever else was asked before it and still waits its turn is refused when that
turn comes, since a close discards what has not begun, as it discards what is not saved. Closing every notebook stops
the run in each before it closes any of them, because stopping one run hands the C# turn on, and a run another notebook
still let wait for it would start.

Every version says more than the cells, because Verso's editors show more. It says whether the notebook differs from
its file, as Verso's own comparison finds it, worked out at open too, so a save is told as a version like any other. It
says what became of the kernels: how many times one was started afresh — by a stop, or by Verso's Restart Kernel; Run
All starts them afresh too and says nothing of it, as Verso's editors say nothing — whether one is being started now,
and why the last start failed, until a kernel starts afresh or a cell begins. It says everything the engine runs that no
run owns, each from the engine's word that it began to its word that it ended: a block a change runs, and what a stop
left behind, which a view shows running with nothing to stop and the grace close never closes over. A turn tells every
view what it does while it does it, a click's block included, and it is the notebook's one publisher, so no two versions
are made at once and no view loses a change. Every version carries the toolbar's buttons as they stand then, each asked
through a look at that version, and a button whose part cannot say whether it can be pressed is not pressable and says
why; a cell's panel, what its kernel offers and what a word means are answered beside the turn — during any run, and
through a stop's fresh kernel — for a cell the last version holds; a read of a kernel that a start afresh overlapped, one
under way as the read began or one begun while it read, answers nothing — whether the kernel answered or failed as it
was put away — since what it would say belongs to a kernel no longer there; a kernel's own failure, with no start afresh
between, reaches whoever asked. It says which layout the notebook is shown in, what
the layout allows and whether it has a properties panel: the notebook's own has one, the dashboard and the presentation
have none, and there the panel and its fields are refused, so a form never rewrites a block where editing is refused.
It lists the layouts and the themes the engine has, each switched by its id in the notebook's turn and saved as the
notebook's choice, and names the theme the notebook chose, which a view draws from the theme's own tokens. It carries
what a layout draws of its own — the dashboard's tiles, the presentation's column — as HTML with a slot for each cell it
shows, and none for the notebook's own list; a layout that fails to draw holds up no version and says why. What a person
does to that arrangement — a tile moved, resized or run — goes to the layout's own part as a change: a move runs nothing
and waits for no C# run, a tile's run is a run told and stopped as any, a file the part hands over goes to whoever
acted, and every view is told the cells changed after each act. It says what the notebook says of itself — its title,
its default kernel, when it was made and when it was last saved, the version of its format — and the title is changed
as Verso's editors change it: a change every view is told, unsaved until saved, and the name of what the notebook is
exported as. Each kind a cell can be says whether it is shown rendered once it has run, by the engine's own word, and a
kind named with no language takes the language Verso's editors give it. What a cell shows carries a failure's name and
where it happened, and the stream a text came on.

A notebook opens and saves as Verso's browser editor opens and saves one. Nothing the engine falls back on — a kernel, a
layout, a theme — is written into it, so a file that names none is saved naming none, and a layout the engine lacks is
shown in the notebook's own and kept by its name. What the layouts arranged, such as the dashboard's tiles, and what
the parts keep as settings are handed back at open and taken back at every save and at every look at what is unsaved,
and every save stamps when the notebook was last saved. A new notebook names C# as its kernel, and when it was made.

`DeepSharp.Verso.Serve` is the tool `deepsharp-serve`, run beside a notebook or a folder of them. The notebook it serves
runs code as the person who started it, so the server is shut to everyone else. It listens on this computer alone,
on a port the system picks unless one is named. It answers only a request that carries the token it said when it
started — in the address, or in the cookie its first page sets, named after the port, because a browser keeps a
cookie for a computer whatever its port and two servers would overwrite each other's. For the same reason it makes a
change only for its own page: a browser carries that cookie for a page from any port of this computer and names the
page a request comes from, so a request that changes something must name the server's page when the cookie carried
it — a page on another port closed a notebook and threw away what was typed before this was checked — while a program
that carries the token in the address names none. It answers only under a name
of this computer, because a site can make its own name point here and a browser would then carry the site's page to
the server. And it serves its own page, which it carries, and nothing from the folder it runs in: a server that
served its folder answered a private file lying beside the notebook. It says where it is once, on the address it
really bound, and then nothing, because a C# cell takes the console over while it runs; a port already taken stops
it before it says anything or opens a browser. It is packed as a .NET tool for .NET 8 and .NET 10 — told to pack at
all, since the web SDK otherwise packs nothing and says so only in a warning — and told to keep starting on the newer
runtime once .NET 8 is gone, as Verso's own command-line tool is. Before a package leaves a run of the workflows, the
one just made is installed as a person installs it and started in a folder of its own — its newest build, its build
for .NET 8 on .NET 8, and that build on the newest runtime — and asked what a browser would: its page, the folder's
list, and over a notebook's socket the notebook, a run of its step, and a file beside it asked for as a notebook, which
is refused. A package can lack what the build had while every suite, which runs the build, stays green: when the build
for .NET 8 carried two libraries of its own that the newer runtime has built in, a package without one of them failed on
.NET 8 while its build for .NET 10 served the notebook. A notebook no page has shown for a minute, with
nothing running and nothing unsaved, closes. The notebooks it serves are the one it was started beside, or the files
of its folder that Verso's engine says it reads — asked of the engine, so a format it learns is served without a word
here — and a page names one by its file name alone: any other name, a path among them, is not found. A page holds one
socket to a notebook, as Verso's editor holds one connection a tab: its first frame is the notebook as it stands, with
the kinds a cell can be and the layouts and themes the engine has, then each change, numbered by version. On the same
socket the page asks what a person does — typing, running, a click on a block's control, a cell added, taken away,
moved or turned into another kind, what a kernel offers and what a word means, a toolbar button pressed, a cell's panel
read or one of its fields changed, the layout, the theme or the title changed, an act on what a layout drew, a save and
a close — and each ask is answered to that page alone, by the id it came with, never before the change it made; a Stop
is answered at once, past everything asked before it, since what waits may be waiting for the very run it stops. The
socket is opened only for the server's own page, as a change is made for no other, and a notebook the server does not
serve, or cannot open, is refused on it in words. A cell a change replaced answers the version it is gone from, so the
page knows what to wait for, and the page is told in plain words that the cell is no longer in the notebook; any other
refusal says why. A socket ends when the server stops, rather than holding the stop up, when the page goes away, so the
notebook it showed can close, and when a person closes the notebook without saving it. The notebooks close as soon as
the server is told to stop, each run under way stopped as a close stops it, so a request that waits for a run is
answered rather than waited out; and the tool ends its process once it has stopped, as Verso's own host does, so a
thread a cell left going cannot keep it alive. The toolbar is the engine's, and a file a button hands
over goes to the page that pressed it, in the answer to its press, and is saved there as a download under its own
name — never to another page, and never written beside the notebook. A click that names no payload hands its part an
empty one, as Verso hands every part words. A cell's panel is its parts' sections, and a field's new value goes to its
part in the form Verso's browser editor hands one on — a number as a double, a word as a string, a switch as a bool,
several choices as a list of words, nothing as nothing, and anything else as the JSON it came in — read from the JSON
the page sent it in; each part reads it, so the server holds no rule of its own about what a field may hold. An ask that
lacks what it names is refused, saying what it lacks. A number typed with a fraction reaches a part that counts in whole
numbers as that editor's number does, cut to its whole part; handed on as JSON, the same number put the count back to
its default. Saving writes the notebook to its file. A page adds a cell after another or at the end,
takes one away, moves one past the neighbour it names, turns one into another kind, and asks what a cell's kernel offers
as its text is typed; each is the host's own verb, answered as every other one is. In a folder, a page can make a new
notebook, under the bare name of a `.verso` file: a name that holds a folder on any system, another format or a hidden
name is refused, a name a file already has is a conflict, and a server started beside one notebook makes none, which
the page is told with the list, so it offers the new notebook only where one can be made. That notebook is the one
file the server writes that a person did not save, made whole beside the others and never over one.

The page is one file the tool carries — its markup, its style and its script, with no framework and nothing fetched
from anywhere else. Mermaid 11.17.2 and KaTeX 0.18.9, which Verso's editor fetches from a network, are carried inside it
as their packages publish them, KaTeX's faces written in, and each runs only once a diagram or a formula first shows.
DOMPurify 3.4.16 is carried the same way and runs as the page starts, because every piece of HTML the page places passes
it first — what a cell shows, what a layout draws, a button's icon — and what would run is taken out: a handler on an
element, an address that is script, a script, a frame. Opening a notebook is not running it, and Verso's editor places
such HTML as it was written: an output a notebook file carried ran its handler as the page itself, which holds the
notebook's socket and can run its cells. The tool's third-party notices name all three, with their licences. Its icon is
its own as well, so a browser asks the server for none. The page is put together once as the server starts and named by
a tag, so a browser that has it is told it is unchanged, and asks each time it opens it. It draws a notebook as Verso's
editors draw one: a block's output as the HTML its part wrote, less what would run; a failure with its name and where
it happened; standard error labelled, and no failure; JSON as a tree and CSV as a table; progress as a bar; a Mermaid
diagram, and a Markdown cell's formulas typeset in KaTeX's own faces; a widget in a sandboxed frame of its own, which
reaches nothing of the page and takes its look only from the page that made it, shown as the state it saved; cells
shown rendered stay rendered until their text is opened. It sends what a person does the way Verso's own router means to,
and not the way that router does it: a button on its click, a box or a select on its change, and nothing on a key —
Verso's router sends on the click, the change and every key alike, so a tick went twice and a Tab on a focused button
sent its gesture again (measured: six of twelve clauses failed, and none with the router the page carries). Typing is
sent as it is typed, as Verso's editor sends each change — the whole text, at once, before anything asked after it, so
a click acts on the text as the person left it — and what the notebook tells of a cell while an edit of it is on its way
never writes over what was typed since. Text typed while the page has no connection, or sent on one that went before it
was answered, is kept, its cell marked as not sent yet, and sent once the page has a connection again. A connection
that drops is opened again by itself, and the notebook is drawn onto what the page shows, so the cell being written
keeps its text, its focus and its selection; while it is down the page says it reconnects, and, when the notebook holds
changes not saved, that they are lost if the server has stopped. The address loses the token once the cookie carries
it. The page is tested in a real browser against the real server, a test for each of these promises; its script is not
counted in the coverage, which stays C#'s, and the code scanning reads it as it reads the C#. The server does not save a
notebook under another name: a page names a notebook by its file, and a name changed under the pages that show it would
send their clicks to another notebook.

A cell is chosen as Verso's editor chooses one: by a click or the focus in its text; by one click on what a cell shown
rendered shows, which then shows its text, or on a cell nobody writes; and, as the notebook opens, the first cell not
shown rendered, while a connection opened again keeps what the person chose. A click beside the cells takes the choice
away, and so does Run All pressed on the toolbar, so every cell shown rendered shows its rendering — its keys keep the
choice, as Verso's editor keeps it, but a cell shown rendered leaves it, as it does whenever it runs. A cell that goes,
taken away on this page or in another, hands the choice to the first cell, or to the cell a click put in its place. In
the dashboard and the presentation no cell is drawn chosen, as Verso's editor hands its cells no choice there, so a
cell shown rendered stays shown; back in the notebook's own layout the choice is as it was. The panels are Verso's.
Metadata shows the title, which a person changes there, the default kernel, the file, when the notebook was made and
last saved, and its format. Properties, in a layout that has them, are the chosen cell's: the panel says it reads them
while it reads another cell's — the fields of the cell chosen before go at once, so nothing is changed on a cell no
longer chosen — and says a cell has none when there are none or they cannot be read; each field says what it is under
its label, and is read and handed on as Verso's browser editor reads it, a choice matched in any case, a switch written
as a word, several choices written as one line of words. View lists the layouts and the themes the engine has and
switches them as chosen; the notebook is drawn in its theme once it chose one, and in the page's own look until then,
and nothing of it is kept in the browser. Beside the notebook's name stand its kernel, with how many other languages
its code cells are written in, and how many cells it has; a name past twenty characters is cut short there, and whole
in its tip.

A notebook shown in the dashboard or the presentation is drawn as the engine arranges it: what the layout draws of its
own, and each cell's element placed in the slot that names it, showing only what the cell shows — the dashboard's
tiles, moved by their bar, resized by their corner and run by their button, each handed to the layout's own part; the
presentation's column, with the text of the cells it shows whole. Back in the notebook's own layout, the list returns in
the notebook's order; a layout that fails to draw shows the list, and the page says why.

The page writes a notebook as Verso's editor does. Under the last cell is a button for every kind a cell can be added
as, and the same row opens under the chosen cell, so a cell goes after it; the new cell is chosen with its text open. A
cell's bar changes its kind through a list of the kinds, moves it past the neighbour it is drawn beside, and deletes it
— after asking, which Verso's editor does not, because a delete takes the cell away from every page that shows the
notebook. Each is offered only where the notebook's layout allows it, read from every version, so the dashboard offers
running and nothing else, and a cell's text cannot be written there. The engine's own buttons are drawn as Verso's
editor draws them: a cell's bar and the export menu hold only those that can be pressed there now — clearing what a
cell shows once it shows something, an export the notebook allows — with no menu when nothing can be exported, and no
Run Cell on a bar whose own button runs the cell. A code cell's text folds from a button on its bar to its first lines
that say something, as Verso's editor folds it, the fold kept with the cell, and a click on the lines chooses it.
Outputs set hidden are said to be hidden in their place — "2 outputs hidden" — and a click on that chooses a cell shown
rendered; outputs cut short cut each text and each failure to the lines the cell keeps, five when it keeps no count
above nought, and whatever is drawn as a page to 160 pixels. A cell shown rendered — Markdown, Mermaid, HTML — shows its
text while it has none, has no output, or is chosen, as Verso's editor decides, and leaving its text renders it again,
when something is written there; a kind nobody writes, the parameters form, shows neither its text nor a run line, and
is run once when it is drawn with nothing to show, as Verso's editor runs it, so the form draws itself. The form's
controls name no part, so the page does for them what Verso's own script does: its row for a new parameter opens and
closes in the page alone, and a parameter added, a value set — on Enter, or on its change — and a parameter taken away
go to Verso's parameters part as that script sends them. A cell's text takes the keys Verso's editor takes: Shift+Enter
runs the cell and chooses the one below — a new code cell when it is the last — Ctrl+Enter runs it and stays,
Alt+Enter runs it and inserts a code cell below, Ctrl+Alt+Enter runs every cell, Tab and Shift+Tab indent and outdent
to four-space stops, and Escape leaves the text. As a cell's text is typed, the page asks its kernel what may come next
— on Ctrl+Space, and after a dot or a quote — and lists it under the text; the arrows walk the list, Enter, Tab or a
click takes one in place of the part of it already typed, and Escape closes it. A line under the text says what the
word at the cursor means once the cursor rests there, and goes when the text is left. An answer that comes back after
the text moved on is for text nobody has any more, and is dropped.

Beside the notebook's name, a status says what its kernels do, as Verso's editor says it — idle, running, being started
afresh, or failed to start afresh and why — and Save carries a dot while anything differs from the notebook's file. A
run is offered as Verso's editor offers one: not while a run is under way, since what a cell's run button or Shift+Enter
asked then would only wait behind it and run after its Stop — a cell's run button says another cell is running — and
while one is, the page's one Stop stands where Run All stood, so no second run is pressed there. A run that waits for a
C# run in another notebook says so, on its cell or on the page's own line for a button that runs no cell yet, and Stop
ends the wait; a cell a change runs, or a stop left behind, shows running with nothing to stop. The page says things in
three ways, as Verso's editor keeps them apart: a notice, gone after three seconds and put back to three by a newer one —
"Saved to titanic.verso", "Kernel restarted"; a sentence that stands while what it says holds — a change waiting for
the run under way, a run waiting for another notebook's C# run, a stopped run that goes on in the background until the
tool stops, a connection being opened again; and an error, in a banner of its own with a Dismiss, kept until it is
dismissed or another takes its place, and cleared as a save begins. A close goes past everything the page has waiting,
as a stop does, since what waits may be waiting for the very run the close stops; it asks first only when something
would be lost — what is not saved, typing not sent yet or a change not answered yet among it, or the run under way,
which the question says is stopped — and the page sends nothing for the notebook after it, no change that waited, no
refresh and no socket opened again, since each would open the closed notebook again. A folder is listed, even when it
holds one notebook, with a way to make a new one; a server beside one notebook opens it at once.

## Decisions

### A tensor knows nothing about arithmetic

`Tensor` is a shape and the values that fill it. Everything done to one is done by an `ITensorBackend`.

The alternative — methods on the tensor itself, `a.Add(b)` — reads better for one line and then decides the
architecture: the tensor has to know where its arithmetic happens, so it has to hold a backend, so either
every tensor carries one or there is a shared one somewhere. The second is a global that any code in the
process can reach and replace, and that is not built here.

So the backend is handed to what needs it, and the model stays independent of where its work runs. That seam
exists from the first line rather than being retrofitted, because retrofitting it means touching every
operation that was written without it.

Its operations are the ones a first network and its backward pass need, and the backward pass of each is written
in the same operations — a matrix product and its transpose, a row added to every row and the rows summed, a
mean, a scale by one value, a tensor filled with one value — so working out a gradient never reaches past the
seam. `Add` stays two tensors of one shape: a bias is added by `AddRow`, which says so, and two shapes that differ
by mistake are still refused rather than stretched to fit. .NET's vector primitives hold no matrix product, so the
light engine writes its own, and every total it adds up is kept in double precision: added up in single
precision, ten million tenths came to a mean of 0.1087937.

The layers, the losses and the convolution are written in the seam's operations as well, so the seam grows by what
they need and nothing else: rectified and stepped values, tanh, the sigmoid, exp, log, a square root, softplus, a
division, a row's log-shares, a reshape, and an image's windows unfolded into rows and folded back. There is no rule for
stretching one shape over another: a row multiplied into every row is the row added to a tensor of noughts and then
multiplied, and a statistic over rows is a transpose and a sum. Images are channels last, as Keras keeps them, so a
batch of them needs no permuting between a convolution's unfolding and its matrix product. Softplus is an operation of
its own, worked out in double precision and exact at both ends — .NET's `LogP1` is `Log(1 + x)`, and gave nought for
the softplus of −40, which is 4.2e−18 — because a binary cross-entropy built from rectified values and magnitudes sent
back nought or minus one at a logit of nought, where PyTorch sends back a half. A row's log-shares are shifted by its
largest value, so logits of a thousand are as good as logits of one. The vector primitives do not promise the last
bit — tanh of nought came to −5.96e−8, exp of nought to 1.0000001 — so a value is checked to a millionth, and every rule
by nudging its inputs.

### A tensor never changes

Handing the same tensor to two layers is safe, and a caller who reuses a scratch buffer cannot rewrite a tensor they
already handed over — `Tensor.From` copies the values it is handed. An in-place variant will come when a measurement
shows the copying costs something that matters; until then, the correctness is worth more than the allocation.

This is what that costs, measured on one machine, an AMD Ryzen 9 9950X3D, for a training step as the loop takes it on
one thread: the batch gathered, the forward pass, the loss, the backward pass and Adam's update. Thirty-two Titanic
rows of 14 features, through a dense layer of 16, rectified, into one output with a binary cross-entropy: 19.5 µs a
step on .NET 10 and on .NET 8, and 51 KB allocated. Thirty-two images of 28 by 28 with one channel, through eight
filters of 3 by 3, rectified, flattened, into a dense layer of 10 with a cross-entropy: 11.9 ms a step on .NET 10 and
11.8 ms on .NET 8, and 12.6 MB allocated. The collector paused the first for less than half a per cent of its time and
the second for between 0.7 and 1.4 per cent, so collecting what the copies leave behind is not where a step's time
goes. Nothing is changed in place, and the matrix product stays the plain one — a row at a time, each total kept in
double. These are one machine's numbers, not a promise about another's.

### A tensor's values live where its engine keeps them

`Tensor` stays one sealed type, and where its values live is its `TensorStorage`: this machine's memory for a tensor
made here, or the storage of the engine that made it. An engine that keeps its values somewhere of its own — memory it
allocated outside .NET, a graphics card — derives a storage, hands back what it makes as `Tensor.On(shape, storage)`,
knows its own storage by its type when a tensor comes back to it, and takes any other in where an operation reads it.
The other two ways were weighed and refused: an engine's own tensor type in the signatures, or a tensor generic over its
storage, would each have changed every published member that takes a `Tensor` — a layer's `Compute` among them, so every
network written as code — and a model would have had to name its engine. Behind the one type nothing a model says
changes, and nothing is added to `ITensorBackend`, so an engine written against it before still compiles.

`Values` stays what it was, a span over this machine's memory, for every tensor. For one on an engine's storage the
values are copied out the first time they are read, once — by whichever reader comes first, while any reading at the
same moment waits for that copy — and the copy is kept: a tensor never changes, so it never goes out of date. It is
never a view of the engine's own memory, which the engine could let go of while it was being read. The library reads a
tensor's values more than once — a network's file reads each of its numbers twice, to check it and to write it, and the
light engine reads a matrix's values once for every row it adds a bias to — and each read would otherwise be a copy: in
a two-epoch run on an engine of the tests' own, measured with nothing kept, one slot's values were copied out 22 times.

What the seam refuses is said once as well, in `TensorOperandExtensions`, and every engine asks it first, the light one
included. An engine whose own library would stretch a column over a row, or answer the mean of nothing with something
that is not a number, refuses what the light engine refuses, with the same exception, in the same words.

A test hands an engine written outside the library's internals, every tensor it makes standing on a storage of its own,
to a whole run — the training passes, the report and the file — and to serving: it comes to what the light engine comes
to, and copies no tensor out twice. The cost on the light engine is two more fields on every tensor, sixteen bytes: 1.8
KB more of what the Titanic step above allocates, in the same time.

### Every engine is held to one contract, per operation and per step

The tests of the seam, the layers, the losses, the optimizers, the loop and the network's file are written once, against
what the library publishes and nothing else, each taking the engine it runs on. The core's suite runs them on the light
engine and on an engine of its own that differs from it in the two ways libtorch does: every value it makes lives in
memory it allocated outside .NET, and it adds its totals up in single precision — a product's inner sum one fused
multiply-add after another, a mean's halves pairwise. An engine's own package runs the same tests on that engine: the
libtorch engine's suite runs every one of them on libtorch, on the processor, and on a graphics card where it is built
for one. A test that reaches past what the library publishes, a slot filled by the door the library keeps to itself or
what an optimizer remembers, stays with the light engine.

An engine is held to the light engine operation by operation. A total — a product's inner sum, a column's sum, a mean,
what folding adds onto one place — lies within three roundings of the size of the terms it adds up: three times a
float's unit roundoff, 2⁻²⁴, times the sum of the terms' sizes, since two engines add the same terms in their own order
and not necessarily in double. A value worked out value by value lies within PyTorch's own float tolerance, a relative
1.3e−6 and an absolute 1e−5, and what only moves values moves them exactly. One Titanic training step as the loop takes
it is held the same way: its outputs, its loss and its four gradients within three roundings, its parameters after Adam
within the float tolerance. On the tests' native engine a product came to 2.24 roundings at most and a fold to 2.21; the
step's outputs to 0.42, its gradients to 1.07, its loss to the light engine's to the bit, and its parameters after Adam
to 1.5e−8 apart, a thousandth and a half of the tolerance. On libtorch, on the processor, a product came to 2.24
roundings at most, a column's sum to 0.28, a mean to 0.094 and a fold to 1.21, or 0.97 padded as 'same'; and on this
machine's graphics card, an RTX 5070 Ti with libtorch built for CUDA 12.8, every contract held as well.

The promise is per step, not per run. Two engines that round in another order drift apart over many steps, as any two
float engines do: the networks sample's Titanic run stopped after forty-six epochs and kept the thirty-sixth on both
engines, and ended with weights 0.08 apart, giving the sample's first passenger, a man of twenty-two in third class,
0.1638 on the light engine and 0.1655 on the native one; its epochs' losses differed by a relative 1.4e−8 at the tenth
and 1.1e−3 at the thirtieth. On libtorch the same run went the same way: forty-six epochs, the thirty-sixth kept, the
same four slots 0.08, 0.031, 0.0069 and 0.00066 apart from the light engine's, the passenger at 0.16553865, to the digit
the native engine gives, and 0.815 of the test passengers right on both. On the graphics card it went the same way
again, and ended with its slots 2.4e−7 from the light engine's, the passenger at 0.16384740 and 0.815 right; served on
the card and on the light engine, the network it trained answered the 135 test passengers within 1.2e−7 of itself. On
one engine a run is the same run again, bit for bit — which is why a checkpoint names the engine its run was on.

What a run keeps of an engine's memory is what something holds, and nothing more: the network's slots, what the
optimizer remembers of each, the best epoch's slots when the run is to end holding them, and every checkpoint handed out
and kept — each of which holds, beside its own epoch, the best epoch's slots as they stood when it was taken, since a
run going on from it ends holding those. The native engine counts its storages that still hold memory. After a forced
collection at the tenth and the fiftieth epoch of a Titanic run on it that keeps every fifth checkpoint and drops the
rest, 28 and 140 were alive, exactly those; with no storage letting its memory go, 23,162 and 115,762 were.

On libtorch a storage owns libtorch's tensor. TorchSharp ends every tensor with whatever dispose scope was open on the
thread that made it, and a storage takes its tensor out of any such scope the moment it exists: a scope of the caller's
around a run ends, and the network's slots, what its optimizer remembers, the best epoch and every checkpoint the caller
keeps still read — moved into the caller's scope instead, as a training loop of its own would keep a step's survivors,
they would all have ended with it. Nothing but the collector ends a storage, and since the collector cannot see
libtorch's memory, each storage tells it how many bytes it holds: of five hundred tensors of four megabytes made and
dropped one after another, with no collection asked for, at most eleven were alive at once; told nothing, all five
hundred were. Nothing is let go of sooner, at the end of a step, because what a step made may be held by code of the
caller's: a layer written as code can keep a tensor of its pass, and an optimizer of one's own what it moved.

### A shape is a value with no meaningless default

`default(Shape)` cannot be prevented — a struct can always be brought into existence without a constructor.
So it means something sensible: no axes, which is a single value, which is what a loss is. `Count` is read
rather than stored for that reason alone.

The axis count is accumulated as a `long` and refused above `int.MaxValue`. An `int` product wraps around
silently, and a wrapped count buys a buffer far too small for what is about to be written into it — which
surfaces as memory corruption thousands of operations later, nowhere near the shape that caused it.

### The engine is a choice, and this library is what sits above it

An engine gives you fast arithmetic. It does not give you a way to describe a network in C#, pour real data
into it, drive a training loop, keep checkpoints or show what happened. That is what DeepSharp is, and it is
why an engine sits behind `ITensorBackend` rather than being the thing you program against.

`CpuBackend` is the one that needs no installing: `System.Numerics.Tensors`, the processor's own vector
instructions, nothing native. `TorchBackend`, in `DeepSharp.Backends.TorchSharp`, stands beside it as an equal —
libtorch is a library like any other, and rewriting what it already does well would be the most expensive way to learn
nothing.
`TorchBackend.OnCpu()` runs on the processor and `TorchBackend.OnGpu(0)` on the first graphics card: the device is the
engine's, given when it is made, and making it is the one word there is — no word on the options, no engine a run
reaches for.

What differs between them is what a project has to ship and where the time goes, not what a model has to say. The light
one travels inside an application. The libtorch engine's package brings TorchSharp alone, 3.3 MB, with what TorchSharp
brings — Google.Protobuf, SharpZipLib and SkiaSharp — and the application brings libtorch, for the processor on each
platform it runs on or for a card: libtorch-cpu-win-x64 2.10.0 is 79.9 MB to download and 267.9 MB installed,
libtorch-cpu-linux-x64 128.2 MB and 498.0 MB, libtorch-cpu-osx-arm64 56.9 MB and 245.1 MB; TorchSharp-cpu brings all
three, and a build with TorchSharp-cuda-windows came to 4.2 GB. A package that brought libtorch would bring every
platform's to every application, and a card's gigabytes to one that runs on the processor, so it brings none, and making
the engine where there is none names the packages that bring one. At Titanic's size the light engine is the faster:
measured side by side in a harness that hands every step fresh copies of its rows, a training step took 38.8 µs on it
and 336 µs on libtorch on one thread — 365 on libtorch's sixteen — since libtorch spends longer handing each small
operation over than the operation takes. At a convolution's size libtorch is: the convolution step above took 12.1 ms on
the light engine in that harness, and 3.5 ms on libtorch on one thread and 2.2 on sixteen, the collector told of every
tensor it made and collecting twice a step. The graphics card pays at a larger size still. On this machine's RTX 5070 Ti
a Titanic step took 1.2 ms, each of its small operations handed across to the card and the loss read back, and the
convolution step 1.8 ms, about what libtorch's sixteen threads take on the processor; the same convolution over 256 images
with 32 filters took 345 ms on the light engine, 27.5 on libtorch on the processor and 5.0 on the card. Two thousand of
the smaller convolution's steps on the card left its memory where the first thirty had: libtorch's context and cache
took 283 MiB of it, and the most it held while the two thousand ran was 12 MiB more, as much in the first thousand as in
the second, since each tensor tells the collector its bytes wherever they live. Every one is legitimate, the choice is
the caller's, and it is one line.

The engine never sets how many threads libtorch works with — that is the application's, for the whole process — and
never draws a random number, so libtorch's generator stays as the application left it: every draw a run makes is
DeepSharp's own. Two hundred Titanic steps on libtorch came to the same bits at 1, 4, 16 and 32 threads.

The line is written where the arithmetic is asked for, and nothing keeps it. A run takes its engine from its options,
`new FitOptions(seed) { Backend = TorchBackend.OnGpu(0) }` — the options hold the light one unless they name another,
the one place that says which engine a run takes when none is named — and that engine does the whole run: the training passes, the looks at the
validation rows, and the measures the pipeline's report takes once the last epoch is over, because the report is part of
the run. A trained network serves on the engine its caller hands it, `trained.Predict(rows, engine)`, and on a light one
of its own when none is named. The network holds no engine and its file names none, so a network trained on one engine
is read back and served on another, and a trained network's file is the same whichever engine did its arithmetic;
only a checkpoint names one, the engine its run was on, since a run goes on from it as the same run there alone. An
engine that
one door honours and the next ignores fails without a sound — on the light engine both give the same numbers, so nothing
shows which engine ran them — which is why every door that runs a network takes its engine the same way, and a test
hands each of them an engine that counts what it is asked: a report of two parts over the sine rows asks it for twelve
operations, all of which reach it.

### A gradient is worked out by a backend that records one pass

A network learns by knowing which way to move each of its numbers, and that is worked out backwards from the loss
through every operation the pass ran. The recording is a backend like any other: `RecordingBackend` wraps the backend
the arithmetic runs on, is handed to the pass that wants gradients, writes each operation down as it runs it, and is
dropped once `GradientsOf` has worked back from the loss. Nothing holds one or reaches for one, so there is no mode
switched on somewhere and no recording that outlives its pass: the rule that nothing reaches for a shared instance
holds for gradients too.

What it keeps is a record of one pass's arithmetic, not a second description of the model. It is written as the code
runs and dropped after, so there is no graph beside the model to keep in step with it, and a model written as plain
code is differentiated as it was written.

Every operation on the seam carries its rule for sending a gradient back, written in the seam's own operations, so an
operation added to the seam without one does not compile, and the way back reads no tensor's values: it runs
wherever the arithmetic does. A tensor read twice gets the sum of what both readings send back. The numbers asked
about are known by which tensor they are, never by what they hold, and one the pass never read is refused rather than
given a gradient of nothing. Each rule is checked by nudging every input a little either way and watching the loss, on
the wrapped backend alone, so the recording is never used to check itself.

An engine with its own way of working out gradients, as libtorch has, keeps its values behind the tensor in a storage of
its own, as any engine may, and the recording runs over such an engine as it is: the way back reads no tensor's values,
so it runs on the engine's own tensors. A run on an engine of the tests' own whose every tensor stands on such a
storage, its arithmetic the light engine's, trains to what the light engine trains to; on the tests' native engine every
gradient checked by nudging holds, and a training step comes to the light engine's within the rounding every engine is
held to.

The way back is worked out only towards the tensors asked for, as libtorch records an operation only when one of the
tensors it reads wants a gradient. A tensor leads back to one asked for when it is one, or when the pass made it from
one; a share that could reach none of them is never worked out. Nobody asks how the rows of a batch, the answers a loss
reads or a factor filled in moved the loss, so a training step no longer works out the gradient of its own rows: a
convolution's way back folds nothing onto its images, and a dense layer's takes no product for its rows. What is asked
for comes to the same bits whether or not everything else the pass read is asked for too, which a test holds on every
engine, and a step's gradients, and every slot thirty steps on, came to the same bits as when the way back reached
everything, on the light engine and on the native one. The Titanic step's way back asks its engine for 19 operations
where it asked for 22, and the convolution step's for 25 where it asked for 33, with no fold among them. On the light
engine the Titanic step took 19.5 µs where it took 27 on either runtime and allocates 51 KB where it allocated 54; the
convolution step took 11.9 ms where it took 15.2 on .NET 10, and 11.8 where it took 14.2 on .NET 8, and allocates 12.6 MB
where it allocated 13.7. On the native engine the convolution step took 7.0 ms where it took 9.1, on .NET 10.

Every engine works its gradients out this way, libtorch included, on its merits. libtorch keeps a history of its own,
but only for tensors marked, before the pass runs, as wanting a gradient, and only while a mode is switched on for the
thread that runs it: marking a slot's tensor writes a flag on a tensor the network, its best epoch and its checkpoints
share, and a mode switched on for a thread is a shared instance by another name. One rulebook, each rule checked by
nudging, keeps the gradients the same on every engine. What the recording cost on libtorch, when it was measured before
an engine could keep a storage of its own, came from copying every tensor an operation read into the engine and every
result back out, 256 copies a Titanic step; on a storage of its own, the native engine takes in two tensors a step — the
batch's rows and their answers — and copies one out, the loss.

### Every draw is counted, and none is kept

What a layer starts at, the order an epoch takes its rows in and which values a dropout leaves out are all drawn from
one source, `RandomStream`, and it keeps nothing but its seed. A draw is counted from the seed, what it is for, the
epoch and the step — Philox, the counter-based generator NumPy ships, its first words pinned against NumPy's — so
asking for the same purpose at the same point gives the same numbers whatever was asked before, a run resumed at an
epoch draws what it would have drawn, and a layer added to a network moves no other layer's start. `System.Random` gave
the same numbers on both runtimes when measured, but promises nothing across versions and cannot write its state down,
so it could not resume a run. The purposes are named: `initialise:` and the place a layer stands at, `shuffle`, and
`dropout:` with the layer's path.

### A layer is code, and a network is layers

A layer takes a tensor and a pass and gives a tensor; its forward is the same for every layer, and what a layer does
is its own. A network is a layer that holds layers — written as code, naming each layer it adds, or as a `LayerStack`,
whose layers are named by their place — and every number it holds is a slot with a dotted path, `0.weight`,
`2.running_mean`: its own slots first, then those of the layers it holds, as PyTorch lists a module's state. The layers it
holds are walked the same way, each with its path, so the layer holding any slot is named by the slot's path — in a
network written as code as much as in a stack, whose code keeps to itself only the order it runs them in. A slot
either learns — a `Parameter`, which an optimizer moves — or is measured — a `RunningStatistic`, which only a training
pass moves, so the validation rows never shape what the network keeps. A pass says what it is for — training, with the
stream and the place in the run, or evaluation — so no layer carries a mode somebody forgets to switch. A layer belongs
to one network and a network never holds itself, and a snapshot of every slot brings them all back, the running
statistics included.

### The layers start and work as PyTorch's do

A dense layer's and a convolution's weights start uniform by their fan-in with a slope of √5 and their biases uniform
within one over the square root of it — PyTorch's pair, whole; Glorot's and zeros can be named. A dense layer keeps its
weights as inputs by outputs, the other way round from PyTorch, so its product needs no transpose. Dropout scales what
it keeps, so an evaluation pass lets everything through unchanged, and it refuses to leave out every value, where
PyTorch hands back zeros. Batch and layer normalisation work over the last axis, and batch normalisation's momentum is
PyTorch's — the share a new batch takes — which is the complement of Keras's. A convolution unfolds its image's windows
into rows and multiplies them by its kernel. Every layer's forward pass and its gradients were matched against PyTorch
on the walked rows.

A window pads an image by the border it states, the same rows and columns on every side, or as TensorFlow's and Keras's
`padding='same'` does: along each axis as many places as the stride fits into the image, the rows or columns those need
beyond it split in two, the odd one after. No stated border gives that. Through a window of three at a stride of two
over 28 by 28, TensorFlow pads none before and one after, and a stated border of one gave the same fourteen by fourteen
places with values up to 4.56 away; through a window of two, a stated border gives 27 rows or 29, where 'same' keeps 28.
`Window.BordersOver(rows, columns)` is the one rule for the border on each side, and every engine's unfolding and
folding starts its first patch where it says, so a window padded as 'same' stands at the same places on every engine; the
convolution came within 1.2e−7 of TensorFlow's rule on the light engine and 3.1e−7 on the tests' native one. What a
window covers is one stride, down and across alike, a patch of neighbouring places and every channel of a place at once;
a stride of its own for each axis, a dilated window, channels split into groups and pooling are not built, and a file
that says any of them is refused at the setting or the kind it names.

Described in Keras's words, the normalisations take Keras's settings: `BatchNorm(momentum, epsilon)` takes Keras's
momentum, the share the running statistics keep, and the layer keeps its complement; `LayerNorm(epsilon)` takes Keras's
epsilon, which means what PyTorch's does. The words without settings stay PyTorch's. A network Keras 3 described and
ran — windows padded as 'same', a batch normalisation of momentum 0.9 and epsilon 0.001, a layer normalisation of
epsilon 0.001, a last sigmoid — written here in Keras's words and holding Keras's numbers in Keras's own order answered
as Keras did within 1.8e−7 on every logit and 9e−8 on every chance, on both engines; with PyTorch's epsilons its chances
lay up to 0.0099 away.
What still works as PyTorch's is how a batch normalisation trains: its running variance takes each batch's variance
counted over one row fewer, where Keras counts it over every row. After one training batch of 96 places its running mean
came within 1.9e−9 of Keras's and its running variance lay up to 8.4e−4 above it — that count's difference, to within
1.9e−8.

### A loss says what a network's numbers mean

A network gives numbers, and its loss says how they are read: mean squared error as they are, cross-entropy as shares
through a softmax, binary cross-entropy as chances through a sigmoid. The loss takes the raw numbers, so there is no
softmax layer to forget or to apply twice, and a prediction goes through the same activation — which is why a network's
file names its loss.

A sigmoid layer could still be applied twice, and 0.4.0 applied it: a stack ending in one, compiled with a binary
cross-entropy, answered logits of −20, −1, 1 and 20 with 0.5000, 0.5668, 0.6750 and 0.7311, every one a half or more,
and its training read the sigmoid's outputs as logits. `Compile` now refuses such a stack, naming the layer. A
description in Keras's words ending in `.Sigmoid()` before a binary cross-entropy is Keras's habit — the activation on the
network, and a loss that takes what it gives — so compiling it lifts the sigmoid into the loss: the network ends before
it, and the same logits come out 0.0000, 0.2689, 0.7311 and 1.0000. Only a stack says what its last layer is; a network
written as code writes its own forward pass, which is not run or guessed at here, so it is compiled as it is written. A
file keeps the loss its network was compiled with, so one 0.4.0 wrote with such a pair is read, served and gone on from
as it was written. The loss is always named, never assumed, and each says which answers it could have meant — shares
of a whole, answers between nought and one — so a row it could not have meant is refused, named, before anything is
trained on it.

### Optimizers and schedules as PyTorch writes them

`Sgd`, with momentum, and `Adam` follow PyTorch's formulas in PyTorch's order, matched step by step. Weight decay,
Nesterov momentum, AMSGrad and AdamW are not built: each does nothing by default in PyTorch, and nothing asks for one
yet. An optimizer keeps what it remembers of each parameter by the parameter itself, and writes it by path under
PyTorch's names — `momentum_buffer`; `exp_avg`, `exp_avg_sq` and the steps — so a checkpoint carries it. A
learning-rate schedule is worked out from the epoch in closed form — constant, falling in steps, exponential, a cosine
— and asked once an epoch. None watches the validation loss: the rows a model is chosen on would then shape the weights
it is chosen for.

### The loop is Keras's fit, and so is its judgement

`Fit` takes the training rows, the validation rows and options that start from a seed nobody can leave out; the rest
is Keras's: one epoch and batches of thirty-two unless said, the rows shuffled afresh every epoch and the last batch
kept short — 623 rows are nineteen batches of thirty-two and one of fifteen. Every batch is worked out through a
recorder of its own, and an epoch's loss is its batches' losses weighted by their rows, totalled in double. The
validation rows are looked at once an epoch, by evaluation passes, and never trained on. Early stopping is Keras's to
the letter: the first epoch judged is the best so far, a later one is better when it beats the best by more than the
least fall that counts, and the run stops once it has waited as long as its patience; restoring the best brings back
every slot, the running statistics too. A loss that is not a finite number — a batch's, or an epoch's on the validation
rows — is refused with where it happened, since training or judging by it would learn nothing.

A checkpoint holds what the run needs to go on — every slot, what the optimizer remembers, how far early stopping had
got and the epochs so far — and records what the run went under: the seed, the batch size, the early stopping and the
engine. The seed decides every draw; the batch size decides which rows each step takes, and a dropout's draws are
counted by the step, so it moves them as surely as the seed does; the early stopping decides where the run ends and
which weights it ends holding; the engine decides how every step's totals are rounded, and two engines that add up in
another order drift apart. A checkpoint that recorded only the seed went on under any other batch size or early stopping
without a word, as another run: the Titanic network of the networks sample — seed 20260929, a hundred epochs at most,
early stopping of patience ten that restores the best — runs forty-six epochs and keeps the thirty-sixth; gone on from
its checkpoint after the twentieth, it ran thirty-five epochs and kept the twenty-fifth in batches of sixteen, stopped
after thirty-one with a patience of three, ran all hundred with a patience of thirty, and without early stopping ended
holding the hundredth epoch's weights rather than the best's. On another engine it went on without a word too: gone on
from its file on libtorch or on the tests' native engine, after its first, fifth, tenth or twentieth epoch alike, it
kept the same epoch and ended with weights up to 3.0e−7 from those of the run it was taken of, which the light engine
goes on to the bit — another run, under the same seed, with nothing to tell the two apart. So a run handed another seed,
another batch size, other early stopping or another engine than its checkpoint records is refused, naming each
difference, before anything is put back, as is a checkpoint of a network of other slots or other shapes. The engine is
recorded as it names itself: one that implements `INamesItsVersionAndDevice` with the version of what works its
arithmetic out and the device it works it out on, since another version or another device rounds otherwise too — the
light engine with DeepSharp's own version and the processor, the libtorch engine with the libtorch it runs on, as
TorchSharp states it, and `cpu` or `cuda:0` — and one that names only itself by its name, and is held to that. The
epochs are not among them, since going on to more of them is what a resume is for: resumed, six epochs are three, a
checkpoint and three more, bit for bit, on one engine. A checkpoint 0.4.0 wrote records neither the batch size nor the
early stopping nor the engine, and goes on under whatever it is handed, as it did then; the checkpoints of the run that
goes on record what that run was handed.

### A network is written down as its kinds and its numbers

A network's file holds what it is made of — each layer by the name its kind is registered under, with the settings it
is rebuilt from — every slot's numbers by its path, and its loss by name. Read back through a catalog of the kinds a
reader knows, it is the same network to the last bit: every float is written in the shortest form that reads back to
it, and one that is not a finite number is not written at all. A network written as code is saved and read by the name
it is registered under, as a step from another package is. A kind nobody registered is refused, naming the package it
came from or the nearest kind the catalog knows; a setting missing, of the wrong kind or read by no kind, a slot
missing, extra, of the other kind or of another shape, and a value that is not a finite number are each refused at
their line and column, all at once — the same form of refusal as a pipeline's file. A window's padding is written as the
rows and columns on every side, or as the word `same`, as Keras writes it; the word came with the second version of the
network's part, and a part of the first, which 0.4.0 wrote, holds numbers only. A checkpoint adds what the run needs
to go on: the seed, the batch size and the early stopping the run went under — its patience, its least fall that counts
and whether it restores the best, or null for a run that had none — the engine it was on, by its `name` and, where the
engine names them, its `version` and its `device`, the optimizer with what it remembers, checked by the optimizer itself
as it is read, the schedule, how far early stopping had got, and the epochs so far. The batch size, the early stopping
and the engine came with the second version of the training part; a part of the first, which 0.4.0 wrote, holds none
of them and is read as not saying, and a checkpoint read from one is written again as that version wrote it rather than
with values nobody recorded, while a part of the second that is missing the batch size or the early stopping is refused
at its place. A part that names no engine says nothing of it, and its checkpoint goes on on any. Every object in the
file holds only its own keys, so a misspelt one is refused rather than read as nothing.

Placing every fault at its line and column costs nothing until there is a fault. The reader walks the text once for
where each key, each object and each list stands; a number standing in a list is counted, not kept, since a network's
file holds millions of them, and where one stands is worked out by walking its list only when a fault is placed there,
as are the lines. A tensor's values are not parsed with the rest of the file either: the text is parsed with each of
them standing as an empty list, and its numbers are counted and read from the text where they stand when the tensor is
read. A parsed document keeps a row of twelve bytes for every value, taken from .NET's pool of arrays at the size of the
whole text rounded up to a power of two, and read once, a checkpoint's numbers had cost more than twice its text. Keeping
the place of every number and parsing every value had cost 430 bytes a parameter to read a network of four million
parameters, one dense layer of 2000 by 2000; it now takes 36, the first time as every time after: the text's own copy in
UTF-8, 24, and the numbers themselves, 4. Whether a list is a tensor's values is said by the reader of the parts, which
reads them — a slot's under the parameters or the running statistics, what the optimizer remembers, a slot of the best
epoch's — and a tensor's numbers are read from the text wherever it stands, so the rule decides what the parse costs and
never what is read.

A checkpoint's network and its run are read from one reading of its file, by `NetworkDocument.ReadCheckpoint`, which
hands both back as a `SavedCheckpoint`; `ReadNetwork` and `ReadTraining` each read the whole text, and a checkpoint
read through both had read it twice. A checkpoint of that network under Adam, three numbers a parameter, took 372 bytes
a parameter the first time and 173 after, and 432 and 232 with the best epoch's slots besides; read once, it takes 96
and 128, the first time as after. `CheckpointFile.Read` reads through it, so the file is read whole before it is held to
the pipeline handed over: a file that is wrong itself is refused for that, whichever pipeline it is handed.

### Numbers trained elsewhere go in by path, under the same rule

A model trained somewhere else reaches a network here through an importer, `IImporter`, whose `Read` hands back what
reading a network's own file does: the network, its slots holding the file's numbers, and its loss. An importer of numbers
alone is handed the network they belong to; one of a file that describes the network builds it. Either ends in
`network.Load`, handing each number with the path of the slot it is for and where its file holds it, and the load holds
them to the network as PyTorch's `load_state_dict` does with `strict=True`: every slot once, none the network does not
have, each of its slot's shape and every value a finite number. Whatever is wrong is refused at once, before a slot is
touched, so a network holds all of what a file holds or none of it; and each fault is named where its file holds it — a
tensor's name, a dataset's path — because a file of numbers has no lines and columns to name it by. The reader of a
network's own file hands its slots to the same load, so a missing, extra or misshapen slot is refused in the same words,
whichever file it came from; what that reader checks besides is its own text — how it writes a tensor down, and the key
a slot stands under, parameters or running statistics, which another framework's file does not say.

An import is a network and its loss, and records no fit of a pipeline, so it stands behind one only as any network
does: compiled with an optimizer and its loss and fitted here on the rows the pipeline prepares, for one epoch or for as
many as training it further takes. That fit is what the one file records, and the file is read back beside that fit
alone. The networks sample's Titanic network Keras trained, fitted one epoch further behind the wiki's pipeline, was
written as the one file and read back holding the same numbers to the bit, giving the passengers the same chances.

### PyTorch's numbers are read in the layout each layer says

`DeepSharp.Import.PyTorch` reads a safetensors file PyTorch saved a network's state in, or the file `torch.save` writes
a state dictionary into, into the same network written here, as a stack or as code, handed to it with its loss. The two
readers differ only in the file they open; what is said of the network, the layout of every number and the load are
one `PyTorchFile`'s, so the same numbers land in the same slots, bit for bit, whichever file they came in. PyTorch names its state as the slots here are named, but lays
some of it out otherwise, and how each number is laid out is said by the layer that holds its slot, never read off the
number's shape: a square
matrix turned round is as square as one that is not. A linear layer's weights are kept outputs by inputs there and are
turned round; a convolution's kernel is kept channels out, channels in, rows, columns there, and here the window's rows,
columns and channels in by the channels out; every bias and everything a normalisation keeps are kept alike; and the count
of batches PyTorch keeps beside a batch normalisation's statistics is left out by its name, since nothing here keeps it.
PyTorch lays an image out channel by channel and a network here row by row, so the rows a flatten makes of images are in
another order there, and the numbers the next linear layer — and any normalisation before it — keeps for them are each
put in their place here. That takes knowing what the flatten is handed: an example the network takes, run through it as
zeros, or what the flatten is handed, said outright; told neither, the reader refuses those numbers rather than guess.
A network written as code shows its layers but not the order its forward pass runs them in, so its weights and kernels
are turned as a stack's are, while whether one of its linear layers or normalisations reads rows made of images is read
only as the caller states it for that layer, wherever such rows could reach it, and refused otherwise. A layer of a kind
the reader does not know has none of its numbers read; nor has a linear layer whose rows a reshape lays out as images,
which PyTorch would read channel by channel too.

The file is read by a port of safetensors' own reader, borrowed rather than written, so whatever that reader refuses is
refused here, in its words; a file whose note names another framework's layout is refused too, and one with no note is
read as PyTorch's — the reader is PyTorch's, and safetensors' own writers for PyTorch leave the note out. Numbers held in
16 bits are widened exactly and those in 64 bits rounded to the nearest 32-bit float, as PyTorch rounds them; whole
numbers and truth values are refused. Everything then goes in through `network.Load`, so every fault among the tensors
is named at once by the tensor's name and a network takes all of a file's numbers or none of them. Read so, the Titanic
network PyTorch trained answers a man of 22 in third class within one rounding of PyTorch's chance and every test
passenger within five, written as a stack or as code; with a batch normalisation within ten, since PyTorch folds the
normalisation into one scale and one shift where here a feature is normalised and then scaled; and convolutions
flattened into a linear layer within 3e−8, as a stack and as code. The files and what PyTorch answered are kept with the tests, with the scripts that made them.

### What Keras saved is read in Keras's own words

A model Keras 3 saved — to its own `.keras` archive, or to the HDF5 file it saved one to before — is read by
`DeepSharp.Import.Keras`, whose `KerasFile` is the importer that builds the network its file describes. The description
does not become a second representation of a network: each layer it names is written in Keras's words here and lowered as
any description in those words is, which is why those words take Keras's settings for the normalisations; each activation
a Keras layer carries becomes a word of its own after it; and what comes out is a stack like any other. Keras lays its
numbers out as the slots here keep them — a dense layer's weights inputs by outputs, a convolution's kernel rows by columns
by channels in by channels out — so they go in unturned, a kernel under its slot's shape when its rows and columns are the
window's. A layer's numbers are found by the name the file gives them, never by where they stand: Keras files them under a
group named after the layer's class and how many of that class came before it, while the layer bears a name of its own —
in a network whose two dense layers were built after two others, `layers/dense` held the numbers of `dense_2` — so numbers
matched by place would go into another layer of the same shape without a word.

The loss is the one the model was compiled with, and the network ends where that loss takes over. Keras's habit is a last
sigmoid before a binary cross-entropy trained on the chances it gives, or a last softmax before a categorical one, and each
is lifted into the loss, as the words lift it; a model trained on its logits is read as it stands, and one whose end and
loss disagree — a sigmoid before a loss that takes logits, or a loss trained on chances its last layer never gives — is
refused at that layer. The networks sample's Titanic network, trained in Keras 3 on the pipeline's training rows and saved
both ways, answered the 135 test passengers within five roundings of a single-precision number of Keras's chances — 68 of
them to the bit — and the two passengers the sample serves within two, from either file. A network of every kind the
reader builds — a reshape, windows padded as 'same' and 'valid', a batch normalisation, a layer normalisation, a dropout,
and a softmax its categorical cross-entropy owns — answered within six roundings of Keras's shares, where PyTorch's
epsilons put them up to 3.2e−3 away.

What no network here is built of is refused at the layer that says it, every such layer at once: a stride of its own for
each axis, a dilated window, channels in groups and pooling, as a network's own file refuses them; any other kind of layer
or activation; a layer without a bias, a normalisation over another axis than the last or without its shift or its scale;
and a model of another kind than a Sequential, working in another precision than single, or saved without its loss. The
numbers are then held to the layers the description builds by `network.Load`, each fault at its dataset's path. The file
is read with .NET's own zip and JSON readers and with PureHDF, a managed HDF5 reader. Its settings are read by hand rather
than through the JSON serializer, which a program that trims itself or is compiled ahead of time has no reflection for: a
.NET 10 program run from a single file is such a program, and the serializer refused the first setting it was asked for
there.

### What ONNX holds is lowered onto layers, its layouts as each node declares them

A network exported to ONNX is read by `DeepSharp.Import.Onnx`, whose `OnnxFile` is handed the loss the network answers
through — a graph is a forward pass and names none — and builds the network the graph is. Three writers of such graphs
were measured and are read: PyTorch's `torch.onnx.export`, by its default exporter and by the TorchScript one before it;
Keras's own `model.export(format="onnx")`; and tf2onnx, which is how a TensorFlow SavedModel reaches ONNX. The graph is not
kept: a graph beside the model is what this library leaves out, so the graph is read once, each node written in the words
of the layer it is — a Gemm, or a MatMul with the Add of its bias after it, a dense layer; a Conv a convolution; a
BatchNormalization with ONNX's momentum, which is Keras's, and its epsilon; a Flatten, or the Reshape PyTorch's default
exporter writes a flatten as — and the words lowered as any description is, onto a stack like any other. A Cast into
single-precision numbers, which Keras's export puts round nearly every node, is nothing here. What reads as a stack is a
chain: each node takes the value the one before it made. A node that takes another — a network adding its input back after
a layer is the plainest case — is a branch no stack has, and is refused at that node, as is every node of an operator no
layer here is: pooling, a leaky relu, and the rest.

Layouts are declared, never inferred. A weight matrix is turned round because its Gemm says it is written outputs by
inputs, as PyTorch writes one, and is read as it stands when the Gemm says otherwise, or when a MatMul multiplies by it, as
Keras and tf2onnx write a dense layer — a square matrix would look the same either way. A kernel is written channels out by
channels in by rows by columns because ONNX defines Conv so, and is laid out rows by columns by channels in by channels
out, as a slot here keeps it. An image in ONNX's Conv is channels, rows and columns; here its channels come last, so an
image goes into the network and comes out of it that way. TensorFlow's images have their channels last already, and
tf2onnx says so: a Transpose moving them after the batch before the first convolution, and one moving them last again
before the flatten. That exact pair is read as the layout it declares — the graph takes images as they are here, and its
flatten flattens them place by place — and any other Transpose is refused. An image flattened into a row with its
channels after the batch is flattened channel by channel there and place by place here, and the numbers laid along such a
row — the next dense layer's weights, and a batch normalisation's before it — are turned from the one order to the other,
the image's rows, columns and channels taken from an example of nothing sent through the layers as they were lowered, so
the shape each layer makes is worked out by that layer and nowhere else. A graph that gives such a row as its output is
refused: its values would come out in another order. tf2onnx writes TensorFlow's 'same' out as pads, one side at a time;
pads that are the border `Window.BordersOver` gives the image reaching the convolution are read as 'same', and any others
are refused — at a stride of one before the network is built, since 'same' pads every image alike there, and at a longer
stride once the lowering has found the image.

The exporters were measured rather than assumed. PyTorch 2.10's default exporter folds a batch normalisation after a
convolution into the convolution, writes a flatten as a reshape into rows of the example's length, and kept every number
of 288 bytes or more in a file beside the graph and each of 192 or fewer in it; that file is read from beside the graph's
own file, a name the graph gives that leads outside the graph's folder is refused, and a graph read from another stream
is refused naming the file it needs. The TorchScript exporter before it writes a Flatten,
and hands a number two slots share on through an Identity. Keras 3.15.1, on PyTorch, writes each dense layer as a MatMul
and an Add, casts around each. tf2onnx 1.17.0, converting TensorFlow 2.21.0's SavedModels, writes dense layers as a MatMul
and an Add, or as a Gemm whose weights are inputs by outputs; a batch normalisation as a Mul by a number for each channel,
its shift folded into a neighbouring bias; and, for a SavedModel of batches of any length, a flatten whose target a side
graph of Shape, Gather, Cast, Slice and Concat works out from the batch's length. No layer here is a Mul, and a side graph
off the chain is a branch, so both are refused at their nodes: a network with a batch normalisation does not reach here
through tf2onnx, and one that flattens images does when its SavedModel is exported for a batch of a fixed length, whose
flatten's target is then a number.

The Titanic network trained in PyTorch on the pipeline's training rows, exported both ways, answered the README's passenger
within one rounding of a single-precision number of PyTorch's chance and all 135 test passengers within five, 46 of them to
the bit, and put the same numbers into the same slots from either file; the one Keras trained, exported by Keras or
converted by tf2onnx, answered the 135 within five of Keras's chances, 68 to the bit, and put into the slots the very bits
its `.keras` archive does. A network of two convolutions, two batch normalisations, a flatten and two dense layers gave
each of six images PyTorch's output within 4.5e−8, one of a layer normalisation, a sigmoid and a softmax its cross-entropy
owns gave every share within two roundings, and one of a 'same' and a strided convolution, a flatten and two dense layers
that tf2onnx converted gave each image TensorFlow's output within 2.4e−7. The graph is read with OnnxSharp — ONNX's own
messages, generated for .NET, 118 KB, borrowed rather than written, and no longer changed, which a reader of a format that
grows only by addition can be — and the Google.Protobuf that parses them, by hand rather than through anything that needs
reflection.

### A pickle is a program, so it is read as PyTorch reads weights

The file `torch.save` writes is an archive of storages and one pickle, `data.pkl`, and a pickle is not data: it is a
program whose instructions name functions and call them. Reading one with a pickle library means letting the file say
what gets built, and the protection PyTorch itself settled on is not a library at all — `torch.load(weights_only=True)`
runs its own interpreter, `torch/_weights_only_unpickler.py`, that carries out a short list of instructions and builds
only what a short list of names stands for. This package does the same, and for the same reason: the one part of reading
this format that cannot be borrowed is the refusal.

That goes against the rule that whatever can be taken from outside is taken, so the outside was measured first. The pickle
reader for .NET, Razorvine.Pickle, can be taught PyTorch's names only in a dictionary every reader in the process shares,
and for any name it has not been taught it builds something — a dictionary standing for `os.system`, a real
`System.Decimal` for `decimal.Decimal` — before anything could refuse it. TorchSharp.PyBridge reads `.pt` files on top of
it and brings TorchSharp with it; the one small reader of PyTorch checkpoints is built for .NET 10 alone and keeps any name
it meets. None of them can say no to a name, which is the whole of the job.

So the interpreter carries out PyTorch's instructions and no others, and a name is the one thing it is careful with. What
a name may stand for is fixed in the package — an `OrderedDict`, a tensor rebuilt by `_rebuild_tensor_v2` or
`_rebuild_parameter`, the storage classes of a kind of number — and every other name is refused at the byte that names
it, before another byte is read; nothing is ever looked up by a name a file holds, and a test reads the package's own
metadata to hold it to that. The list is shorter than PyTorch's: PyTorch lets a file build a `torch.Size`, a `Counter`, a
device or bytes, none of which a state dictionary is made of, and here those are refused by name as well. A reference
table of files written wrongly or with hostile intent, each handed to PyTorch's reader when it was made, holds the two
readers together: whatever PyTorch refuses is refused here, and what PyTorch reads and this reader does not is named.

A hostile file can also ask for work instead of code, and there the difference in kind shows. PyTorch rebuilds a tensor
as a view of its storage and copies nothing; a reader for this library gathers each tensor's numbers into a slot, so a
view becomes a copy, and a file of a hundred small views of one large storage would ask for a hundred copies. So a
tensor's numbers are gathered only when a slot of the network takes them, a view that repeats its storage's numbers is
refused, and a dict is keyed only by plain values or a tuple of them, since comparing two keys must not walk a structure a
file could make as deep as it likes. What reading a file costs is then bounded by the file and by the network it is read
into.

The archive around the pickle is read by PyTorch's rules, not by those of the zip reader underneath. A file is an
archive only when it begins as one does, as PyTorch tells it from the format before 1.6, so bytes before an archive, or
an old file with an archive after it, are no archive here either. A record is found by its name whatever the case of its
ASCII letters, as PyTorch finds it, and a name held twice is refused, since which of the two PyTorch would take is its
zip reader's secret. The records read hold together no more bytes than the file itself — torch.save stores each as it
is, so its own files never do — and a record is read to the length it says and checked against the check the archive
keeps of it, so an archive of a megabyte cannot ask for a gigabyte. A dict's keys are hashed by the runtime's own string
hash, seeded afresh in every process, over a text only equal keys share, since the hash the runtime gives a number or a
tuple is one a file can make eighty thousand keys share. And the file is read whole into memory, so one longer than an
array holds is refused by name, before it is read when its stream can say its length.

### What a file says is shown as words

A refusal names what a file holds — a tensor's name, a record, a key — and whoever wrote the file chose those words. So
every message that shows them shows them through one rule, `Quoted()`: a character that would break the message's line,
hide what follows it or turn it round is written as its escape, and a long name is cut short with its length. The rule
lives beside the load of a network's slots, which names what every importer hands it, so the network's own file and every
reader of another framework's refuse in the same words, and none of them lets a file write a line into a log. The network
file's own reader shows a key, a parameter's or a feature's name and a kind it does not know the same way, Keras shows a
model's class, a layer's, an activation's and a loss's name, and ONNX shows a graph's, a node's and a value's name and
where a tensor's bytes are kept beside the graph. `DeepSharp.Pipelines` carries none of the deep-learning core as a
dependency, so a pipeline file's own key, step verb and parameter names are shown through the same rule kept there as its
own internal copy — one rule, read the same way wherever a file's words reach a message, never one project's dependency
on another. A rule kept twice holds only while the two are compared, so a test hands both copies every kind of text the
rule treats apart and holds them to the same answer.

### One model, two vocabularies

There are two ways people already know how to describe a network, and neither is going to convince the other
to change. TensorFlow and Keras speak in stacks: add the layers, compile the model, fit it to the data.
PyTorch speaks in code: write a module, write its forward pass, call it.

DeepSharp offers both, and they are not two libraries. The declarative vocabulary is a **builder that lowers
onto the same model object** the imperative one produces — a `Sequential` is read once and turned into the
layers and the forward pass it describes, and from that point on nothing downstream knows which door it came
through. The training loop, the checkpoints and the charts see one thing.

That is what keeps this honest. A separate graph for the declarative side would mean two engines to keep in
step, and they diverge on the first unusual model — the one somebody builds declaratively and then wants to
reach into. Lowering means there is nothing to keep in step: the builder is a translator that runs once, not
a runtime that runs alongside.

Lowering is `Lower(shape, stream)`: every width is worked out from the shape of one example — a row for a dense layer,
an image of rows, columns and channels for a convolution — and every layer draws its start from the stream by the
place it stands at, the draws a network written as code takes for the same place, so the two doors give the same
network bit for bit, and the same file. A word an example cannot reach — an image handed to a dense layer, a window
larger than the image — is refused naming the word, its place and the shape that reached it. `Compile` takes the
description as it stands and draws nothing: a description whose input is stated is checked there, word by word, and the
network is built at its first fit, from the shape of the rows it learns from and the run's seed, so one recorded number
reproduces the whole run — the start, the shuffles and the dropouts. Keras builds a model whose input is stated at
once, from a random source every model in its process shares, and nothing here shares one; a network wanted before any
rows is lowered by hand and compiled. Before its first fit a compiled description has no network, and says how to have
one now. `Lower` keeps every word, and `Compile` leaves out a last activation the loss applies itself, so a description
lowered by hand to be compiled with such a loss is written without it.

### Where a network meets a pipeline, and the one file they are kept in

`compiled.Fit(prepared, options)` hands the network the training rows the pipeline hands over, every feature between
minus one and one, and judges it by the validation rows; the test rows never reach the loop, and a validation part of no
rows leaves the run unjudged, which early stopping refuses. A row the loss could not have meant is named as it was read.
What comes back is the network behind its pipeline: its history, its measures when the pipeline declares a report, and
what it predicts for rows served later, through the loss's activation and the pipeline's way back — a value or the
chance of one, a count for a share of a whole, a price for a return; never a class label. The measures are taken on the
engine the run was handed, and `trained.Predict(rows, engine)` serves on the engine its caller hands it — a light one of
its own when none is named — both by the one evaluation a network answers rows by, `network.Predict(features, loss,
engine)` in the core, which a compiled network's own `Predict` goes through as well, with the rows turned into what a
network takes by the one conversion its training rows go through, so a served row is answered as the report measured
the network. Neither asks the engine for every row in one pass, however many are handed in: the report asks it a chunk
at a time, as many rows as the run's own batch size, and a served row's chunks are thirty-two, Keras's own default for a
model's `predict`. A row's answer never depends on which other rows share its pass, so chunking moves no answer — on the
light engine a chunked answer comes to the bit what one whole pass over the same rows makes — only how many passes carry
it, which an engine of your own that counts what it is asked can see.

A feature that held one value on every training row taught the network nothing, and the network says so. The encoder
keeps a place for a category the training rows never held and marks a cell that was empty, so a served row always has
somewhere to go. Written one column per category, the place kept is a column named after the encoded one and `other`,
and the mark of a gap one named after it and `was_missing`; a category the training rows hold under either name would
take that column and lose its rows' category without a word, so the fit refuses it and names the declarations that
write no such column. A file fitted before that refusal replays as it was written, since a replay learns nothing. On
Titanic those places — `pclass_other`, `pclass_was_missing`, `sex_other` and `sex_was_missing` — are
nought on all 623 training rows, so no training row ever moved the first layer's weights for them, and after training
they are still the random start's, bit for bit. A passenger written 'Male', of a fourth class or of no sex given is
answered from there, by the seed rather than by anything learned: across six seeds the same passenger was given 0.257 to
0.495 as 'Male' and 0.290 to 0.509 with no sex, against 0.162 to 0.186 as 'male', while test accuracy stayed between
0.785 and 0.815, and no measure could show it, since no row the report measures lands in those places. So what the
network was trained on records each feature that held one value on every training row, with that value;
`Predictions.Unfamiliar` names, for each served row, the features it moves away from it; and the report counts the rows
of each part that do, beside the rows it measured. The answer is left as it is — a pipeline that should answer no
category its training rows never held declares `Unseen.Refuse` — and the pipeline learns nothing of the network: a model
says it of the rows it predicted, as a learner says what it needs of the features it is handed, and the report counts
what it is told. A file 0.4.0 wrote does not record it and is read as not saying, never as saying that nothing is
unfamiliar; the network's part of the file went to its second version with the record, so 0.4.0 names the newer version
rather than a key it does not know.

The network and the pipeline it was trained behind are one file: the network's part, which says what it was trained on —
its features, its answers and the output that named them, the seed, the epoch its numbers come from, each feature that
held one value on every training row, and the SHA-256 of the pipeline's own file as it was fitted — and the pipeline's
file itself, as it writes itself. The same names are not the same fit: fitted again once the file had grown by a few
passengers, all fourteen of Titanic's features kept their names while the numbers the fit learned moved, so a network is
read only beside the very fit it was trained behind, and refused beside any other when it is loaded and when a run goes
on. A checkpoint is the same file with the run's part added, and serves as a trained network too. What the run did and
how the report measured it are output of the run, and stay out of the file, as does the engine that did its arithmetic.
The file is written with a line feed between lines on every system, as the pipeline's own file inside it is; 0.4.0 wrote
the network's part with the line endings of the machine it ran on, and such a file is read as it stands.

The file is one text, and .NET makes no text longer than 1,073,741,791 characters, which sets the largest file there is.
A network's numbers take between 23.6 and 24 characters a parameter, so the one file holds a network of up to about 45
million parameters: one of 43.7 million was written and read back, and one of 45 million refused with the
`OutOfMemoryException` .NET refuses so long a text with. A checkpoint holds, beside every parameter, what Adam remembers
of it, and the best epoch's number when its run keeps it — 76 and 104 characters a parameter — so it holds a network of
about 14 million parameters, or about 10 million: one of ten million keeping the best epoch's was written and read back,
and one of 10.6 million refused. The text is made once, from the writer's bytes with the carriage returns some machines
end a line with left out first, so the largest file is the same on every system; made first and ended with line feeds
afterwards, the text had been a character a line longer on Windows, where a network of 43.7 million parameters could
not be written. The file's top and the pipeline it carries are each read a piece of the file at a time, keeping only
the pipeline's own bytes, so neither copies the network's numbers nor counts their lines to place a fault that is not
there: the one file of a network of four million parameters behind the Titanic pipeline takes 36 bytes a parameter to
read, the first time as after, where it took 501 and 435.

What the network recorded is checked against the pipeline's text as the file carries it, never against that pipeline
written again by whichever DeepSharp reads the file. A pipeline writes the version of the pipeline file into its text,
so the same fit written again by a later DeepSharp is another text, and checked that way every network's file 0.4.0
wrote would have become another fit the day that version went up. The text is taken from the file and written again by
one writer, which writes the digest and checks it: indented by two spaces, a line feed between lines, strings escaped
as System.Text.Json escapes them by default, the keys and the numbers as they stand. That is the layout a pipeline
writes its own file in, so a file DeepSharp wrote comes out byte for byte as it was written, and one another writer
indented, put on one line or escaped otherwise comes out the same; keys in another order, or a number spelled another
way, are another text, since only the writer that chose them could say them back. A network read from its file writes
that text back, not the pipeline written again. The wiki's Titanic network and a checkpoint of it that the published
0.4.0 wrote are kept with the tests, read in each of those ways: they serve the passenger at a logit of −1.7700741 and
go on to the eight epochs 0.4.0 went on to.

Where two pipelines are compared — the one a checkpoint carries, and the one handed over to go on behind — both pass
through that writer with the version each names left out, so a checkpoint taken under one version goes on behind its
fit run again under the next. The version tells a reader what a step's words mean, and the checkpoint's pipeline is
compared rather than read, so the version is left out only between versions in which no step has come to mean anything
else: from the third, which the first network's files carry, to the one this library writes. A test holds every step
this repository ships to that, so a step that comes to mean something else cannot ship before the comparison knows its
old meaning; a checkpoint whose pipeline names a version outside those is refused, as is one whose network was trained
behind another fit than the pipeline it carries.

The writer lives with the pipeline, in `DeepSharp.Pipelines`, as `PipelineText`: the text, the digest a network records,
the digest two pipelines are compared by, and the window of versions they may be compared across. So one rule decides
wherever two pipelines meet — a network's file and the pipeline it carries, a checkpoint and the one it goes on behind,
and predictions handed back as text and the run that measures them — and a learner's package reaches it as a public type,
since packages are taken at versions of their own and nothing internal to one can be counted on by another.

### The learner is declared with the pipeline, and the pipeline still knows no learner

Pipeline-driven design is the course from raw data to a validated model, declared in advance as one replayable
artefact. For four releases the last stretch of that course was not in the artefact: the pipeline was declared, and
then somebody wrote a network beside it and fitted it by hand. Two runs of one declaration could train different
models, and the file said nothing about which. A declaration now names the learner it is written for, and the file
says what was trained behind those steps.

What makes that possible without breaking the first package boundary is that the pipeline names a learner without
knowing one. `INamesTheLearner` is a step that acts on nothing — like an output that only names the answer and a
report that only names its measures — and it says exactly one thing the pipeline itself uses: `Needs`, which the
pipeline already had. Everything else about the learner is written in the step's own parameters, and only the package
that brings the verb reads them. So `DeepSharp.Pipelines` still references no learner, no engine and no network, and
the rule runs one way as it always has.

The step is the last of the declaration's four no-act shapes, and it is held like the others: at most one, below the
output it learns to answer, and it says which columns it leaves behind. That last part is not decoration. A run for a
learner leaves out the steps that learner does without, and it may only leave a step out when nothing below reads what
that step touched; a step that says nothing of its columns may read any of them, so a learner step that kept quiet
would silently make every scale look read from below and nothing could be left out. The run would still be correct and
would simply do more than it had to — which is the kind of cost that never shows up as a failure.

**The model is text in the same words the saved network uses.** A declared network's layers are a list of named parts,
each with the settings that name takes: `dense` with its `units`, `adam` with its `rate`, `binaryCrossEntropy` with
nothing. Those are the names a network's own file already writes, so a declaration and the network it trains speak one
language and nothing is translated between them. The parts are carried by one kind of parameter, `PartsParameter`,
which takes the names it accepts as data from the step that uses it — the pipeline validates a model it cannot
understand, and the vocabulary stays in the package that owns it.

Every setting a name takes is written down, and a file that leaves one out is refused where it is read. A declaration
is replayed from what it says; a number filled in behind the writer's back would make two runs of one file differ as
soon as a default moved. That is also why the file names no vocabulary: TensorFlow's words and PyTorch's lower onto
one model, and where a caller leaves a number unsaid each door writes the number its own library leaves there. Keras's
Adam and PyTorch's differ in their epsilon, so the two doors write different files — which is the honest outcome,
because they are different runs. A `vocabulary` key would say which library was used to type the model, and nothing
would read it.

**The engine is a name, and the application says what it stands for.** A declaration that carried an engine object
could not be a file; one that carried a package name would tie a model to a machine. So the step names an engine —
`light` unless said — and an application hands that name an engine before the run, through `Engines.Use`. The same
declared model trains on the light engine where nothing is installed and on libtorch where libtorch is there, and a
name nothing was given is refused by name rather than falling back to something that would quietly give other numbers.

**Training is the learner package's, not the pipeline's.** `Run()` answers what it always answered: the prepared rows.
`pipeline.Train()` lives in `DeepSharp.Learners.Networks`, runs the pipeline for the learner the declaration names,
and trains it — the same `TrainedNetwork` the hand-written door gives, number for number, which a test holds both
doors to. The pipeline file gains a step and nothing else: a model saved by an earlier release still loads, because a
network is held to its pipeline by the digest of the text that pipeline carries, and that text is kept verbatim.

### Charts come from the training loop, not from the caller

The metrics the training loop already keeps are what the charts are drawn from — the caller never assembles arrays
to plot. `DeepSharp.Charts` draws a run's loss curve and the learning rate every epoch took from its history; the
measures as bars beside the average, each confusion matrix as a heatmap of counts, what was predicted against what was
there and what was left over from a report's measures; and a correlation from the rows it was drawn from — each as the
text of an SVG. The drawing itself is **MatPlotLibNet**, in a package of its own, so a trainer on a headless machine
does not carry a renderer, and nothing of MatPlotLibNet shows in what the package hands out, so the drawing library can
change without a line of anybody's code changing with it.

## What is borrowed on purpose

Anything available from outside is something nobody here has to write or maintain, and that is the default
answer for every well-solved problem: DataFrames, file formats, compression, the test runner, the vector
maths. A hand-rolled version of any of those costs forever and buys nothing.

libtorch is on that list too. What this repository builds is the part nobody else provides: the C# shape of
a model, the path data takes into it, the loop that trains it, and the picture at the end.

The one rule about a dependency is where it lands. Anything heavy gets its own package, so a project that
does not want it never carries it, and the core stays light enough to travel inside an application. A model
is written against the seam and cannot tell which engine is underneath.

## What is not borrowed

Tuning earned elsewhere does not come here. A speed-up measured on one machine, one family of models and
one kind of data is a guess with a good reputation until it is measured again in this repository, on this
code — and by then it is cheaper to find it than to have carried it.

A published binding is the opposite case. TorchSharp is a dependency: versioned, readable, replaceable, and
tested by people who are not us. `DeepSharp.Backends.TorchSharp` is written against its published package and nothing
else: that is borrowing a library, not borrowing a result.

## What is deliberately absent

- **A global backend, context or session.** Nothing reaches for a shared instance. The notebook keeps a
  session, and it is the notebook's own: held by the block type Verso loaded for that notebook, handed to
  every part through the host that loaded it, never static and never shared. One thing is as wide as the process,
  because what it guards already is: the turn C# runs take in `DeepSharp.Verso.Api`. Verso's C# kernel takes over the
  process's console while it runs, so a turn held any narrower let two runs print into each other.
- **A graph that is not a model.** The declarative front door produces the same object the imperative
  one does. Two representations of one network means two engines to keep in step, and they diverge on the
  first unusual model.
- **A random source the process shares.** Every draw is counted from a seed that is handed in, which is why a network
  described in Keras's words is built once the run's seed is known.
- **A schedule that watches the validation loss**, and weight decay, Nesterov momentum, AMSGrad and AdamW. The first
  would let the rows a model is chosen on shape its weights; each of the others does nothing by default in PyTorch, and
  nothing asks for one yet.
- **A class label from a prediction.** A network answers a chance or a share; where the line between classes lies is
  a decision about the use. The measures that count classes read a chance of a half or more as the class, as
  scikit-learn's do.
- **A required engine.** No backend is mandatory. The core carries the light one; anything heavier is its
  own package, and a model cannot tell which is underneath it.
