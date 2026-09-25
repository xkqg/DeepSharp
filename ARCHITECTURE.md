# Architecture

This file records the decisions that shape DeepSharp, and why each one was taken. A decision that is not
written down here is not a decision, it is a habit.

## The layout

```
Src/DeepSharp/Tensors/            the engine side: Shape, Tensor, ITensorBackend, CpuBackend
Src/DeepSharp.Pipelines/          the data side
    IPipelineStep.cs                what a step is: a verb, how it writes itself, what it reads, what it does
    StepParameters.cs ParameterKinds.cs
                                    a step's parameters, said once; the closed set of kinds they hold
    StepCatalog.cs                  the verbs a file may hold, and the one door a step is read through
    PipelineDeclaration.cs DeclarationRules.cs
                                    the steps in order, the rules every declaration keeps, the keys of its prefixes
    PipelineDocument.cs PipelineFileException.cs PipelineFileSchema.cs VerbReference.cs
                                    the file: its envelope, every fault at its line and column, the schema, the reference
    PipelinePreset.cs               what a pipeline decided about its columns, saved on its own
    ColumnState.cs                  which columns there are at each step, followed from the schema down
    ColumnChoices.cs                what can be done to a column, and how each stands
    Steps.cs Schema.cs Splits.cs RowOrder.cs
                                    reading, declaring, putting in order, dividing
    Features.cs TimeParts.cs Maths.cs
                                    what is worked out from a single row
    FillStrategy.cs FillNaN.cs Transforms.cs Outliers.cs DropColumns.cs DropWarmUp.cs
                                    what learns, and what takes rows or columns away
    Evidence.cs                     the profile and the correlation a run is declared to produce
    RowSource.cs SourceFolder.cs Binding.cs Table.cs RowIdentity.cs TrainingValues.cs
                                    rows, where a path is read from, typed columns, who a row is, what a fit sees
    Walk.cs Execution.cs Views.cs Fitting.cs Handover.cs
                                    the one walk every run is, the data after any step, what a fit learned, the handover
    Outputs.cs WayBack.cs           what a model is asked to predict, and how its answers come back into their units
Src/DeepSharp.Pipelines.DataFrame/   a reader through MatPlotLibNet.DataFrame
Src/DeepSharp.Pipelines.Indicators/  indicators over a series, as verbs
Src/DeepSharp.Verso.Notebooks/       a pipeline written as a notebook in Verso
Samples/                          runnable programs and the published data they read
Tst/DeepSharp/                    the tests of the libraries
Tst/DeepSharp.Verso.Notebooks/    the notebook's tests, run inside Verso's own engine
```

The tensor library and the pipeline library do not reference each other, and a test reads their assembly
references to keep it that way.

The notebook's tests are a suite of their own because Verso's engine and the validator the core's tests hold
the pipeline schema to each need a different version of the C# compiler, and one test program can load only
one. The coverage check and the release both run every suite they find, by the name every suite has, on every
runtime the suite is built for. The check measures on the newest and runs the others: the two builds of one
assembly, measured together, merge as one module and most of its branches lose their counts, so every class
read as fully covered. The code is one code on both runtimes, so one measurement covers it.

## The design language: PDD, pipeline-driven design

A design is named after whatever drives it, and here that is **the pipeline**.

Every piece of work follows one sequence: collect the data, add the features, normalise, deal with the
missing values, split into train, validation and test, build the model, and check it against data it has
never seen. The sequence is not the interesting part — everyone does those steps. What matters is that the
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

`Pdd.Create().ReadCsv(path)`, `.ReadParquet(path)`, `.ReadExcel(path)`, `.ReadDb(connection, sql)`,
`.ReadBinance(symbol, interval, from, to)` — one verb per source, and every one of them an **extension
method defined in the package that brings the dependency**.

The obvious alternative is a method per format on the pipeline type itself. It reads the same and costs
the whole architecture: the core would have to reference Parquet.Net, ExcelDataReader and an HTTP client,
every project would carry all of them, and adding a format would mean editing the core.

As extensions, a verb exists exactly when its package is referenced. Reference `DeepSharp.Pipelines.Parquet`
and `.ReadParquet` appears; do not, and it is not in the list. Nothing is carried that is not asked for,
and a new format is a new package rather than a change here.

The escape hatch is one method wide: `.Read(IRowSource)` takes rows and a declared schema, so a format
nobody shipped is still a few lines away rather than a fork.

### Five readers ship, and a funnel carries the rest

Reading data is a solved problem with a long tail, and reproducing that tail is a library in itself —
pandas exposes nineteen readers. What ships here is the short head, chosen by where data actually arrives
and by costing a thin adapter rather than an implementation:

- **CSV** and **SQL** cost nothing at all: the DataFrame already loads both, the second through whichever
  ADO.NET provider the caller brings.
- **Parquet** is where data of any size lives, **Excel** is how data arrives from people rather than
  systems, and **JSON** is what an API hands back. Each is an existing .NET library plus a few lines.
- **Live sources** are their own family, fetched and landed rather than read during training.

Everything else stays one `IRowSource` implementation away — rows plus a declared schema, which is the one
door the DataFrame opens for anything enumerable. Not shipping a reader is not the same as refusing a
format, and that distinction is what keeps the list short.

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

Two kinds are built, both about the data before anything learns from it: a profile of the columns, which
names for every problem it finds the step that answers it, and the rows a correlation is drawn from. Each is
measured on the rows the split trains on — the split below it as much as one above — because a profile over
every row lets the rows a model will be measured on shape what it is shown. What they produce is output: it
is kept with the run, in `PreparedData.Evidence`, and never written into the pipeline's file, since the file
is what is replayed and a replay learns nothing. The measures of a trained model join them when there is a
model to measure.

The drawing lives in an optional package that draws with **MatPlotLibNet** — today the notebook's, which draws
the correlation as a heatmap; the pipeline holds only the declaration and the numbers.

### The rule that gives it meaning

**Anything that learns from the data is fitted on the training split alone, and then replayed unchanged.**
A mean and a standard deviation, the value that fills a gap, the categories an encoder knows, the bounds of
a clip — each is a parameter, each is learned once, and each is applied identically to validation, to test,
and to data arriving in production long afterwards.

Break it and nothing goes red. A mean computed over the whole set gives a model that scores beautifully in
validation and disappoints the day it meets real data, because the validation rows had already been allowed
to influence what the model saw. The same failure wears a second costume at serving time: a feature computed
one way while training and another way live, which is the same leak running backwards.

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
.FillMissing("trades", With.Mean)      // fitted on train; With.Median, With.Zero,
.FillMissing("volume", With.Previous)  // With.Constant(0), With.Previous, With.Refuse
.FillNaN("range", With.Refuse)         // the default: a NaN stops the run, because it should not be there
```

Dropping the row is not among the strategies, though it was once meant to be. A strategy stands below the split,
and dropping rows there would quietly change how many each part holds — the shares the split promised. So it is
a verb of its own, `DropGaps`, that stands above the split beside `DropWarmUp`: the rows are settled first and
divided afterwards. Leaving a column out is a verb too, `Drop`, anywhere below the schema.

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

### The declaration is a file, and the chain can write it

The pipeline is saved as one file with two blocks, because they have different authors. The
**declaration** — the steps in order with their arguments — is written by a person. The **fitted** part —
the means, the fill values, the category lists, the boundary the split landed on — is written by the fit.
Keeping them apart is what allows the same declaration to be re-fitted on fresh data, two runs to be
compared by diffing the declaration alone, and a serving process to load the file without ever knowing the
builder existed.

```
{
  "version": 2,
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
order they came in: what the fit saw travels with what it learned.

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
writes of it — how it stands, its kind, the kind a category was, whether the source may lack it. What a column offers
and what it is to the output follow from those, so they are not listed for themselves; the output is listed once,
before and after, the schema's order when the columns both schemas name stand in another, and what the schema does
with the columns it does not name when that changes. A column a step makes is never said to be missing from the
source. The source's columns the preset never showed are named new. A drop the preset saved can only be made of a
column that reaches the end of the pipeline: one that nothing makes any more, that the schema leaves out, or that a
step below takes away cannot be dropped, and it is listed as not made, with how it stands, rather than refused —
refusing would hold back every other decision the preset saved, and a notebook's saved columns are written again from
the blocks at their next change anyway. A take-over whose steps would break a rule is refused with every fault and
applies nothing, and the same pipeline, preset and header give the same answer whoever asks: a notebook's list, its
Apply, and code. A source's header is read on its own, by the same reading as the whole file, so a large file costs
its first line. A chain written in code takes a preset over in two places, because it is written in order: the
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
handed one error at a time, five times over, stops using the thing.

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
whole, since it may hold words this one does not know.

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
steps say, so neither can fall behind in silence.

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

Standard, min-max, max-abs, robust, quantile and power each learn something different — a mean and a
spread, two extremes, a magnitude, a median and its quartiles, a whole distribution, a shaping parameter —
which is why the kind is named in the declaration and what it learned is stored apart from it. Standard and
min-max are both moved by a single extreme value, so on prices and volumes the robust and quantile forms
are the ones that describe the data rather than the spike.

Row-wise normalisation is a different verb, not a member of this family: it works across a row, learns
nothing, and is fitted nowhere.

Two decisions are made here rather than discovered later. What happens outside the learned range while the
model is running — clip, pass through, or refuse — because a price meets a new high and min-max has no
answer of its own. And whether the answer is normalised, because if it is, the way back is part of the
saved pipeline; without it every error is reported in normalised units and every model looks excellent.

### The pipeline ends at the data, and the learner is a plug

Everything up to and including normalising is the same whatever is going to learn from the result, so that
is where the pipeline stops: a prepared, split dataset plus the declaration of what the run has to prove.
What learns from it is chosen at that seam — a network built here, a trainer from an established .NET
machine-learning library, or something a caller wrote — and each plugs in at the same point.

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

And the preparation happens once, here. A learner that brings its own normalisers does not get to use them:
running them again would leave the saved pipeline describing something other than what the model was
actually fed.

### That seam is the first package boundary

The pipeline and the things that learn are separate packages, and the reference only runs one way: a
learner package knows the pipeline, the pipeline knows no learner. It is the first cut in the library
because it is the one that decides what a project has to carry — a service that prepares data and hands it
to an established .NET trainer should never drag a tensor engine along, and a network that trains on data
somebody else prepared should not drag a CSV reader.

The contract between them lives on the pipeline side: prepared splits, the declared evidence, and what a
learner says it needs. Everything else about a learner is its own package's business.

A cut like this is not kept by intention, so it is pinned: a test reads the assembly references and fails
the moment the pipeline acquires one it is not allowed to have. Direction is easy to state and easy to
break, and by the time it is broken the fix is no longer a line.

### A package is named after the role it plays, never after the vendor it brings

The segment after `DeepSharp.` says which side of a seam a package sits on; the segment after that says
which outside thing it carries.

```
DeepSharp                   tensors, the backend seam, the light engine, the model, the training loop
DeepSharp.Pipelines               the pipeline, ending at prepared splits and the declared evidence
DeepSharp.Pipelines.<Format>      a reader: Parquet, Excel, Json
DeepSharp.Pipelines.<Source>      a live source: it fetches and lands, it does not read during training
DeepSharp.Learners.<Name>   something that learns from prepared data, behind the learner seam
DeepSharp.Backends.<Name>   an engine behind ITensorBackend
DeepSharp.Import.<Name>     reading weights or a model trained somewhere else
DeepSharp.Charts            drawing, from the metrics the loop already keeps
DeepSharp.<Host>.<Part>     a front end inside a host: DeepSharp.Verso.Notebooks
```

Naming by role rather than by vendor is not tidiness. A package called after a framework implies that the
framework is *in* there as itself, and for one of them that would quietly contradict the design: the
declarative vocabulary — stack the layers, compile, fit — is a **way of speaking**, not an engine. It lives
in the core and lowers onto the same model the imperative door produces, so there is nothing to put in a
package named after it. What deserves a package of its own is reading what that framework *saved*, and
`DeepSharp.Import.<Name>` says exactly that and nothing more.

The same test applies to the other two. An engine belongs under `Backends` because a model cannot tell
which one is underneath; a trainer from an established .NET library belongs under `Learners` because it
sits beside the model rather than below it. Both distinctions disappear the moment a package is named
after the logo instead.

A front end is the one exception, by decision: it is named after its host first and after what it is there
second — `DeepSharp.Verso.Notebooks`. It carries nothing of the host inside it, it plugs into it, and a person
looks for it by the host's name, in the host's own list of extensions. Another part for the same host takes the
same prefix. The package is not Verso's; it is DeepSharp's way into it.

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
| a learner | what learns from prepared data | a network here, a trainer from elsewhere |
| an importer | what does a model trained elsewhere look like here | a saved-model or weight-file reader |

A package that implements two of them is doing two jobs and should be two packages; a package that
implements none is a convenience and belongs in whatever it is convenient for. A front end is the one
deliberate exception: the notebook implements none of the four, because it is not a part of the pipeline but a
way of writing one and looking at it, and it sits beside the seams the way the charts do.

A seam is only real when two implementations of it differ in kind, so each one is held to that: the backend
has a pure-managed engine beside a native one, rows arrive from a file and from a database reader, a
learner is a network or a decision tree, an importer reads one foreign format or another. An interface
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
column nobody checked. A warning on a feature that predicts the answer perfectly is worth having as well,
but it is the second line of defence; the first is that unnamed means absent.

A column can also be named and left out. `excluded` keeps it in the schema with its kind while nothing reads it,
the source is not asked for it and the rest of the file does not reach it, so bringing it back is taking a word
away rather than remembering what the column was. A category says which kind it `was` for the same reason:
without it, making a column a category overwrites the only record of what it held before. Leaving a column out
this way is a decision about the columns, so it lives where the columns are declared; and since neither word is
written where it says nothing, a schema that uses neither is the same bytes, and the same key, as before.

What can be done to a column is said once, for every door that changes the columns. Taking one in, leaving it
out and changing its kind each hand back the steps they would make, and never judge them: the rules every
declaration keeps do that, so what is offered and what is allowed cannot drift apart. A column no step reads is
left out in the schema; one a step reads, or a step made, is dropped after the last step that reads it, since
excluding it from the schema would leave that step reading nothing. How a column stands is said without where a
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

### The samples read real files, landed in the repository

`Samples/data` holds two published datasets, fetched once and committed: one with gaps and categories and
no time column at all, one that is a price series with a date. Invented numbers demonstrate the happy path
and nothing else, while these two between them force the design to answer for a column that is three
quarters empty, a boolean dialect, a category that is sometimes absent, and a split that cannot be made by
time because there is no time in the file. Landed rather than downloaded at run time, for the same reason
a live source is fetched and landed: a sample that reaches the network is a sample that behaves
differently on the day the network does.

### The pipeline ends at a handover, and the same one serves

`Batch(part)` is where the pipeline stops: rows of numbers, their column names in a fixed order, and the
answers handed over separately when the pipeline names an output — as many numbers a row as the output names,
in the order it names them, seventy for a histogram of weights. An output of one answer hands it over as one
label a row as well. A network built here, a trainer from an established .NET library and a caller's own
learner all take that same handover, which is the only reason two of them can honestly be compared.

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
something away, `ISplitStep` for dividing the rows, `IFittedStep` for learning and replaying, and
`IProducesEvidence` for proof. An output that only names the answer acts on nothing, because it names the answer
rather than changing the data; `INamesTheAnswer` says it is an output, whichever kind, and a pipeline has one. An
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

The same key finds the rows that are there more than once. A repeated row in training and in test is one a
model meets again after learning it, which reads as skill and is not, so a profile counts them, and a split
keeps every copy of one in the same part.

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

A box on the grid changes the declaration, never the data. Every column has a box saying whether it is in, and
every column the schema takes a box saying whether it is a category, and both go through the one set of column
rules every door that changes the columns uses. Unticking a column nothing reads excludes it in the schema, which
keeps its kind; unticking one a step reads, or one a step made, writes a `drop.columns` block below the last step
that reads it and below the step that made it — never higher, even under a schema that keeps the rest of the file,
where any name may be read from the schema down. Ticking either brings it back as it was, and a column the schema
does not name comes in as text where the source has it — which only the source's own header says, so a tick made
before anything read the source shows the source first and takes nothing in. Ticking a category remembers the kind
the column was, and unticking gives that kind back. A box the rules would not let change is drawn but cannot be
clicked: the answer cannot be left out, nor the last column a schema takes, and a category that does not say what it
was keeps its box ticked. What the boxes say is what the grid draws: a column that is not in is black, and its values
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
schema does not name comes in with the kind its row shows: the saved file's kind for it, else text. A list finds its
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
showing a block writes nothing. It is written whole under a name of its own and then moved into place, so nothing
ever meets half a file, and the same decisions again leave it untouched. A file that cannot be read is never written
over, since it may hold what this notebook cannot see; a file that cannot be written says so at the block, and the
blocks keep the decisions. The source's columns it holds belong to the list of columns, and a write keeps them.

The columns saved beside a notebook are taken over in two presses, and the first changes nothing. The toolbar's
"Take over the saved columns" reads the file and the source's first line, works the take-over out against the blocks
as they stand, and lists it at the schema's block — at the source's, for blocks without a schema, which the saved one
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

Whatever changes the blocks — one gesture, or a change to several at once — goes through one way of writing them,
and it writes only the blocks whose steps changed. The steps both lists start and end with are left alone; between
them a step is matched with the one of the same verb in the same order, so it is written again where it stood and
keeps what the notebook holds about that block, or left alone, view and all, when it did not change. A step added
goes after the block before it, and one taken away is removed. So a change to the schema and to the output leaves
every step between them as it was, rather than writing them all again and clearing what they showed.

What lives between gestures is a session per notebook: which block shows which view, what a gesture asked a
block for, one view kept under its key and the bytes it was read from, and the rows the source opened last,
keyed by the read step, the path and a SHA-256 of the bytes, which are read and hashed on every use. Keeping
the parsed rows took close to half off every show on files of five and eleven megabytes, measured inside
Verso's own engine. The session is held by the block type Verso loaded, the one object every part of the
notebook reaches — the block type's own kernel directly, every other part through the host that loaded it —
and one gesture on it runs at a time. It is never static and never shared with another notebook. A grid whose
view, or whose boxes, the blocks no longer match is cleared, not worked out again: work runs when somebody asks
for it.

C# cells in the same notebook are handed the pipeline as text — the declaration, with what the whole pipeline's
run learned while it is the run of the steps declared now over the bytes there now — because the notebook's
types and a cell's are loaded apart and are not the same types even when their names are. The key is one no C#
variable can have, so a cell reads it afresh every time rather than the value it saw first. It is taken back
whenever the blocks may no longer make what it holds, and only a gesture or the toolbar's run hands it over
again, which is as far as the notebook can see: an edit that was never run, or a block deleted or moved, tells
no part of it anything.

A notebook of blocks is saved as a `.verso` file. Saving one as Jupyter is refused, because a Jupyter file has
no place for a block type and would keep the steps as code cells; opening a Jupyter file whose code cells read
as steps is refused too. Two ways around that are written down rather than trusted: `verso convert` writes
Jupyter without asking any extension, and a package installed from Verso's Extensions panel is loaded only after
a Jupyter file has been opened, so the refusal on the way in holds only for an install at the top of Verso's
extensions folder. A test pins the order Verso opens things in, so a Verso that changes it is noticed.

The same package runs wherever Verso does, and it is the same code in each: Verso's VS Code extension, the
browser editor `verso serve` starts, and an application that takes Verso's engine as an ordinary dependency. They
drive a part through the same interfaces and differ in what surrounds it — how the package is loaded, and who
draws the output and passes a click on. What does differ is the runtime, and not the way one would guess: the
VS Code host takes the newest .NET installed, while the browser host stays on .NET 8 for as long as .NET 8 is
there. A package built for .NET 10 alone failed to load in the browser — "System.Runtime, Version=10.0.0.0"
not found — so every package ships a build for .NET 8 and one for .NET 10, and a host takes the one that
matches the runtime it is on. The Extensions panel keeps one install per runtime, in a folder named after it,
and loads the newest one the runtime it is on can run, so two hosts on one machine each find their own. The
code is one code for both: where .NET 10 had a shorter way to say something, the way both runtimes have is the
one used — a JSON writer's line ending among them, so a pipeline file is written with a line feed on every
machine and runtime — and both suites run on both runtimes.

The block's kernel answers Verso's question for the faults in a text, but no Verso editor asks it: they ask for
completions and hover texts only. A fault is therefore shown where a block runs, on its card, each at its line.

An application that embeds the engine gets the notebook's parts from Verso's own discovery, which reads every
assembly beside the application that references Verso's abstractions; the notebook's own tests open it exactly
so. Verso's editor itself is not published for applications to reuse, so such an application does what the
editor does: it draws a block's output, which is HTML; it hands a control's `data-action` and `data-extension-id`
to the part the control names, with its `data-payload` — or, for a control that carries none, its state, as Verso's
own router sends it: `true` or `false` for a box, the value it is at for a select — and with the notebook's variables
and operations; it draws again when a block's output is updated or a gesture says it changed the blocks; it gives the
toolbar's buttons — Run, Export and the take-over — a context of its own; and it saves through a serializer that knows
the cell types, so what the blocks show stays out of the file. An application that only wants the pipeline reads each
block with `StepCatalog.ReadStep` and builds the declaration through its constructor, or has the command line write
the file with `verso export --format "Export the pipeline" --extensions <the published package>`. That file holds
the steps and no fit, because nothing ran them there.

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

### A tensor never changes

Handing the same tensor to two layers is safe, and a caller who reuses a scratch buffer cannot rewrite a
tensor they already handed over — `Tensor.From` copies. An in-place variant will come when a measurement
shows the copying costs something that matters; until then, the correctness is worth more than the
allocation.

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
instructions, nothing native. A TorchSharp backend belongs beside it as an equal — libtorch is a library
like any other, and rewriting what it already does well would be the most expensive way to learn nothing.

What differs between them is what a project has to ship, not what a model has to say. The light one travels
inside an application; libtorch is 76 MB per platform. Both are legitimate, the choice is the caller's, and
it is one line.

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

### Charts come from the training loop, not from the caller

When the training loop exists, the metrics it already keeps are what the charts are drawn from — the caller
never assembles arrays to plot. The drawing itself is **MatPlotLibNet**, in a separate optional package,
so a trainer on a headless machine does not carry a renderer.

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
tested by people who are not us. If a backend is ever written against it, that is borrowing a library, not
borrowing a result.

## What is deliberately absent

- **A global backend, context or session.** Nothing reaches for a shared instance. The notebook keeps a
  session, and it is the notebook's own: held by the block type Verso loaded for that notebook, handed to
  every part through the host that loaded it, never static and never shared.
- **A graph that is not a model.** The declarative front door will produce the same object the imperative
  one does. Two representations of one network means two engines to keep in step, and they diverge on the
  first unusual model.
- **A required engine.** No backend is mandatory. The core carries the light one; anything heavier is its
  own package, and a model cannot tell which is underneath it.
