# Architecture

This file records the decisions that shape DeepSharp, and why each one was taken. A decision that is not
written down here is not a decision, it is a habit.

## The layout

```
Src/DeepSharp/            the engine side: Shape, Tensor, ITensorBackend, CpuBackend
Src/DeepSharp.Pipelines/  the data side: the declaration, its steps, the catalog, the builders
Samples/                  runnable programs, one per thing worth showing
Tst/DeepSharp/            the tests, mirroring both libraries' folders
```

The two libraries do not reference each other, and a test reads their assembly references to keep it that
way.

## The design language: PDD, pipeline-driven design

A design is named after whatever drives it, and here that is **the pipeline**.

Every piece of work follows one sequence: collect the data, add the features, normalise, deal with the
missing values, split into train, validation and test, build the model, and check it against data it has
never seen. The sequence is not the interesting part — everyone does those steps. What matters is that the
whole of it is **declared in advance as one artefact and then replayed**, instead of being assembled again
at each stage by whoever happens to be writing that stage.

That is the difference between a pipeline and a script. A script does the steps; a pipeline is a thing you
can hand to somebody, save beside a model, and run again a year later on data that did not exist yet.

### The chain is the enforcement, not the decoration

Everything is reached through a factory and built with a fluent chain, and the stages of that chain are
different types. A pipeline under construction offers `Normalise` and `FillMissing` **only after** it has
been told how to split, because those are the operations that learn from the data.

Which split is a separate question, and the library does not answer it: random for independent rows, by
time when you predict forward, stratified when a class is rare, by group when several rows belong to one
entity, explicit when the data already carries the answer. Each is offered at the same point and none is
the default, because a default there is a guess about somebody else's data. What is fixed is the ordering
alone.

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

The drawing lives in a separate, optional package; the pipeline holds only the declaration.

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
.FillMissing("volume", With.Previous)  // With.Constant(0), With.Previous, With.DropRow, With.Refuse
.FillNaN("range", With.Refuse)         // the default: a NaN stops the run, because it should not be there
```

The strategy is a named value rather than a flag. A boolean parameter says nothing at the place it is
written — `Fill("trades", true, false)` has to be read with the signature open beside it — whereas the
name is the sentence.

Marking where a gap was is not a parameter either: a `trades_was_missing` column is emitted alongside, and
always. Filling destroys the distinction between "absent" and "the value happened to be that", and it
destroys it irreversibly; a caller who does not want the column drops it like any other column, which is a
verb they already have.

### The declaration is a file, and the chain can write it

The pipeline is saved as one file with two blocks, because they have different authors. The
**declaration** — the steps in order with their arguments — is written by a person. The **fitted** part —
the means, the fill values, the category lists, the boundary the split landed on — is written by the fit.
Keeping them apart is what allows the same declaration to be re-fitted on fresh data, two runs to be
compared by diffing the declaration alone, and a serving process to load the file without ever knowing the
builder existed.

The format is deliberately not the pipeline. One internal declaration model is the truth, and every format
is a front end that produces it: JSON for machines, because it diffs and travels; YAML for people, because
it carries comments and loses the punctuation; a spreadsheet or a generated form later, for the same reason
and at the same cost, which is a parser rather than a redesign. The same seam as the readers.

One thing does not survive being written down, and it decides the shape of the rest: an inline lambda. A
custom step is therefore registered under a name and looked up while parsing, and a file naming a step that
nobody registered **refuses to load** instead of quietly skipping it. That pushes features towards a named
vocabulary, which is exactly what makes the file portable.

### One validator, and a template that falls out of the same model

The template and the validator are both generated from the declaration model rather than written beside
it. Anything hand-maintained drifts from the code it describes, and a validator that drifts is worse than
none: a file can then be legal in a way the fluent chain is not.

Validation has two halves and they run at different moments. **Shape** — does the step exist, are its
parameters the right kind, do the three split fractions add to one — needs no data at all. **Binding** —
does that column exist in the source, is it numeric — needs the source open. Both finish before anything
runs, and both report every fault at once with its position in the file. Someone who is handed one error at
a time, five times over, stops using the thing.

```
btceur.pdd.yaml(7,3): column 'trades' is not in btceur-1d.csv — there is a 'numberOfTrades'
btceur.pdd.yaml(4,10): train+validation+test = 0.95, must be 1.0
btceur.pdd.yaml(2,8): step 'parquet' exists, but the package DeepSharp.Pipelines.Parquet is not referenced
```

That last message is deliberately not the same as the first kind. "I do not know this step" and "I know it,
but you have not installed it" are two different problems for the reader, and collapsing them costs an
afternoon.

The file also names the declaration version it was written against, so a pipeline from a year ago either
loads or says precisely which step changed underneath it.

### One description drives both doors

A pipeline can be written in C# or written as a file, and both have to reach exactly as far. That is only
true if neither is the description: the step types are, and everything else is derived from them. The
fluent chain is one projection of those types, the JSON schema and the starter template are another, the
validator reads the same metadata, and the reference page in the documentation is generated rather than
typed. Add a step, and every one of those gains it without anyone remembering to go and edit it.

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
answer of its own. And whether the target is normalised, because if it is, the way back is part of the
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
implements none is a convenience and belongs in whatever it is convenient for.

A seam is only real when two implementations of it differ in kind, so each one is held to that: the backend
has a pure-managed engine beside a native one, rows arrive from a file and from a database reader, a
learner is a network or a decision tree, an importer reads one foreign format or another. An interface
shaped around a single implementation is not a seam, it is that implementation with a longer name — and the
cost is paid later, by the second implementation that turns out not to fit.

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
never assembles arrays to plot. The drawing itself lives in a separate, optional package, so a trainer on a
headless machine does not carry a renderer.

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

- **A global backend, context or session.** Nothing reaches for a shared instance.
- **A graph that is not a model.** The declarative front door will produce the same object the imperative
  one does. Two representations of one network means two engines to keep in step, and they diverge on the
  first unusual model.
- **A required engine.** No backend is mandatory. The core carries the light one; anything heavier is its
  own package, and a model cannot tell which is underneath it.
