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

- **It fits an application that has a host.** `services.AddDeepSharpPipelines()` registers the pipeline
  factory and the catalog of verbs, so a pipeline is resolved the way everything else in a .NET application
  is. A package that brings verbs of its own registers them as a contribution, and every contribution is
  applied whichever order the registrations happen to be written in. None of it is required: a console
  program that writes `Pdd.Create()` with no container anywhere works exactly the same, and a test holds
  that door open. `Samples/DeepSharp.Sample.Pipelines` shows both.
