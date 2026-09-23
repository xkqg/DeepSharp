# Project rules — see CONTRIBUTING.md

All contributor rules for this repository live in [**CONTRIBUTING.md**](CONTRIBUTING.md): a failing test
before the code, zero warnings, the coverage gate, class naming, the version being handed out rather than
invented, the documentation sweep, and the commit style. Read it before any commit.

The decisions behind the design, and what is deliberately absent, are in
[**ARCHITECTURE.md**](ARCHITECTURE.md). Read it before changing a shape of the code rather than a line of it.

## PDD — the design language here

This repository follows **pipeline-driven design**: the sequence from raw data to a validated model is
declared in advance as one replayable artefact, and **anything that learns from the data is fitted on the
training split alone and replayed unchanged** on validation, on test and on live data. `ARCHITECTURE.md` carries the
reasoning and the failure it prevents. A step that recomputes a learned parameter outside the pipeline is
a defect even when every test passes.

## The rule that decides what gets built here

**Anything you can take from outside is something you do not have to write or maintain.** That is the
default, and it holds for the boring, well-solved things: a DataFrame, a file format, a compression scheme,
a test runner, a maths primitive. Writing one of those here buys nothing and costs forever.

libtorch is on that list as much as anything else. TorchSharp is maintained by people who are not us, and
rewriting what it already does well would be the most expensive way to learn nothing. What gets built here
is the part nobody else provides: the C# shape of a model, the path data takes into it, the loop that trains
it, and the picture at the end.

The one rule about a dependency is **where it lands**. Anything heavy gets its own package, so a project
that does not want it never carries it. The core stays light enough to travel inside an application, and a
model is written against the seam and cannot tell which engine is underneath.

## What this repository is not

- **Not a binding.** TorchSharp and TensorFlow.NET hand you the Python API in C# syntax. This is a library
  written for C# that sits on top of such an engine and provides what an engine does not: the model's shape,
  the data path, the training loop and the charts.
- **Not a home for hand-written GPU code.** Reaching a GPU means using an engine that already does, through
  the seam. No CUDA is written here, and the GPU training host it would reach stays a separate repository.
- **Nothing is carried over from that host.** Its tuning, its shortcuts and the tricks it earned for its own
  workload stay where they were measured — they were written for one machine, one model family and one kind
  of data, and moved here they would be cargo. A published binding such as TorchSharp is a different matter
  entirely: that is a dependency anyone can read, version and replace, and it is a fair backend to sit behind
  the seam. The distinction is not GPU versus CPU; it is a shared library versus somebody else's tuning.
- **Not a place to copy MatPlotLibNet into.** The charting library is a dependency of the optional drawing
  package, never a base. Its machinery — the gate, the contract tests, the release discipline — is a
  template, not a reference.

## Running it

```
dotnet build DeepSharp.slnx -c Release          # ends at zero warnings, or it fails
dotnet run --project Tst/DeepSharp/DeepSharp.Tests.csproj -c Release
dotnet run --project Tst/DeepSharp.Notebooks.Verso/DeepSharp.Notebooks.Verso.Tests.csproj -c Release
./tools/coverage/run.ps1 -Check                  # every suite, 90/90 per class, as CI runs it
```

There are two suites: the notebook's runs inside Verso's own engine, which needs another version of the C#
compiler than the core suite's schema validator. Each test project is an executable — xunit v3 runs
in-process, so `dotnet test` is not how a suite is run here. Pass `-- -class <full name>` or
`-- -method <full name>` to run one: a single dash, because the runner refuses `--class` as an unknown option.
