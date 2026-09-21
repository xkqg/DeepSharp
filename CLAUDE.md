# Project rules — see CONTRIBUTING.md

All contributor rules for this repository live in [**CONTRIBUTING.md**](CONTRIBUTING.md): a failing test
before the code, zero warnings, the coverage gate, class naming, the version being handed out rather than
invented, the documentation sweep, and the commit style. Read it before any commit.

The decisions behind the design, and what is deliberately absent, are in
[**ARCHITECTURE.md**](ARCHITECTURE.md). Read it before changing a shape of the code rather than a line of it.

## What this repository is not

- **Not a binding.** TorchSharp and TensorFlow.NET hand you the Python API in C# syntax with a native engine
  underneath. This is a library written for C#, with no native dependency in the core.
- **Not a home for GPU code.** The fleet's GPU training host is a separate repository and stays that way.
  The backend seam is how heavy work reaches a native engine; it never becomes required here.
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
```

The test project is an executable — xunit v3 runs in-process, so `dotnet test` is not how a suite is run
here. Use `--class` or `--method` to run one.
