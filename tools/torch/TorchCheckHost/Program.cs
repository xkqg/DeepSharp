// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// An application that takes DeepSharp.Backends.TorchSharp from the package just made, and nothing of DeepSharp's
// source: what it asks of TorchBackend.OnCpu() is what any application asks. tools/torch/check.sh builds this twice —
// once bringing no libtorch, where the engine is refused, naming every package that brings one, and once bringing the
// processor's, where it fits a step of the networks sample's Titanic pipeline.
//
//   TorchCheckHost refused
//   TorchCheckHost titanic <the passenger list>

using DeepSharp.Backends.TorchSharp;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

if (args is not [var expect, ..] || expect is not ("refused" or "titanic"))
{
    Console.Error.WriteLine("TorchCheckHost refused|titanic");
    return 2;
}

if (expect == "refused")
{
    return Refused();
}

if (args is not [_, var csv])
{
    Console.Error.WriteLine("TorchCheckHost titanic <the passenger list>");
    return 2;
}

return Titanic(csv);

// No libtorch package was brought: TorchBackend.OnCpu() is refused, and the refusal names every package that would
// bring one — the processor's for each platform, TorchSharp-cpu, and a graphics card's — so a person reads the fix
// off the message rather than a stack trace.
int Refused()
{
    try
    {
        var engine = TorchBackend.OnCpu();

        Fails($"brought no libtorch, and TorchBackend.OnCpu() made an engine anyway, naming {engine.Version} on {engine.Device}.");
        return 1;
    }
    catch (InvalidOperationException refused)
    {
        string[] named =
        [
            "libtorch-cpu-win-x64", "libtorch-cpu-linux-x64", "libtorch-cpu-osx-arm64",
            "TorchSharp-cpu", "TorchSharp-cuda-windows", "TorchSharp-cuda-linux",
        ];
        var missing = named.Where(package => !refused.Message.Contains(package, StringComparison.Ordinal)).ToArray();

        if (missing.Length > 0)
        {
            Fails($"was refused, but not naming {string.Join(", ", missing)}: {refused.Message}");
            return 1;
        }

        Holds($"brought no libtorch: TorchBackend.OnCpu() was refused, naming every package that brings one");
        return 0;
    }
}

// The processor's libtorch was brought: the engine fits one epoch of the networks sample's Titanic pipeline, every
// loss a finite number, every slot it learned on libtorch's own storage.
int Titanic(string csv)
{
    var engine = TorchBackend.OnCpu();
    var trained = new Sequential().Dense(16).Relu().Dense(1)
        .Compile(new Adam(0.01), new BinaryCrossEntropy())
        .Fit(Prepared(csv), new FitOptions(seed: 20260929) { Backend = engine, Epochs = 1 });

    if (trained.History is not { Epochs: [var epoch] } || !double.IsFinite(epoch.Loss))
    {
        Fails($"brought the processor's libtorch, and the step did not come to a finite loss: {(trained.History is null ? "no history" : string.Join(", ", trained.History.Epochs.Select(each => each.Loss)))}");
        return 1;
    }

    // The storage itself is internal to the engine's own package, so an application outside it reads what it is by
    // name, the way an application outside would: nothing else stands for "this tensor's values live in libtorch".
    var notOnLibtorch = trained.Network.Slots()
        .Where(named => named.Slot.Value.Storage.GetType().FullName != "DeepSharp.Backends.TorchSharp.TorchStorage")
        .Select(named => named.Path)
        .ToArray();

    if (notOnLibtorch.Length > 0)
    {
        Fails($"brought the processor's libtorch, and fit a step, but {string.Join(", ", notOnLibtorch)} did not learn on its storage: the package run here is not the one referenced from the feed.");
        return 1;
    }

    Holds($"brought the processor's libtorch: fit a step of the networks sample's Titanic pipeline, loss {epoch.Loss}, every slot on libtorch's storage");
    return 0;
}

void Holds(string what) => Console.WriteLine($"    {what}");

// A line GitHub's runner shows as an error, and a failure this host ends with.
void Fails(string what) => Console.WriteLine($"::error::{what}");

// The networks sample's Titanic pipeline: read, declared, divided by who survived, the age filled, every category
// one-hot, four scales onto minus one to one, the answer.
static PreparedData Prepared(string csv) =>
    Pdd.Create()
        .ReadCsv(csv)
        .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("pclass", "sex").Optional("age", ColumnKind.Number).Number("fare"))
        .SplitStratified("survived", train: 0.70, validation: 0.15)
        .FillMissing("age", With.Median)
        .EncodeCategories()
        .Normalise("age", Scale.MidRange)
        .Normalise("fare", Scale.MidRange)
        .Normalise("sibsp", Scale.MidRange)
        .Normalise("parch", Scale.MidRange)
        .Target("survived")
        .Build()
        .Run();
