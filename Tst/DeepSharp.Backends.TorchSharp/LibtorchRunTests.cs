// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json.Nodes;
using DeepSharp.Backends.TorchSharp;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// The engine reaches a run the one way an engine does: the line that names it, <c>new FitOptions(seed) { Backend =
/// TorchBackend.OnCpu() }</c>, which trains the network, judges it by its validation rows and takes its report's measures on
/// libtorch; and a trained network serves on it when its caller hands it over. The network holds no engine and its file
/// names none, so a network trained on libtorch serves on the light engine, and one trained on the light engine on libtorch;
/// a checkpoint names the engine its run was on, and a run goes on from it on that engine alone.
/// </summary>
public class LibtorchRunTests
{
    [Fact]
    public void TheNetworksSamplesTitanicRun_TrainsAndIsJudgedAndMeasuredOnLibtorch_ThroughTheOneLine()
    {
        // The networks sample's run, as it stands, with the engine named: fourteen features, a dense layer of sixteen, Adam at
        // a hundredth, and Keras's early stopping keeping its best epoch.
        var engine = TorchBackend.OnCpu();

        var trained = new Sequential().Dense(16).Relu().Dense(1)
            .Compile(new Adam(0.01), new BinaryCrossEntropy())
            .Fit(Passengers.Prepared(), new FitOptions(seed: 20260929) { Backend = engine, Epochs = 100, EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true } });
        var test = trained.Measures!.Parts.Single(part => part.Part == Part.Test).Values.Single(value => value.Metric == Metric.Accuracy);

        // Every number the network learned stands where libtorch keeps it; the run stopped once the validation rows stopped
        // improving and ended holding its best epoch; and the report measured it better than the most common answer.
        Assert.All(trained.Network.Slots(), named => Assert.IsType<TorchStorage>(named.Slot.Value.Storage));
        Assert.Equal(Stopping.NoLongerImproving, trained.History!.Stopped);
        Assert.InRange(trained.History.Epochs.Count, 2, 99);
        Assert.Equal(trained.History.BestEpoch, trained.TrainedOn.Epoch);
        Assert.True(test.Value > 0.75 && test.Baseline < 0.62, string.Create(CultureInfo.InvariantCulture, $"test accuracy {test.Value}, the most common answer's {test.Baseline}"));
    }

    [Fact]
    public void ANetworkTrainedOnLibtorch_ServesOnItAndOnTheLightEngineAlike_AndItsFileReadsBackAsServed()
    {
        var engine = TorchBackend.OnCpu();
        var trained = new Sequential().Dense(16).Relu().Dense(1)
            .Compile(new Adam(0.01), new BinaryCrossEntropy())
            .Fit(Passengers.Prepared(), new FitOptions(seed: 20260929) { Backend = engine, Epochs = 5 });

        var onLibtorch = trained.Predict(Passengers.Served(), engine).Answers;
        var onLight = trained.Predict(Passengers.Served()).Answers;
        var read = TrainedNetwork.FromJson(trained.ToJson(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());

        // The same numbers, answered by one evaluation pass on each engine: within PyTorch's float tolerance of each other.
        AssertAlike(onLight, onLibtorch);
        Assert.Equal(onLight, read.Predict(Passengers.Served()).Answers);
        Assert.Equal(onLibtorch, read.Predict(Passengers.Served(), engine).Answers);
    }

    [Fact]
    public void ANetworkTrainedOnTheLightEngine_ServesOnLibtorch_AnsweringAlike_ItsPredictionsWorkedOutThere()
    {
        var engine = TorchBackend.OnCpu();
        var compiled = new Sequential().Dense(16).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy());
        var trained = compiled.Fit(Passengers.Prepared(), new FitOptions(seed: 7) { Epochs = 5 });
        var features = Tensor.From(new Shape(1, 14), [.. Enumerable.Range(0, 14).Select(at => (at % 3) - 1f)]);

        AssertAlike(trained.Predict(Passengers.Served()).Answers, trained.Predict(Passengers.Served(), engine).Answers);
        Assert.IsType<TorchStorage>(compiled.Predict(features, engine).Storage);
    }

    [Fact]
    public void ACheckpointOfARunOnTheLightEngine_GoesOnOnLibtorchNoMore_NorOneOfARunOnLibtorchOnTheLightEngine_EachRefusedNamingBoth()
    {
        var prepared = Passengers.Prepared();
        var libtorch = TorchBackend.OnCpu();
        var light = new CpuBackend();
        var onLight = Checkpointed(prepared, light);
        var onLibtorch = Checkpointed(prepared, libtorch);

        var toLibtorch = GoneOn(onLight, prepared, libtorch);
        var toLight = GoneOn(onLibtorch, prepared, light);

        Assert.Equal(
            $"The checkpoint was taken of a run on the engine 'cpu' {light.Version} on cpu, and going on under the engine 'torch' {libtorch.Version} on cpu would round every step otherwise. (Parameter 'options')",
            Assert.Throws<ArgumentException>(toLibtorch).Message);
        Assert.Equal(
            $"The checkpoint was taken of a run on the engine 'torch' {libtorch.Version} on cpu, and going on under the engine 'cpu' {light.Version} on cpu would round every step otherwise. (Parameter 'options')",
            Assert.Throws<ArgumentException>(toLight).Message);
    }

    [Fact]
    public void ACheckpointOfARunOnLibtorch_GoesOnOnAnEngineMadeTheSameWay_ToTheRunItWasTakenOf_BitForBit()
    {
        // The engine is recorded by what it names — libtorch's version and the device — not by which object it was: another
        // engine made on the processor is the same engine, and the run goes on to the bit.
        var prepared = Passengers.Prepared();
        var straight = Small().Fit(prepared, new FitOptions(seed: 7) { Backend = TorchBackend.OnCpu(), Epochs = 4 });

        var goneOn = GoneOn(Checkpointed(prepared, TorchBackend.OnCpu()), prepared, TorchBackend.OnCpu())();

        Assert.Equal(straight.History!.Epochs, goneOn.History!.Epochs);
        Assert.Equal(
            straight.Network.Slots().Select(named => named.Slot.Value.Values.ToArray()),
            goneOn.Network.Slots().Select(named => named.Slot.Value.Values.ToArray()));
    }

    [Fact]
    public void ACheckpointTakenOnAnotherLibtorch_IsRefusedOnThisOne_NamingBoth()
    {
        // The same engine on the same device, over another libtorch: libtorch promises no equal bits from one release to the
        // next, so to the checkpoint it is another engine.
        var prepared = Passengers.Prepared();
        var engine = TorchBackend.OnCpu();
        var file = JsonNode.Parse(Checkpointed(prepared, engine))!;
        file["training"]!["engine"]!["version"] = "2.9.0.0";

        var wrong = Assert.Throws<ArgumentException>(GoneOn(file.ToJsonString(), prepared, engine));

        Assert.Equal(
            $"The checkpoint was taken of a run on the engine 'torch' 2.9.0.0 on cpu, and going on under the engine 'torch' {engine.Version} on cpu would round every step otherwise. (Parameter 'options')",
            wrong.Message);
    }

    // The file of a run's checkpoint after its second epoch, on the engine handed.
    private static string Checkpointed(PreparedData prepared, ITensorBackend engine)
    {
        var compiled = Small();
        var files = new List<string>();

        compiled.Fit(prepared, new FitOptions(seed: 7) { Backend = engine, Epochs = 2, Checkpoints = new Checkpoints(checkpoint => files.Add(CheckpointFile.Write(compiled, prepared, checkpoint))) });

        return files[^1];
    }

    // The run gone on from a checkpoint's file to its fourth epoch, on the engine handed.
    private static Func<TrainedNetwork> GoneOn(string file, PreparedData prepared, ITensorBackend engine) => () =>
    {
        var resumed = CheckpointFile.Read(file, NetworkCatalog.BuiltIn(), prepared);

        return resumed.Compiled.Fit(prepared, new FitOptions(seed: 7) { Backend = engine, Epochs = 4, ResumeFrom = resumed.Checkpoint });
    };

    private static CompiledNetwork Small() => new Sequential().Dense(16).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy());

    // Each chance within PyTorch's float tolerance of the other engine's: an absolute 1e-5 and a relative 1.3e-6.
    private static void AssertAlike(IReadOnlyList<double[]> expected, IReadOnlyList<double[]> actual)
    {
        Assert.Equal(expected.Count, actual.Count);

        for (var row = 0; row < expected.Count; row++)
        {
            Assert.True(
                Math.Abs(expected[row][0] - actual[row][0]) <= 1e-5 + (1.3e-6 * Math.Abs(expected[row][0])),
                string.Create(CultureInfo.InvariantCulture, $"row {row}: {actual[row][0]} against {expected[row][0]}"));
        }
    }
}
