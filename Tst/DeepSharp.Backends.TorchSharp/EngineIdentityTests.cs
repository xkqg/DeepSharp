// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Backends.TorchSharp;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// What a checkpoint says of the engine a run was on: at full width the engine names itself exactly as the release before this one
/// had it name itself, so a checkpoint that release wrote of a run on libtorch goes on under it; at a narrower width it names the
/// width as well, and a run goes on only under the width it was taken under. The file the release before wrote was made by
/// <c>Fixtures/make-0.14.0.torch.cs</c>, from the packages on nuget.org.
/// </summary>
public class EngineIdentityTests
{
    private static readonly string Folder = Path.Join(AppContext.BaseDirectory, "Fixtures");

    [Fact]
    public void AnEngine_NamesTheWidthItKeepsItsWorkIn_AndAnEngineAtFullWidthNamesNone()
    {
        Assert.Null(((INamesItsPrecision)TorchBackend.OnCpu()).Precision);
        Assert.Null(((INamesItsPrecision)TorchBackend.OnCpu(Precision.Full)).Precision);
        Assert.Equal("bfloat16", ((INamesItsPrecision)TorchBackend.OnCpu(Precision.BFloat16)).Precision);
    }

    [Fact]
    public void ACheckpointOf0140OnLibtorch_IsReadBack_AndGoesOnUnderAFullEngine_AsTheRunItWas()
    {
        var prepared = Prepared();
        var resumed = CheckpointFile.Read(Text("checkpoint.json"), NetworkCatalog.BuiltIn(), prepared);

        Assert.Equal(2, resumed.Checkpoint.Epochs);

        var kept = resumed.Compiled.Fit(prepared, new FitOptions(seed: 20261017)
        {
            Epochs = 3,
            BatchSize = 16,
            Backend = TorchBackend.OnCpu(),
            ResumeFrom = resumed.Checkpoint,
        });

        Assert.Equal(3, kept.History!.Epochs.Count);
    }

    [Fact]
    public void ACheckpointOf0140OnLibtorch_GoneOnUnderAReducedEngine_IsRefused_NamingBothEngines()
    {
        var prepared = Prepared();
        var resumed = CheckpointFile.Read(Text("checkpoint.json"), NetworkCatalog.BuiltIn(), prepared);

        var refused = Assert.Throws<ArgumentException>(() => resumed.Compiled.Fit(prepared, new FitOptions(seed: 20261017)
        {
            Epochs = 3,
            BatchSize = 16,
            Backend = TorchBackend.OnCpu(Precision.BFloat16),
            ResumeFrom = resumed.Checkpoint,
        }));

        Assert.Contains("bfloat16", refused.Message, StringComparison.Ordinal);
        Assert.Contains(TorchSharp.torch.__version__, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFullEngine_NamesItselfInACheckpointExactlyAsTheReleaseBeforeDid()
    {
        var prepared = Prepared();
        var compiled = new Sequential().Dense(4).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy());
        var written = string.Empty;

        compiled.Fit(prepared, new FitOptions(seed: 20261017)
        {
            Epochs = 1,
            BatchSize = 16,
            Backend = TorchBackend.OnCpu(),
            Checkpoints = new Checkpoints(checkpoint => written = CheckpointFile.Write(compiled, prepared, checkpoint)),
        });

        using var document = JsonDocument.Parse(written);
        var training = document.RootElement.GetProperty("training");

        Assert.Equal(2, training.GetProperty("version").GetInt32());
        Assert.Equal(
            $"{{\"name\":\"torch\",\"version\":\"{TorchSharp.torch.__version__}\",\"device\":\"cpu\"}}",
            training.GetProperty("engine").GetRawText().Replace(" ", string.Empty, StringComparison.Ordinal).Replace("\n", string.Empty, StringComparison.Ordinal).Replace("\r", string.Empty, StringComparison.Ordinal));
    }

    [Fact]
    public void ARunOnAReducedEngine_GoesOnFromItsCheckpoint_AsTheSameRunItWould()
    {
        var prepared = Prepared();
        Checkpoint? third = null;
        var straight = new Sequential().Dense(4).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy());

        straight.Fit(prepared, new FitOptions(seed: 20261017)
        {
            Epochs = 6,
            BatchSize = 16,
            Backend = TorchBackend.OnCpu(Precision.BFloat16),
            Checkpoints = new Checkpoints(checkpoint => third = checkpoint.Epochs == 3 ? checkpoint : third),
        });

        var resumed = new Sequential().Dense(4).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy());

        resumed.Fit(prepared, new FitOptions(seed: 20261017) { Epochs = 6, BatchSize = 16, Backend = TorchBackend.OnCpu(Precision.BFloat16), ResumeFrom = third });

        foreach (var (one, other) in straight.Network.Slots().Zip(resumed.Network.Slots()))
        {
            Assert.Equal(one.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits), other.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits));
        }
    }

    private static string Text(string name) => File.ReadAllText(Path.Join(Folder, $"torch-0.14.0.{name}"));

    private static PreparedData Prepared() =>
        new Pipeline(PipelineDeclaration.FromJson(Text("pipeline.json"), StepCatalog.BuiltIn().WithNetworks()), rows: null, SourceFolder.Of(Folder)).Run();
}
