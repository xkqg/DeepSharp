// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A run that scales its loss says so in its checkpoint, with where the scale had got to, and the epochs that skipped a step say how
/// many: the training part is then the ninth version, a part of an earlier one that says either is refused by that, and a run that
/// scales nothing and skips nothing is written exactly as it always was.
/// </summary>
public class LossScaleFileTests
{
    private static readonly Tensor Far = Tensor.From(new Shape(8, 1), [1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f]);

    private static readonly Tensor FarAnswers = Tensor.From(new Shape(8, 1), [.. Enumerable.Repeat(1000f, 8)]);

    [Fact]
    public void ARunUnderADynamicLossScale_IsWrittenAsTheNinthVersion_ReadBack_AndWrittenAgainAsItWas()
    {
        var (compiled, checkpoint) = Taken(new CpuBackend(), LossScale.Dynamic(initial: 1024, growth: 2, interval: 3, backoff: 0.5), epochs: 4);
        var written = Written(compiled, checkpoint);

        Assert.Equal(9, VersionOf(written));
        Assert.Contains("\"lossScale\":{\"scale\":2048,\"clean\":1,\"initial\":1024,\"growth\":2,\"interval\":3,\"backoff\":0.5}", written, StringComparison.Ordinal);

        var read = NetworkDocument.ReadCheckpoint(written, "network", "training", NetworkCatalog.BuiltIn());

        Assert.Equal(new ScalerState(2048, 1), read.Run.Checkpoint.State.Scaler);
        Assert.Equal(LossScale.Dynamic(initial: 1024, growth: 2, interval: 3, backoff: 0.5), read.Run.Checkpoint.State.Pace!.Value.LossScale);
        Assert.Equal(written, Written(read.Run.Compiled, read.Run.Checkpoint));
    }

    [Fact]
    public void ARunUnderAFixedLossScale_IsWrittenAndReadBackFixed()
    {
        var (compiled, checkpoint) = Taken(new CpuBackend(), LossScale.Fixed(256), epochs: 2);
        var written = Written(compiled, checkpoint);
        var read = NetworkDocument.ReadCheckpoint(written, "network", "training", NetworkCatalog.BuiltIn());

        Assert.Equal(LossScale.Fixed(256), read.Run.Checkpoint.State.Pace!.Value.LossScale);
        Assert.Equal(new ScalerState(256, 0), read.Run.Checkpoint.State.Scaler);
        Assert.Equal(written, Written(read.Run.Compiled, read.Run.Checkpoint));
    }

    [Fact]
    public void AnEpochThatSkippedAStep_SaysHowMany_AndOneThatSkippedNoneSaysNothing()
    {
        var (compiled, checkpoint) = Taken(new RoundedBackend(RoundedBackend.Width.Half), LossScale.Dynamic(), epochs: 2);
        var written = Written(compiled, checkpoint);

        Assert.Contains("\"skipped\":1", written, StringComparison.Ordinal);

        var read = NetworkDocument.ReadCheckpoint(written, "network", "training", NetworkCatalog.BuiltIn());

        Assert.Equal([1, 1], read.Run.Checkpoint.State.History.Select(epoch => epoch.Skipped));
        Assert.Equal(written, Written(read.Run.Compiled, read.Run.Checkpoint));
    }

    [Fact]
    public void ARunThatScalesNothingAndSkipsNothing_IsWrittenAsItWasWritten()
    {
        var (compiled, checkpoint) = Taken(new CpuBackend(), loss: null, epochs: 2);
        var written = Written(compiled, checkpoint);

        Assert.Equal(2, VersionOf(written));
        Assert.DoesNotContain("lossScale", written, StringComparison.Ordinal);
        Assert.DoesNotContain("skipped", written, StringComparison.Ordinal);
    }

    [Fact]
    public void ALossScaleInATrainingPartOfAnEarlierVersion_IsRefused_NamingTheVersionThatSaysIt()
    {
        var (compiled, checkpoint) = Taken(new CpuBackend(), LossScale.Fixed(256), epochs: 1);
        var older = Written(compiled, checkpoint).Replace("\"version\":9", "\"version\":8", StringComparison.Ordinal);

        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadCheckpoint(older, "network", "training", NetworkCatalog.BuiltIn()));

        Assert.Contains("ninth version of a training part", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASkippedCountInATrainingPartOfAnEarlierVersion_IsRefused_NamingTheVersionThatSaysIt()
    {
        var (compiled, checkpoint) = Taken(new RoundedBackend(RoundedBackend.Width.Half), LossScale.Dynamic(), epochs: 1);
        var older = Written(compiled, checkpoint).Replace("\"version\":9", "\"version\":8", StringComparison.Ordinal);

        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadCheckpoint(older, "network", "training", NetworkCatalog.BuiltIn()));

        Assert.Contains("ninth version of a training part", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALossScaleThatIsNotAPowerOfTwo_IsRefusedAtItsPlace()
    {
        var (compiled, checkpoint) = Taken(new CpuBackend(), LossScale.Fixed(256), epochs: 1);
        var wrong = Written(compiled, checkpoint).Replace("\"initial\":256", "\"initial\":300", StringComparison.Ordinal);

        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadCheckpoint(wrong, "network", "training", NetworkCatalog.BuiltIn()));

        Assert.Contains("loss scale", refused.Message, StringComparison.Ordinal);
        Assert.Contains("power of two", refused.Message, StringComparison.Ordinal);
    }

    private static (CompiledNetwork Compiled, Checkpoint Checkpoint) Taken(ITensorBackend engine, LossScale? loss, int epochs)
    {
        Checkpoint? last = null;
        var compiled = new LayerStack(new Dense(1, 1, new RandomStream(5).Draw("initialise:0", 0, 0))).Compile(new Sgd(0.01), new MeanSquaredError());

        compiled.Fit(new TrainingData(Far, FarAnswers), validation: null, new FitOptions(seed: 3)
        {
            Epochs = epochs,
            BatchSize = 8,
            Backend = engine,
            LossScale = loss,
            Checkpoints = new Checkpoints(checkpoint => last = checkpoint),
        });

        return (compiled, last!);
    }

    private static string Written(CompiledNetwork compiled, Checkpoint checkpoint)
    {
        var buffer = new ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, compiled.Network, compiled.Loss);
            writer.WritePropertyName("training");
            NetworkDocument.WriteTraining(writer, compiled, checkpoint);
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    private static int VersionOf(string file)
    {
        using var document = JsonDocument.Parse(file);

        return document.RootElement.GetProperty("training").GetProperty("version").GetInt32();
    }
}
