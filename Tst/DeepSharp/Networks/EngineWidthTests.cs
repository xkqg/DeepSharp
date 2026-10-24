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
/// A run on an engine that keeps its work in a narrower width than a single number says so in its checkpoint, as a run says the
/// device it was on: the training part is then the ninth version, a run goes on only on an engine of the same width, and a run on an
/// engine at full width is written exactly as it always was.
/// </summary>
public class EngineWidthTests
{
    [Fact]
    public void ARunOnAnEngineKeepingItsWorkNarrow_NamesTheWidthInItsCheckpoint_WhichIsThenTheNinthVersion()
    {
        var (compiled, checkpoint) = Taken(new WideningBackend(new CpuBackend(), "bfloat16"));
        var written = Written(compiled, checkpoint);

        Assert.Equal(9, VersionOf(written));
        Assert.Contains("\"precision\":\"bfloat16\"", written, StringComparison.Ordinal);
    }

    [Fact]
    public void ARunOnAnEngineAtFullWidth_IsWrittenAsItWasWritten_TheSecondVersionAndNoWidth()
    {
        var (compiled, checkpoint) = Taken(new WideningBackend(new CpuBackend()));
        var written = Written(compiled, checkpoint);

        Assert.Equal(2, VersionOf(written));
        Assert.DoesNotContain("precision", written, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpointOfNarrowWork_IsReadBack_AndWrittenAgainAsItWas()
    {
        var (compiled, checkpoint) = Taken(new WideningBackend(new CpuBackend(), "half"));
        var written = Written(compiled, checkpoint);

        var read = NetworkDocument.ReadCheckpoint(written, "network", "training", NetworkCatalog.BuiltIn());

        Assert.Equal(written, Written(read.Run.Compiled, read.Run.Checkpoint));
    }

    [Fact]
    public void ACheckpointOfNarrowWork_GoneOnUnderFullWidth_IsRefused_NamingBothEngines()
    {
        var (compiled, checkpoint) = Taken(new WideningBackend(new CpuBackend(), "bfloat16"));

        var refused = Assert.Throws<ArgumentException>(() => compiled.Fit(Rows(), validation: null, new FitOptions(seed: 3)
        {
            Epochs = 2,
            BatchSize = 2,
            Backend = new WideningBackend(new CpuBackend()),
            ResumeFrom = checkpoint,
        }));

        Assert.Contains("bfloat16", refused.Message, StringComparison.Ordinal);
        Assert.Contains("engine", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpointOfFullWidth_GoneOnUnderNarrowWork_IsRefused_NamingBothEngines()
    {
        var (compiled, checkpoint) = Taken(new WideningBackend(new CpuBackend()));

        var refused = Assert.Throws<ArgumentException>(() => compiled.Fit(Rows(), validation: null, new FitOptions(seed: 3)
        {
            Epochs = 2,
            BatchSize = 2,
            Backend = new WideningBackend(new CpuBackend(), "bfloat16"),
            ResumeFrom = checkpoint,
        }));

        Assert.Contains("bfloat16", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ATrainingPartOfAnEarlierVersion_ThatNamesAWidth_IsRefused_NamingTheVersionThatSaysIt()
    {
        var (compiled, checkpoint) = Taken(new WideningBackend(new CpuBackend(), "bfloat16"));
        var older = Written(compiled, checkpoint).Replace("\"version\":9", "\"version\":8", StringComparison.Ordinal);

        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadCheckpoint(older, "network", "training", NetworkCatalog.BuiltIn()));

        Assert.Contains("ninth version of a training part", refused.Message, StringComparison.Ordinal);
    }

    private static TrainingData Rows() => new(
        Tensor.From(new Shape(4, 1), [0.1f, 0.2f, 0.3f, 0.4f]),
        Tensor.From(new Shape(4, 1), [0.2f, 0.4f, 0.6f, 0.8f]));

    // A run of one epoch on the engine whose checkpoint is kept.
    private static (CompiledNetwork Compiled, Checkpoint Checkpoint) Taken(ITensorBackend engine)
    {
        Checkpoint? kept = null;
        var compiled = new LayerStack(new Dense(1, 1, new RandomStream(5).Draw("initialise:0", 0, 0))).Compile(new Sgd(0.1), new MeanSquaredError());

        compiled.Fit(Rows(), validation: null, new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2, Backend = engine, Checkpoints = new Checkpoints(checkpoint => kept = checkpoint) });

        return (compiled, kept!);
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
