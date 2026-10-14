// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The training part of a checkpoint is written as the oldest version that says everything in it, as a network's part is:
/// the second unless the run clipped its gradients, the third when it did — and the version an optimizer or a schedule names
/// for itself when it says more than an older library would understand. A run whose optimizer says something new is
/// therefore refused by its number by a library that cannot read it, never by a key that library has not heard of; and a run
/// that says none of it is written exactly as it was.
/// </summary>
public class TrainingPartVersionTests
{
    [Fact]
    public void ARunThatClippedNothing_UnderAnOrdinaryOptimizer_IsWrittenAsTheSecondVersion() =>
        Assert.Equal(2, TrainingVersion(new Sgd(0.1), clip: null));

    [Fact]
    public void ARunThatClipped_UnderAnOrdinaryOptimizer_IsWrittenAsTheThirdVersion() =>
        Assert.Equal(3, TrainingVersion(new Sgd(0.1), new GradientClip(1.0)));

    [Fact]
    public void ARunWhoseOptimizerSaysMore_IsWrittenAsTheVersionTheOptimizerNames_WhateverItClipped()
    {
        Assert.Equal(5, TrainingVersion(new SaysMore(0.1), clip: null));
        Assert.Equal(5, TrainingVersion(new SaysMore(0.1), new GradientClip(1.0)));
    }

    [Fact]
    public void ARunWhoseScheduleSaysMore_IsWrittenAsTheVersionTheScheduleNames()
    {
        var compiled = new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(0.1), new MeanSquaredError(), new SaysMoreSchedule());

        Assert.Equal(5, VersionOf(Written(compiled, Checkpoint(compiled, clip: null)), "training"));
    }

    [Fact]
    public void AnOptimizerThatNamesAVersionBelowTheSecond_RaisesNothing_ForItIsTheDefaultsToo() =>
        Assert.Equal(2, TrainingVersion(new SaysNothingMore(0.1), clip: null));

    [Fact]
    public void ATrainingPartOfTheFifthVersion_IsReadBack_AndNothingOfALaterOneIs()
    {
        var compiled = new LayerStack(new Dense(1, 1, Draws())).Compile(new SaysMore(0.1), new MeanSquaredError());
        var written = Written(compiled, Checkpoint(compiled, clip: null));
        var catalog = NetworkCatalog.BuiltIn().Register<SaysMore>();
        var read = NetworkDocument.ReadCheckpoint(written, "network", "training", catalog);

        Assert.IsType<SaysMore>(read.Run.Compiled.Optimizer);
        Assert.Equal(written, Written(read.Run.Compiled, read.Run.Checkpoint));

        var later = written.Replace("\"version\":5", "\"version\":6", StringComparison.Ordinal);
        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadCheckpoint(later, "network", "training", catalog));

        Assert.Contains("version 6", refused.Message, StringComparison.Ordinal);
        Assert.Contains("up to version 5", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheVersionAFileOfThisLibraryCanBe_IsTheFifth() => Assert.Equal(5, NetworkDocument.Version);

    private static int TrainingVersion(Optimizer optimizer, GradientClip? clip)
    {
        var compiled = new LayerStack(new Dense(1, 1, Draws())).Compile(optimizer, new MeanSquaredError());

        return VersionOf(Written(compiled, Checkpoint(compiled, clip)), "training");
    }

    // A run of one epoch whose checkpoint is kept, so the training part has a run to write.
    private static Checkpoint Checkpoint(CompiledNetwork compiled, GradientClip? clip)
    {
        Checkpoint? kept = null;
        var rows = new TrainingData(
            Tensor.From(new Shape(4, 1), [0.1f, 0.2f, 0.3f, 0.4f]),
            Tensor.From(new Shape(4, 1), [0.2f, 0.4f, 0.6f, 0.8f]));

        compiled.Fit(rows, validation: null, new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2, GradientClip = clip, Checkpoints = new Checkpoints(checkpoint => kept = checkpoint) });

        return kept!;
    }

    // The network and the training part, as one file holds them under two keys.
    private static string Written(CompiledNetwork compiled, Checkpoint checkpoint)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();

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

    private static int VersionOf(string file, string part)
    {
        using var document = JsonDocument.Parse(file);

        return document.RootElement.GetProperty(part).GetProperty("version").GetInt32();
    }

    private static Draws Draws() => new RandomStream(5).Draw("initialise:0", 0, 0);

    // An optimizer written outside the library that says it needs the fifth version of the part it is written into.
    private sealed class SaysMore(double rate) : Optimizer(rate), ISaved<SaysMore>
    {
        public static string Name => "saysMore";

        public int FileVersion => 5;

        public static SaysMore Rebuild(JsonElement settings, Rebuilding rebuilding) => new(rebuilding.Number(settings, "rate"));

        public void WriteSettings(Utf8JsonWriter writer) => writer.WriteNumber("rate", Rate);

        protected override SlotMemory? MemoryOf(Parameter parameter) => null;

        protected override void Recall(Parameter parameter, SlotMemory memory)
        {
        }

        protected override Tensor Moved(Parameter parameter, Tensor gradient, double rate, ITensorBackend backend) =>
            backend.Subtract(parameter.Value, backend.Scale(gradient, Scalar(backend, rate)));
    }

    // One that names a version below the second, as an older kind might have: it is the default, so it raises nothing.
    private sealed class SaysNothingMore(double rate) : Optimizer(rate), ISaved<SaysNothingMore>
    {
        public static string Name => "saysNothingMore";

        public int FileVersion => 1;

        public static SaysNothingMore Rebuild(JsonElement settings, Rebuilding rebuilding) => new(rebuilding.Number(settings, "rate"));

        public void WriteSettings(Utf8JsonWriter writer) => writer.WriteNumber("rate", Rate);

        protected override SlotMemory? MemoryOf(Parameter parameter) => null;

        protected override void Recall(Parameter parameter, SlotMemory memory)
        {
        }

        protected override Tensor Moved(Parameter parameter, Tensor gradient, double rate, ITensorBackend backend) =>
            backend.Subtract(parameter.Value, backend.Scale(gradient, Scalar(backend, rate)));
    }

    // A schedule written outside the library that says it needs the fifth version.
    private sealed class SaysMoreSchedule : LearningRateSchedule, ISaved<SaysMoreSchedule>
    {
        public static string Name => "saysMoreSchedule";

        public int FileVersion => 5;

        public static SaysMoreSchedule Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

        public void WriteSettings(Utf8JsonWriter writer)
        {
        }

        protected override double Rate(int epoch, double initial) => initial;
    }
}
