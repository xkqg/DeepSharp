// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// What attention costs as the batch grows. A layer that picks each series out of the batch with a product by a vector pays for every series
/// in the batch once for each series it picks: measured on the light engine, a training step over 128 series of 128 steps and 128 channels
/// held ten gigabytes at its peak. The memory a series costs must be the same whether it is read in a small batch or a large one.
/// </summary>
public class AttentionScalingTests
{
    private const int Channels = 64;
    private const int Steps = 32;

    private static readonly ITensorBackend Engine = new CpuBackend();

    [Fact]
    public void ATrainingStep_AllocatesForEachSeriesAsMuchInALargeBatchAsInASmallOne()
    {
        var small = Allocated(series: 8) / 8.0;
        var large = Allocated(series: 64) / 64.0;

        Assert.True(large <= small * 1.3, $"A series cost {small:N0} bytes in a batch of 8 and {large:N0} in a batch of 64.");
    }

    [Fact]
    public void AMaskedCausalTrainingStep_AllocatesForEachSeriesAsMuchInALargeBatchAsInASmallOne()
    {
        var small = Allocated(series: 8, masked: true) / 8.0;
        var large = Allocated(series: 64, masked: true) / 64.0;

        Assert.True(large <= small * 1.3, $"A series cost {small:N0} bytes in a batch of 8 and {large:N0} in a batch of 64.");
    }

    private static long Allocated(int series, bool masked = false)
    {
        var layer = new MultiHeadAttention(Channels, new Attending(4) { Causal = masked }, new RandomStream(1).Draw("scaling", 0, 0));
        var input = Tensor.From(new Shape(series, Steps, Channels), [.. Enumerable.Range(0, series * Steps * Channels).Select(at => (float)Math.Sin(at * 0.01))]);
        var mask = Tensor.From(new Shape(series, Steps), [.. Enumerable.Range(0, series * Steps).Select(at => at % Steps < Steps - (at / Steps % 5) ? 1f : 0f)]);
        var recording = new RecordingBackend(Engine);

        // The first pass warms what the second is measured against.
        Train(layer, input, masked ? mask : null, new RecordingBackend(Engine));

        var before = GC.GetAllocatedBytesForCurrentThread();

        Train(layer, input, masked ? mask : null, recording);

        return GC.GetAllocatedBytesForCurrentThread() - before;
    }

    private static void Train(MultiHeadAttention layer, Tensor input, Tensor? mask, RecordingBackend recording)
    {
        var pass = Pass.Training(recording, new RandomStream(1), 0, 0);
        var made = mask is null ? layer.Forward(input, pass) : layer.Forward(input, mask, pass);
        var loss = recording.Mean(recording.Multiply(made, made));

        _ = recording.GradientsOf(loss, layer.Parameters().Select(parameter => parameter.Value));
    }
}
