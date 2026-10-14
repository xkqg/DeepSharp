// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// The two normalisations, over the last axis — a row's features, or an image's channels when they come last: how each
/// starts, what it keeps, and what it refuses, on any engine.
/// </summary>
/// <param name="engine">The backend the passes run on.</param>
public abstract class NormalisationContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    [Fact]
    public void ABatchNormalisation_StartsAtAScaleOfOneAndAShiftOfNothing_AndSaysWhatItKeeps()
    {
        var norm = new BatchNorm(3);

        Assert.Equal([1f, 1f, 1f], norm.Weight!.Value.Values.ToArray());
        Assert.Equal([0f, 0f, 0f], norm.Bias!.Value.Values.ToArray());
        Assert.Equal([0f, 0f, 0f], norm.RunningMean.Value.Values.ToArray());
        Assert.Equal([1f, 1f, 1f], norm.RunningVariance.Value.Values.ToArray());
        Assert.Equal(["weight", "bias", "running_mean", "running_var"], norm.Slots().Select(slot => slot.Path));
        Assert.Equal(0.1, norm.Momentum);
        Assert.Equal(1e-5, norm.Epsilon);
    }

    [Fact]
    public void ABatchOfOneRow_CannotBeMeasured_InATrainingPass()
    {
        var norm = new BatchNorm(3);

        Assert.Throws<ArgumentException>(() => norm.Forward(Tensor.Zeros(new Shape(1, 3)), Training()));
    }

    [Fact]
    public void ANormalisationOfFeaturesItDoesNotHave_OrOfNone_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new BatchNorm(3).Forward(Tensor.Zeros(new Shape(4, 2)), Training()));
        Assert.Throws<ArgumentException>(() => new LayerNorm(3).Forward(Tensor.Zeros(new Shape(4, 2)), Training()));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BatchNorm(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LayerNorm(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BatchNorm(3) { Momentum = 1.5 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new BatchNorm(3) { Epsilon = 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new LayerNorm(3) { Epsilon = -1 });
    }

    [Fact]
    public void ALayerNormalisation_KeepsAScaleAndAShift_AndNoRunningStatistic()
    {
        var norm = new LayerNorm(3);

        Assert.Equal(["weight", "bias"], norm.Slots().Select(slot => slot.Path));
        Assert.Equal(1e-5, norm.Epsilon);
    }

    [Fact]
    public void ANormalisationWithoutAScaleOrAShift_KeepsOnlyWhatItHas()
    {
        var withoutScale = new BatchNorm(3, new Affine { WithoutScale = true });
        var withoutShift = new LayerNorm(3, new Affine { WithoutShift = true });
        var neither = new BatchNorm(3, new Affine { WithoutScale = true, WithoutShift = true });

        Assert.Null(withoutScale.Weight);
        Assert.NotNull(withoutScale.Bias);
        Assert.Equal(["bias", "running_mean", "running_var"], withoutScale.Slots().Select(slot => slot.Path));
        Assert.NotNull(withoutShift.Weight);
        Assert.Null(withoutShift.Bias);
        Assert.Equal(["weight"], withoutShift.Slots().Select(slot => slot.Path));
        Assert.Equal(["running_mean", "running_var"], neither.Slots().Select(slot => slot.Path));
        Assert.Equal(3, neither.Features);
        Assert.Equal(3, withoutShift.Features);
    }

    [Fact]
    public void WhatANormalisationLeavesOut_IsWhatItStartedAt_SoItMakesWhatItDidAtTheStart()
    {
        // A scale of one and a shift of nothing change nothing, so leaving them out gives the same numbers, to the bit.
        var rows = Tensor.From(new Shape(4, 3), [0.5f, -1f, 2f, 1.5f, 0.25f, -3f, -0.75f, 2.5f, 1f, 0f, 1f, -2f]);
        var lean = new Affine { WithoutScale = true, WithoutShift = true };

        Assert.Equal(
            Bits(new BatchNorm(3).Forward(rows, Training())),
            Bits(new BatchNorm(3, lean).Forward(rows, Training())));
        Assert.Equal(
            Bits(new LayerNorm(3).Forward(rows, Training())),
            Bits(new LayerNorm(3, lean).Forward(rows, Training())));
        Assert.Equal(
            Bits(new LayerNorm(3).Forward(rows, Training())),
            Bits(new LayerNorm(3, new Affine { WithoutScale = true }).Forward(rows, Training())));
    }

    [Fact]
    public void ANormalisationWithoutAScaleOrAShift_IsWrittenSayingSo_AsTheFifthVersion_AndReadBackAsItWas()
    {
        var network = new LayerStack(new BatchNorm(3, new Affine { WithoutScale = true }), new LayerNorm(3, new Affine { WithoutShift = true }));
        var text = Written(network);
        var read = Assert.IsType<LayerStack>(NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()).Network);

        Assert.Contains("\"scale\": false", text, StringComparison.Ordinal);
        Assert.Contains("\"shift\": false", text, StringComparison.Ordinal);
        Assert.Equal(5, VersionOf(text));
        Assert.Null(Assert.IsType<BatchNorm>(read.Layers[0]).Weight);
        Assert.Null(Assert.IsType<LayerNorm>(read.Layers[1]).Bias);
        Assert.Equal(Written(read), text);
    }

    [Fact]
    public void ANormalisationWithAScaleAndAShift_IsWrittenAsItAlwaysWas()
    {
        var text = Written(new LayerStack(new BatchNorm(3), new LayerNorm(3)));

        Assert.DoesNotContain("\"scale\"", text, StringComparison.Ordinal);
        Assert.DoesNotContain("\"shift\"", text, StringComparison.Ordinal);
        Assert.Equal(2, VersionOf(text));
    }

    private static int[] Bits(Tensor tensor) => [.. tensor.Values.ToArray().Select(BitConverter.SingleToInt32Bits)];

    private static string Written(Network network)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();

        using (var writer = new System.Text.Json.Utf8JsonWriter(buffer, new System.Text.Json.JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, new MeanSquaredError());
            writer.WriteEndObject();
        }

        return System.Text.Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    private static int VersionOf(string text)
    {
        using var document = System.Text.Json.JsonDocument.Parse(text);

        return document.RootElement.GetProperty("network").GetProperty("version").GetInt32();
    }

    private Pass Training() => Pass.Training(_backend, new RandomStream(1), 0, 0);
}
