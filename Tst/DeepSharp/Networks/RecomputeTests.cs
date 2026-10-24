// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// What a recomputed layer costs and keeps: it is run again once however many steps it takes, the pass that runs it again moves no
/// running statistic, a gradient reaches the numbers learned inside it, and what its workings held is not held while the rest of the
/// pass goes on.
/// </summary>
public class RecomputeTests
{
    [Fact]
    public void AReplayingPass_MovesNoStatistic_WhereATrainingPassDoes()
    {
        var norm = new BatchNorm(3, default);
        var training = Pass.Training(new CpuBackend(), new RandomStream(1), 0, 0);
        var replaying = training.Replaying(new CpuBackend());
        var seen = Tensor.From(new Shape(3), [9f, 9f, 9f]);

        norm.RunningMean.Update(seen, replaying);
        Assert.Equal([0f, 0f, 0f], norm.RunningMean.Value.Values.ToArray());

        norm.RunningMean.Update(seen, training);
        Assert.Equal([9f, 9f, 9f], norm.RunningMean.Value.Values.ToArray());
    }

    [Fact]
    public void ARecomputedLayerThatTakesTwoSteps_IsRunAgainOnce_NotOncePerStepItGivesAGradientTo()
    {
        var join = new JoinOfTwo();
        var network = new LayerStack([new Step(new Dense(2, 2, Draws()), "a"), new Step(join, "out", ["input", "a"])]);
        var recording = new RecordingBackend(new CpuBackend());
        var rows = Tensor.From(new Shape(3, 2), [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f]);
        var pass = Pass.Training(recording, new RandomStream(1), 0, 0).Recomputing(Recompute.Layers("out"));

        var made = network.Forward(rows, pass);
        var loss = recording.Mean(recording.Multiply(made, made));
        var gradients = recording.GradientsOf(loss, [rows, .. network.Parameters().Select(parameter => parameter.Value)]);

        Assert.Equal(2, join.Runs);
        Assert.Equal(new Shape(3, 2), gradients[rows].Shape);
    }

    [Fact]
    public void AParameterReadOnlyInsideARecomputedLayer_StillGetsItsGradient_AsWhenTheLayerIsKept()
    {
        var dense = new Dense(Tensor.From(new Shape(2, 2), [0.5f, -0.25f, 0.75f, 0.125f]), Tensor.From(new Shape(2), [0.1f, -0.1f]));
        var plain = Gradients(dense, recompute: null);
        var again = Gradients(dense, recompute: Recompute.Layers("0"));

        Assert.Equal(plain.Select(each => each.Select(BitConverter.SingleToInt32Bits)), again.Select(each => each.Select(BitConverter.SingleToInt32Bits)));
        Assert.Contains(again, each => each.Any(value => value is not 0f));
    }

    [Fact]
    public void AParameterTheRecomputedLayerNeverReads_IsSentNoGradient_NotAFailure()
    {
        var layer = new WithASpare();
        var network = new LayerStack(layer);
        var recording = new RecordingBackend(new CpuBackend());
        var rows = Tensor.From(new Shape(3, 2), [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f]);
        var pass = Pass.Training(recording, new RandomStream(1), 0, 0).Recomputing(Recompute.Layers("0"));

        var made = network.Forward(rows, pass);
        var gradients = recording.GradientsOf(recording.Mean(recording.Multiply(made, made)), [layer.Weight.Value, layer.Spare.Value]);

        Assert.All(gradients[layer.Spare.Value].Values.ToArray(), value => Assert.Equal(0f, value));
        Assert.Contains(gradients[layer.Weight.Value].Values.ToArray(), value => value is not 0f);
    }

    [Fact]
    public void TheLayersAskedFor_AreTheOnesAtThosePaths_AndNeverTheNetworkItself()
    {
        Assert.True(Recompute.Everything.Includes("encoder.self_attn"));
        Assert.False(Recompute.Everything.Includes(string.Empty));
        Assert.True(Recompute.Layers("a", "b").Includes("b"));
        Assert.False(Recompute.Layers("a", "b").Includes("c"));
        Assert.False(Recompute.Layers("a").Includes(string.Empty));
        Assert.Equal(["a", "b"], Recompute.Layers("a", "b", "a").Paths);
        Assert.Null(Recompute.Everything.Paths);
    }

    [Fact]
    public void NoLayerAskedFor_OrOneWithNoPath_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => Recompute.Layers());
        Assert.Throws<ArgumentException>(() => Recompute.Layers("a", " "));
        Assert.Throws<ArgumentException>(() => Recompute.Layers(string.Empty));
        Assert.Throws<ArgumentNullException>(() => Recompute.Layers(null!));
    }

    [Fact]
    public void WhatARecomputedLayersWorkingsHeld_IsNotHeldWhileTheRestOfThePassRuns()
    {
        var plain = LiveAfterTheForwardPass(recompute: null);
        var again = LiveAfterTheForwardPass(recompute: Recompute.Layers("0", "2", "4", "6"));

        Assert.True(again < plain, $"{again} storages are alive with the dense layers recomputed, and {plain} with them kept.");
    }

    private static List<float[]> Gradients(Dense dense, Recompute? recompute)
    {
        var network = new LayerStack(new Dense(dense.Weight.Value, dense.Bias!.Value));
        var recording = new RecordingBackend(new CpuBackend());
        var rows = Tensor.From(new Shape(3, 2), [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f]);
        var pass = Pass.Training(recording, new RandomStream(1), 0, 0).Recomputing(recompute);
        var made = network.Forward(rows, pass);
        var gradients = recording.GradientsOf(recording.Mean(recording.Multiply(made, made)), network.Parameters().Select(parameter => parameter.Value));

        return [.. network.Parameters().Select(parameter => gradients[parameter.Value].Values.ToArray())];
    }

    // The storages the engine still holds, after a collection, at the end of the forward pass of a stack of dense layers.
    private static int LiveAfterTheForwardPass(Recompute? recompute)
    {
        var engine = new NativeMemoryBackend();
        var recording = new RecordingBackend(engine);
        var draws = Draws();
        var layers = new List<Layer>();

        for (var at = 0; at < 4; at++)
        {
            layers.Add(new Dense(16, 16, draws));
            layers.Add(new Relu());
        }

        var network = new LayerStack(layers);
        var rows = Tensor.From(new Shape(32, 16), [.. Enumerable.Range(0, 512).Select(at => (float)Math.Sin(at * 0.1))]);
        var pass = Pass.Training(recording, new RandomStream(1), 0, 0).Recomputing(recompute);
        var made = network.Forward(rows, pass);

        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.KeepAlive(made);
        GC.KeepAlive(recording);

        return engine.Live;
    }

    private static Draws Draws() => new RandomStream(5).Draw("initialise:0", 0, 0);

    // A layer written outside the library that learns a matrix and carries a second parameter it never reads.
    private sealed class WithASpare : Layer
    {
        public WithASpare()
        {
            Weight = AddParameter("weight", Tensor.From(new Shape(2, 2), [0.5f, -0.25f, 0.75f, 0.125f]));
            Spare = AddParameter("spare", Tensor.From(new Shape(2), [1f, 2f]));
        }

        public Parameter Weight { get; }

        public Parameter Spare { get; }

        protected override Tensor Compute(Tensor input, Pass pass) => pass.Backend.MatMul(input, Weight.Value);
    }

    // A layer written outside the library that joins two steps, and counts the times it is run.
    private sealed class JoinOfTwo : Layer
    {
        public int Runs { get; private set; }

        internal override bool Accepts(int steps) => steps == 2;

        internal override Tensor Taking(IReadOnlyList<Tensor> inputs, Pass pass)
        {
            Runs++;

            return pass.Backend.Add(inputs[0], inputs[1]);
        }

        protected override Tensor Compute(Tensor input, Pass pass) => throw new InvalidOperationException("It joins two steps.");
    }
}
