// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A network is layers holding layers, and every number it learns sits in a slot with a path: the layer's place under
/// the network, then the slot's own name — <c>0.weight</c> for the first layer of a stack, <c>hidden.weight</c> for a
/// layer a network written as code calls hidden. What a layer does is said by the pass it is handed, never remembered
/// by the layer.
/// </summary>
public class LayerTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

    [Fact]
    public void ALayerStack_NamesItsLayersByTheirPlace_AndEverySlotByItsPath()
    {
        var stack = new LayerStack(new Weighted(2f, 3), new Plain(), new Weighted(3f, 3), new Tally());

        Assert.Equal(["0.weight", "2.weight", "3.seen"], stack.Slots().Select(slot => slot.Path));
    }

    [Fact]
    public void ANetworkWrittenAsCode_NamesItsSlotsByTheNamesItGaveItsLayers()
    {
        var network = new TwoLayers();

        Assert.Equal(["hidden.weight", "tally.seen", "out.weight"], network.Slots().Select(slot => slot.Path));
    }

    [Fact]
    public void ALayersOwnSlots_ComeBeforeTheSlotsOfTheLayersItHolds()
    {
        var network = new WithOwnSlot();

        Assert.Equal(["scale", "inner.weight"], network.Slots().Select(slot => slot.Path));
    }

    [Fact]
    public void TheParameters_AreTheSlotsThatLearn_AndNotTheRunningStatistics()
    {
        var stack = new LayerStack(new Weighted(2f, 3), new Tally(), new Weighted(3f, 3));

        var parameters = stack.Parameters().ToArray();

        Assert.Equal(2, parameters.Length);
        Assert.All(parameters, parameter => Assert.Equal("weight", parameter.Name));
    }

    [Fact]
    public void ALayerStack_RunsItsLayersInTheirOrder()
    {
        var stack = new LayerStack(new Weighted(2f, 3), new Plain(), new Weighted(3f, 3));

        var output = stack.Forward(Tensor.From(new Shape(2, 3), [1f, 2f, 3f, 4f, 5f, 6f]), Pass.Evaluation(_backend));

        Assert.Equal<float[]>([6f, 12f, 18f, 24f, 30f, 36f], output.Values.ToArray());
    }

    [Fact]
    public void AParameter_RefusesATensorOfAnotherShape()
    {
        var parameter = new Weighted(2f, 3).Weight;

        var wrong = Assert.Throws<ArgumentException>(() => parameter.Replace(Tensor.Zeros(new Shape(4))));

        Assert.Contains("weight", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("3", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("4", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASlot_KeepsWhoItIs_WhenItsTensorIsReplaced()
    {
        var stack = new LayerStack(new Weighted(2f, 3));
        var parameter = stack.Parameters().Single();
        var replacement = Tensor.From(new Shape(3), [5f, 6f, 7f]);

        parameter.Replace(replacement);

        Assert.Same(parameter, stack.Parameters().Single());
        Assert.Same(replacement, parameter.Value);
    }

    [Fact]
    public void ARunningStatistic_RefusesAnEvaluationPass()
    {
        var tally = new Tally();

        Assert.Throws<InvalidOperationException>(
            () => tally.Seen.Update(Tensor.From(new Shape(1), [1f]), Pass.Evaluation(_backend)));
    }

    [Fact]
    public void ATrainingPass_MovesARunningStatistic_AndAnEvaluationPassLeavesItAsItWas()
    {
        var tally = new Tally();
        var rows = Tensor.Zeros(new Shape(2, 3));

        tally.Forward(rows, Pass.Training(_backend, new RandomStream(7), epoch: 0, step: 0));
        var afterTraining = tally.Seen.Value;
        tally.Forward(rows, Pass.Evaluation(_backend));

        Assert.Equal(1f, afterTraining.Values[0]);
        Assert.Same(afterTraining, tally.Seen.Value);
    }

    [Fact]
    public void ARunningStatistic_RefusesATensorOfAnotherShape()
    {
        var tally = new Tally();

        Assert.Throws<ArgumentException>(
            () => tally.Seen.Update(Tensor.Zeros(new Shape(2)), Pass.Training(_backend, new RandomStream(7), epoch: 0, step: 0)));
    }

    [Fact]
    public void ASnapshot_BringsBackEverySlot_TheParametersAndTheRunningStatisticsAlike()
    {
        var network = new TwoLayers();
        var snapshot = network.Snapshot();
        var before = network.Slots().Select(slot => slot.Slot.Value).ToArray();

        network.Parameters().First().Replace(Tensor.From(new Shape(3), [9f, 9f, 9f]));
        network.Forward(Tensor.Zeros(new Shape(1, 3)), Pass.Training(_backend, new RandomStream(7), epoch: 0, step: 0));
        network.Restore(snapshot);

        Assert.Equal(before, network.Slots().Select(slot => slot.Slot.Value));
    }

    [Fact]
    public void ASnapshotOfAnotherNetwork_IsRefused()
    {
        var snapshot = new TwoLayers().Snapshot();

        Assert.Throws<ArgumentException>(() => new TwoLayers().Restore(snapshot));
    }

    [Theory]
    [InlineData("")]
    [InlineData("a.b")]
    [InlineData(" ")]
    public void ASlotOrALayerWhoseNameCannotStandInAPath_IsRefused(string name)
    {
        Assert.Throws<ArgumentException>(() => new Named(name, asLayer: false));
        Assert.Throws<ArgumentException>(() => new Named(name, asLayer: true));
    }

    [Fact]
    public void TwoSlotsOrLayersOfOneName_AreRefused()
    {
        Assert.Throws<ArgumentException>(() => new Twice());
    }

    [Fact]
    public void ALayerAlreadyHeldByAnother_IsRefused()
    {
        var shared = new Weighted(2f, 3);
        _ = new LayerStack(shared);

        Assert.Throws<ArgumentException>(() => new LayerStack(shared));
    }

    [Fact]
    public void ALayerThatHoldsTheOneItWouldBeAddedTo_OrIsIt_IsRefused()
    {
        var holder = new Holder();
        var stack = new LayerStack(holder);
        var alone = new Holder();

        Assert.Contains("holds the one it would be added to", Assert.Throws<ArgumentException>(() => holder.Hold(stack)).Message, StringComparison.Ordinal);
        Assert.Contains("holds the one it would be added to", Assert.Throws<ArgumentException>(() => alone.Hold(alone)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALayerStackOfNothing_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new LayerStack());
    }

    [Fact]
    public void AForwardPass_NeedsAnInputAndAPass()
    {
        var layer = new Plain();

        Assert.Throws<ArgumentNullException>(() => layer.Forward(null!, Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentNullException>(() => layer.Forward(Tensor.Zeros(new Shape(1)), null!));
    }

    [Fact]
    public void APass_SaysWhatItIsFor_AndWhereInTheRunItStands()
    {
        var stream = new RandomStream(42);

        var training = Pass.Training(_backend, stream, epoch: 3, step: 7);
        var evaluation = Pass.Evaluation(_backend);

        Assert.Equal(PassMode.Training, training.Mode);
        Assert.Same(_backend, training.Backend);
        Assert.Equal(3, training.Epoch);
        Assert.Equal(7, training.Step);
        Assert.Equal(PassMode.Evaluation, evaluation.Mode);
        Assert.Same(_backend, evaluation.Backend);
    }

    [Fact]
    public void ATrainingPassesDraws_AreTheStreamsDrawsForTheSamePurposeAtTheSamePlace()
    {
        var stream = new RandomStream(42);

        var pass = Pass.Training(_backend, stream, epoch: 3, step: 7);

        Assert.Equal(stream.Draw("dropout:2", 3, 7).NextSingle(), pass.Draws("dropout:2").NextSingle());
    }

    [Fact]
    public void AnEvaluationPass_DrawsNothing()
    {
        Assert.Throws<InvalidOperationException>(() => Pass.Evaluation(_backend).Draws("dropout:2"));
    }

    [Fact]
    public void APass_NeedsABackend_AndATrainingPassAStreamAndAPlaceInTheRun()
    {
        Assert.Throws<ArgumentNullException>(() => Pass.Evaluation(null!));
        Assert.Throws<ArgumentNullException>(() => Pass.Training(null!, new RandomStream(1), 0, 0));
        Assert.Throws<ArgumentNullException>(() => Pass.Training(_backend, null!, 0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pass.Training(_backend, new RandomStream(1), -1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pass.Training(_backend, new RandomStream(1), 0, -1));
    }

    [Fact]
    public void ALayerSaysWhereItStandsInItsNetwork()
    {
        var hidden = new Weighted(2f, 3);
        var stack = new LayerStack(new Plain(), hidden);

        Assert.Equal("1", hidden.Path);
        Assert.Equal(string.Empty, stack.Path);
        Assert.Equal(string.Empty, new Plain().Path);
    }

    // Multiplies every row by one learned row of weights.
    private sealed class Weighted : Layer
    {
        public Weighted(float value, int width) =>
            Weight = AddParameter("weight", Tensor.From(new Shape(width), [.. Enumerable.Repeat(value, width)]));

        public Parameter Weight { get; }

        protected override Tensor Compute(Tensor input, Pass pass) =>
            pass.Backend.Multiply(input, pass.Backend.AddRow(pass.Backend.Fill(input.Shape, 0f), Weight.Value));
    }

    // Passes its input on, and counts in a running statistic how many training passes it saw.
    private sealed class Tally : Layer
    {
        public Tally() => Seen = AddRunningStatistic("seen", Tensor.Zeros(new Shape(1)));

        public RunningStatistic Seen { get; }

        protected override Tensor Compute(Tensor input, Pass pass)
        {
            if (pass.Mode == PassMode.Training)
            {
                Seen.Update(pass.Backend.Add(Seen.Value, pass.Backend.Fill(new Shape(1), 1f)), pass);
            }

            return input;
        }
    }

    private sealed class Plain : Layer
    {
        protected override Tensor Compute(Tensor input, Pass pass) => input;
    }

    private sealed class TwoLayers : Network
    {
        private readonly Weighted _hidden;
        private readonly Tally _tally;
        private readonly Weighted _out;

        public TwoLayers()
        {
            _hidden = AddLayer("hidden", new Weighted(2f, 3));
            _tally = AddLayer("tally", new Tally());
            _out = AddLayer("out", new Weighted(3f, 3));
        }

        protected override Tensor Compute(Tensor input, Pass pass) =>
            _out.Forward(_tally.Forward(_hidden.Forward(input, pass), pass), pass);
    }

    private sealed class WithOwnSlot : Network
    {
        private readonly Weighted _inner;
        private readonly Parameter _scale;

        public WithOwnSlot()
        {
            _inner = AddLayer("inner", new Weighted(2f, 3));
            _scale = AddParameter("scale", Tensor.From(new Shape(3), [1f, 1f, 1f]));
        }

        protected override Tensor Compute(Tensor input, Pass pass) =>
            pass.Backend.Multiply(_inner.Forward(input, pass), pass.Backend.AddRow(pass.Backend.Fill(input.Shape, 0f), _scale.Value));
    }

    private sealed class Named : Layer
    {
        public Named(string name, bool asLayer)
        {
            if (asLayer)
            {
                AddLayer(name, new Plain());
            }
            else
            {
                AddParameter(name, Tensor.Zeros(new Shape(1)));
            }
        }

        protected override Tensor Compute(Tensor input, Pass pass) => input;
    }

    // A layer that takes another in after it was made.
    private sealed class Holder : Layer
    {
        public void Hold(Layer layer) => AddLayer("held", layer);

        protected override Tensor Compute(Tensor input, Pass pass) => input;
    }

    private sealed class Twice : Layer
    {
        public Twice()
        {
            AddParameter("weight", Tensor.Zeros(new Shape(1)));
            AddLayer("weight", new Plain());
        }

        protected override Tensor Compute(Tensor input, Pass pass) => input;
    }
}
