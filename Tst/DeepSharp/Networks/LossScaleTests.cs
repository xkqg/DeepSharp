// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A run can scale its loss so that small gradients survive an engine that keeps its work in a narrow width, as PyTorch's gradient
/// scaler does: every gradient is worked back from a seed that is a power of two, so it is scaled and scaled down again without losing
/// a digit; a step whose gradients are not numbers is skipped and the scale lowered; and a run of clean steps raises it. A run that
/// scales is the run that does not, to the bit, wherever nothing overflows.
/// </summary>
public class LossScaleTests
{
    // A line the rows lie far above, so the gradient of its one weight is large: scaled by 2^16 it overflows what half holds.
    private static readonly Tensor Far = Tensor.From(new Shape(8, 1), [1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f]);

    private static readonly Tensor FarAnswers = Tensor.From(new Shape(8, 1), [1000f, 1000f, 1000f, 1000f, 1000f, 1000f, 1000f, 1000f]);

    [Fact]
    public void ASkippedStep_LeavesSlotsAdamStepsAndAveragesAsTheyWere()
    {
        var (compiled, history, checkpoint) = Run(new RoundedBackend(RoundedBackend.Width.Half), LossScale.Dynamic(), epochs: 1, new Adam(0.1), average: 0.9);
        var initial = Start().Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray()).ToArray();

        Assert.Equal(1, history.Epochs[0].Skipped);
        Assert.Equal(initial, compiled.Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray()).ToArray());
        Assert.Empty(checkpoint.State.Memory);
        Assert.Empty(checkpoint.State.Averaged!);
    }

    [Fact]
    public void ASkippedStep_HalvesTheScale_AndIsCounted()
    {
        var (_, history, checkpoint) = Run(new RoundedBackend(RoundedBackend.Width.Half), LossScale.Dynamic(), epochs: 1);

        Assert.Equal(1, history.Epochs[0].Skipped);
        Assert.Equal(32768d, checkpoint.State.Scaler!.Value.Scale);
        Assert.Equal(0, checkpoint.State.Scaler.Value.Clean);
    }

    [Fact]
    public void ACleanRunOfSteps_DoublesTheScale_EveryIntervalOfThem()
    {
        var (_, history, checkpoint) = Run(new CpuBackend(), LossScale.Dynamic(initial: 1024, growth: 2, interval: 3, backoff: 0.5), epochs: 7);

        Assert.All(history.Epochs, epoch => Assert.Equal(0, epoch.Skipped));
        Assert.Equal(4096d, checkpoint.State.Scaler!.Value.Scale);
        Assert.Equal(1, checkpoint.State.Scaler.Value.Clean);
    }

    [Fact]
    public void TheScaleChanges_OnlyAtTheStep_NeverBetweenTheBatchesOfAWindow()
    {
        var engine = new WideningBackend(new CpuBackend());
        var options = new FitOptions(seed: 3)
        {
            Epochs = 2,
            BatchSize = 2,
            StepEvery = 2,
            Backend = engine,
            LossScale = LossScale.Dynamic(initial: 1024, growth: 2, interval: 2, backoff: 0.5),
        };

        _ = Start().Fit(new TrainingData(Far, FarAnswers), validation: null, options);

        // Eight rows in batches of two are four batches an epoch, a step every two of them: the seeds of a window are one number, and
        // the number is raised once two steps are clean, at the step that makes them so.
        var seeds = engine.Filled.Where(value => value >= 1000).ToArray();

        Assert.Equal([1024f, 1024f, 1024f, 1024f, 2048f, 2048f, 2048f, 2048f], seeds);
    }

    [Fact]
    public void GradientsAreUnscaledBeforeTheyAreClipped_SoAScaledRunClipsWhatAnUnscaledOneDoes()
    {
        // A bound the unscaled gradients are far below and the scaled ones far above: clipped before they are scaled down, they would be cut.
        var scaled = Run(new CpuBackend(), LossScale.Dynamic(), epochs: 3, new Sgd(1e-5), clip: new GradientClip(1e6)).Compiled;
        var plain = Run(new CpuBackend(), loss: null, epochs: 3, new Sgd(1e-5), clip: new GradientClip(1e6)).Compiled;

        Assert.Equal(Bits(plain), Bits(scaled));
    }

    [Fact]
    public void APowerOfTwoScaledRun_IsTheUnscaledRunBitForBit_WhereverNothingOverflows()
    {
        // Plain steepest descent, which a scale that is not taken off again moves; Adam divides a scale out by itself.
        var scaled = Run(new CpuBackend(), LossScale.Dynamic(), epochs: 4, new Sgd(1e-5)).Compiled;
        var plain = Run(new CpuBackend(), loss: null, epochs: 4, new Sgd(1e-5)).Compiled;

        Assert.Equal(Bits(plain), Bits(scaled));
    }

    [Fact]
    public void ANonFiniteGradientAtAScaleOfOne_StopsTheRun_NamingTheScale()
    {
        var tooFar = Tensor.From(new Shape(8, 1), [.. Enumerable.Repeat(1e6f, 8)]);
        var refused = Assert.Throws<InvalidOperationException>(() => Run(new RoundedBackend(RoundedBackend.Width.Half), LossScale.Fixed(1), epochs: 1, answers: tooFar));

        Assert.Contains("loss scale of 1", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADynamicScale_LoweredToOneAndStillNotFinite_StopsTheRun_NamingTheScale()
    {
        var tooFar = Tensor.From(new Shape(8, 1), [.. Enumerable.Repeat(1e6f, 8)]);
        var refused = Assert.Throws<InvalidOperationException>(() => Run(new RoundedBackend(RoundedBackend.Width.Half), LossScale.Dynamic(initial: 4), epochs: 6, answers: tooFar));

        Assert.Contains("loss scale of 1", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANonFiniteGradientUnderAFixedScale_StopsTheRun_BecauseNothingIsLeftToLower()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Run(new RoundedBackend(RoundedBackend.Width.Half), LossScale.Fixed(65536), epochs: 1));

        Assert.Contains("65536", refused.Message, StringComparison.Ordinal);
        Assert.Contains("fixed", refused.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TheScalerGoesOnFromACheckpoint_AsTheRunThatNeverStopped()
    {
        var scale = LossScale.Dynamic(initial: 1024, growth: 2, interval: 2, backoff: 0.5);
        var straight = Run(new CpuBackend(), scale, epochs: 6);
        Checkpoint? half = null;

        _ = Run(new CpuBackend(), scale, epochs: 3, kept: checkpoint => half = checkpoint);

        var resumed = Run(new CpuBackend(), scale, epochs: 6, resume: half);

        Assert.Equal(Bits(straight.Compiled), Bits(resumed.Compiled));
        Assert.Equal(straight.Checkpoint.State.Scaler, resumed.Checkpoint.State.Scaler);
    }

    [Fact]
    public void ACheckpointTakenUnderAScale_GoneOnUnderAnother_OrUnderNone_IsRefused_NamingBoth()
    {
        Checkpoint? taken = null;

        _ = Run(new CpuBackend(), LossScale.Dynamic(), epochs: 1, kept: checkpoint => taken = checkpoint);

        var other = Assert.Throws<ArgumentException>(() => Run(new CpuBackend(), LossScale.Dynamic(interval: 10), epochs: 2, resume: taken));
        var none = Assert.Throws<ArgumentException>(() => Run(new CpuBackend(), loss: null, epochs: 2, resume: taken));

        Assert.Contains("loss scale", other.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("loss scale", none.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void AGradientThatHalfLosesUnscaled_SurvivesScaled()
    {
        // The rows are one millionth and the answers a hundredth, the line starting from nothing: the weight's gradient is about two
        // hundred-millionths, below the least number half holds, so unscaled it is nought and the weight stays; scaled by 2^16 it is
        // about a thousandth, and moves.
        var rows = Tensor.From(new Shape(4, 1), [1e-6f, 1e-6f, 1e-6f, 1e-6f]);
        var answers = Tensor.From(new Shape(4, 1), [1e-2f, 1e-2f, 1e-2f, 1e-2f]);

        float Learned(LossScale? scale)
        {
            var dense = new Dense(Tensor.From(new Shape(1, 1), [0.3f]), Tensor.From(new Shape(1), [0f]));
            var compiled = new LayerStack(dense).Compile(new Sgd(1e6), new MeanSquaredError());

            _ = compiled.Fit(new TrainingData(rows, answers), validation: null, new FitOptions(seed: 3) { Epochs = 1, BatchSize = 4, Backend = new RoundedBackend(RoundedBackend.Width.Half), LossScale = scale });

            return dense.Weight.Value.Values[0];
        }

        Assert.Equal(0.3f, Learned(null));
        Assert.True(Math.Abs(Learned(LossScale.Fixed(65536)) - 0.3f) > 0.005);
    }

    [Fact]
    public void ALossScale_IsAPowerOfTwo_AndItsGrowthAndBackoffKeepItOne()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => LossScale.Fixed(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => LossScale.Dynamic(initial: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => LossScale.Dynamic(growth: 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => LossScale.Dynamic(growth: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => LossScale.Dynamic(backoff: 0.3));
        Assert.Throws<ArgumentOutOfRangeException>(() => LossScale.Dynamic(backoff: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => LossScale.Dynamic(interval: 0));
        Assert.Equal(65536d, LossScale.Dynamic().Initial);
    }

    private static CompiledNetwork Start(Optimizer? optimizer = null) =>
        new LayerStack(new Dense(1, 1, new RandomStream(5).Draw("initialise:0", 0, 0))).Compile(optimizer ?? new Sgd(0.01), new MeanSquaredError());

    private static IEnumerable<int[]> Bits(CompiledNetwork compiled) =>
        compiled.Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray());

    // One run of the line, one batch an epoch unless said, whose last checkpoint is returned beside the network and what it did.
    private static (CompiledNetwork Compiled, History History, Checkpoint Checkpoint) Run(
        ITensorBackend engine,
        LossScale? loss,
        int epochs,
        Optimizer? optimizer = null,
        double? average = null,
        GradientClip? clip = null,
        Action<Checkpoint>? kept = null,
        Checkpoint? resume = null,
        Tensor? answers = null)
    {
        Checkpoint? last = null;
        var compiled = Start(optimizer);
        var history = compiled.Fit(new TrainingData(Far, answers ?? FarAnswers), validation: null, new FitOptions(seed: 3)
        {
            Epochs = epochs,
            BatchSize = 8,
            Backend = engine,
            LossScale = loss,
            Average = average,
            GradientClip = clip,
            ResumeFrom = resume,
            Checkpoints = new Checkpoints(checkpoint =>
            {
                last = checkpoint;
                kept?.Invoke(checkpoint);
            }),
        });

        return (compiled, history, last!);
    }
}
