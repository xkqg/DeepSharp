// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A rate that follows how the training is going: lowered by a factor once the training loss has stopped falling for as many
/// epochs as the schedule waits, as PyTorch's <c>ReduceLROnPlateau</c> lowers it — the same rate at every epoch for the same
/// losses, from <c>Fixtures/plateau-pytorch.py</c>. It never listens to the validation rows, which are only there to choose
/// between networks, and the epochs it has gone through are in a checkpoint, so a run goes on from one at the rate it would have
/// taken.
/// </summary>
public class TrainingPlateauTests
{
    private static readonly JsonElement Fixture = JsonDocument
        .Parse(File.ReadAllText(Path.Join(Repository.Root, "Tst", "DeepSharp", "Networks", "Fixtures", "plateau-pytorch.json")))
        .RootElement;

    public static TheoryData<string> Walks { get; } = [.. Fixture.GetProperty("walks").EnumerateArray().Select(walk => walk.GetProperty("name").GetString()!)];

    [Theory]
    [MemberData(nameof(Walks))]
    public void TheRateOfEveryEpoch_IsPyTorchsReduceLROnPlateau_ForTheLossesBeforeIt(string name)
    {
        var walk = Fixture.GetProperty("walks").EnumerateArray().Single(each => each.GetProperty("name").GetString() == name);
        var schedule = new TrainingPlateau(walk.GetProperty("patience").GetInt32(), walk.GetProperty("factor").GetDouble())
        {
            Threshold = walk.GetProperty("threshold").GetDouble(),
            Cooldown = walk.GetProperty("cooldown").GetInt32(),
            Minimum = walk.GetProperty("min_lr").GetDouble(),
        };
        var losses = walk.GetProperty("losses").EnumerateArray().Select(each => each.GetDouble()).ToArray();
        var rates = walk.GetProperty("rates").EnumerateArray().Select(each => each.GetDouble()).ToArray();

        for (var epoch = 0; epoch < rates.Length; epoch++)
        {
            Assert.Equal(rates[epoch], schedule.RateAfter(losses[..epoch], walk.GetProperty("initial").GetDouble()));
        }
    }

    [Fact]
    public void ThePlateau_FallsEveryTimeTheLossHasWaitedItsPatienceWithoutFalling()
    {
        var schedule = new TrainingPlateau(patience: 2, factor: 0.5);

        Assert.Equal(0.1, schedule.RateAfter([], 0.1));
        Assert.Equal(0.1, schedule.RateAfter([1.0, 0.9, 0.9, 0.9], 0.1));
        Assert.Equal(0.05, schedule.RateAfter([1.0, 0.9, 0.9, 0.9, 0.9], 0.1));
        Assert.Equal(0.05, schedule.RateAfter([1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], 0.1));
        Assert.Equal(0.025, schedule.RateAfter([1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], 0.1));
    }

    [Fact]
    public void AScheduleThatFollowsTheLoss_CannotSayItsRateFromTheEpochAlone()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => new TrainingPlateau().RateAt(3, 0.1));

        Assert.Contains("losses", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AClosedFormSchedule_TakesTheSameRateWhetherItIsAskedByTheEpochOrByTheLossesBeforeIt()
    {
        LearningRateSchedule[] schedules = [new ConstantRate(), new StepDecay(3, 0.5), new ExponentialDecay(0.9), new CosineDecay(8, 0.001), new LinearWarmup(3)];

        foreach (var schedule in schedules)
        {
            for (var epoch = 0; epoch < 12; epoch++)
            {
                Assert.Equal(schedule.RateAt(epoch, 0.1), schedule.RateAfter([.. Enumerable.Repeat(0.5, epoch)], 0.1));
            }
        }
    }

    [Fact]
    public void ALossThatIsNotAFiniteNumber_IsRefused_ForNoRateCouldFollowIt()
    {
        Assert.Throws<ArgumentException>(() => new TrainingPlateau().RateAfter([1.0, double.NaN], 0.1));
        Assert.Throws<ArgumentException>(() => new TrainingPlateau().RateAfter([double.PositiveInfinity], 0.1));
        Assert.Throws<ArgumentNullException>(() => new TrainingPlateau().RateAfter(null!, 0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingPlateau().RateAfter([], 0));
    }

    [Theory]
    [InlineData(-1, 0.1)]
    [InlineData(2, 0.0)]
    [InlineData(2, 1.0)]
    [InlineData(2, 1.5)]
    [InlineData(2, double.NaN)]
    public void APatienceBelowNothing_OrAFactorThatDoesNotLowerTheRate_IsRefused(int patience, double factor)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingPlateau(patience, factor));
    }

    [Fact]
    public void AThresholdACooldownAndALeastRateOutOfRange_AreRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingPlateau { Threshold = -0.1 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingPlateau { Threshold = 1 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingPlateau { Cooldown = -1 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingPlateau { Minimum = -0.5 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingPlateau { Minimum = double.NaN });
    }

    [Fact]
    public void AWarmUpThatHandsOverToAPlateau_FollowsTheLossesOfTheEpochsAfterIt_AsPyTorchsSequentialLRCannot()
    {
        // PyTorch's SequentialLR refuses a ReduceLROnPlateau, so this composition is this library's own: the warm-up's epochs
        // are the first of them, and the plateau counts the losses of the epochs after.
        var schedule = new LinearWarmup(3, 0.5, then: new TrainingPlateau(patience: 1, factor: 0.5));
        double[] losses = [9, 9, 9, 1.0, 0.9, 0.9, 0.9, 0.9];

        Assert.Contains("SequentialLR does not support", Fixture.GetProperty("sequential").GetString(), StringComparison.Ordinal);
        Assert.Equal(0.1 * 0.5, schedule.RateAfter(losses[..0], 0.1), 1e-15);
        Assert.Equal(0.1 * (0.5 + (0.5 * 2 / 3)), schedule.RateAfter(losses[..2], 0.1), 1e-15);
        Assert.Equal(0.1, schedule.RateAfter(losses[..3], 0.1));
        Assert.Equal(0.1, schedule.RateAfter(losses[..5], 0.1));
        Assert.Equal(0.1, schedule.RateAfter(losses[..6], 0.1));
        Assert.Equal(0.05, schedule.RateAfter(losses[..7], 0.1));
    }

    [Fact]
    public void ATrainingRunsEveryEpochsRate_IsTheScheduleAskedWithTheTrainingLossesOfTheEpochsBefore()
    {
        var schedule = new TrainingPlateau(patience: 1, factor: 0.5) { Threshold = 0.3 };
        var history = Compiled(schedule).Fit(Rows(), validation: null, Options(12));

        for (var epoch = 0; epoch < history.Epochs.Count; epoch++)
        {
            Assert.Equal(schedule.RateAfter([.. history.Epochs.Take(epoch).Select(each => each.Loss)], 0.02), history.Epochs[epoch].LearningRate);
        }

        Assert.True(history.Epochs[^1].LearningRate < 0.02, "The rows were meant to plateau, so the rate must have fallen.");
    }

    [Fact]
    public void ARunGoneOnFromACheckpoint_TakesTheRatesItWouldHaveTaken_AndEndsOnTheSameBits()
    {
        var straight = Compiled(new TrainingPlateau(1, 0.5) { Threshold = 0.3 });
        var stopped = Compiled(new TrainingPlateau(1, 0.5) { Threshold = 0.3 });
        var resumed = Compiled(new TrainingPlateau(1, 0.5) { Threshold = 0.3 });
        var kept = new List<Checkpoint>();
        using var source = new CancellationTokenSource();

        var whole = straight.Fit(Rows(), validation: null, Options(12));

        Assert.Throws<OperationCanceledException>(() => stopped.Fit(
            Rows(), validation: null, Options(12, new Checkpoints(kept.Add), epoch => Cancel(source, epoch.Number == 5), source: source)));

        var rest = resumed.Fit(Rows(), validation: null, Options(12, from: kept[^1]));

        Assert.Equal(whole.Epochs, rest.Epochs);
        Assert.Equal(Bits(straight), Bits(resumed));
    }

    [Fact]
    public void ARunGoneOnUnderAnotherSchedule_OrAnotherRate_IsRefused_NamingTheEpochItsRateDiffers()
    {
        var kept = new List<Checkpoint>();

        Compiled(new TrainingPlateau(1, 0.5) { Threshold = 0.3 }).Fit(Rows(), validation: null, Options(12, new Checkpoints(kept.Add)));

        var otherSchedule = Assert.Throws<ArgumentException>(() => Compiled(new ConstantRate()).Fit(Rows(), validation: null, Options(14, from: kept[^1])));
        var otherRate = Assert.Throws<ArgumentException>(
            () => Compiled(new TrainingPlateau(1, 0.5) { Threshold = 0.3 }, rate: 0.05).Fit(Rows(), validation: null, Options(14, from: kept[^1])));

        Assert.Contains("epoch", otherSchedule.Message, StringComparison.Ordinal);
        Assert.Contains("rate", otherSchedule.Message, StringComparison.Ordinal);
        Assert.Contains("epoch 1", otherRate.Message, StringComparison.Ordinal);
    }

    private static void Cancel(CancellationTokenSource source, bool now)
    {
        if (now)
        {
            source.Cancel();
        }
    }

    private static CompiledNetwork Compiled(LearningRateSchedule schedule, double rate = 0.02) =>
        new Sequential().Dense(4).Tanh().Dense(1).Compile(new Adam(rate), new MeanSquaredError(), schedule);

    private static FitOptions Options(int epochs, Checkpoints? checkpoints = null, Action<Epoch>? onEpoch = null, Checkpoint? from = null, CancellationTokenSource? source = null) =>
        new(seed: 5) { Epochs = epochs, BatchSize = 16, Checkpoints = checkpoints, Cancellation = source?.Token ?? TestContext.Current.CancellationToken, OnEpoch = onEpoch, ResumeFrom = from };

    private static TrainingData Rows()
    {
        var features = new float[96 * 3];
        var answers = new float[96];

        for (var row = 0; row < 96; row++)
        {
            for (var column = 0; column < 3; column++)
            {
                features[(row * 3) + column] = (float)Math.Sin((row * 3.0) + column + 1);
            }

            answers[row] = 2 * features[row * 3];
        }

        return new TrainingData(Tensor.From(new Shape(96, 3), features), Tensor.From(new Shape(96, 1), answers));
    }

    private static float[][] Bits(CompiledNetwork compiled) => [.. compiled.Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray())];
}
