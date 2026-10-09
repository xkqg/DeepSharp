// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using DeepSharp.Tests.Networks;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// The loop that trains a compiled network: the training rows shuffled afresh every epoch and taken a batch at a time,
/// each batch through a recording pass of its own and an optimizer's step, the validation rows looked at once an epoch and
/// never trained on, early stopping as Keras has it, and a checkpoint that a run resumes from as if it had never stopped —
/// under the seed, the batches and the early stopping it was taken under, on the engine it was taken on, and refused under
/// any other.
/// </summary>
public abstract class LoopContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    [Fact]
    public void SixHundredAndTwentyThreeRows_MakeTwentyBatchesOfThirtyTwo_TheLastHoldingFifteen()
    {
        var watcher = new Watcher();
        var network = new LayerStack(watcher, new Dense(2, 1, Draws()));

        network.Compile(new Sgd(0.01), new MeanSquaredError()).Fit(Rows(623), validation: null, new FitOptions(seed: 7) { Backend = _backend });

        Assert.Equal([.. Enumerable.Repeat(32, 19), 15], watcher.Batches.Select(batch => batch.Rows));
    }

    [Fact]
    public void EveryEpoch_ShufflesTheTrainingRowsAfresh_AndNoValidationRowEverReachesATrainingPass()
    {
        var watcher = new Watcher();
        var network = new LayerStack(watcher, new Dense(2, 1, Draws()));

        network.Compile(new Sgd(0.01), new MeanSquaredError())
            .Fit(Rows(100), Rows(20, from: 100), new FitOptions(seed: 7) { Backend = _backend, Epochs = 2, BatchSize = 16 });

        var first = watcher.Batches.Where(batch => batch.Epoch == 0).SelectMany(batch => batch.Marks).ToArray();
        var second = watcher.Batches.Where(batch => batch.Epoch == 1).SelectMany(batch => batch.Marks).ToArray();

        Assert.Equal(Enumerable.Range(0, 100), first.Order());
        Assert.Equal(Enumerable.Range(0, 100), second.Order());
        Assert.NotEqual(first, second);
    }

    [Fact]
    public void EveryBatch_IsWorkedOutThroughARecorderOfItsOwn()
    {
        var watcher = new Watcher();
        var network = new LayerStack(watcher, new Dense(2, 1, Draws()));

        network.Compile(new Sgd(0.01), new MeanSquaredError()).Fit(Rows(100), validation: null, new FitOptions(seed: 7) { Backend = _backend, Epochs = 2 });

        Assert.All(watcher.Batches, batch => Assert.IsType<RecordingBackend>(batch.Backend));
        Assert.Equal(watcher.Batches.Count, watcher.Batches.Select(batch => batch.Backend).Distinct(ReferenceEqualityComparer.Instance).Count());
    }

    [Fact]
    public void ARunOfAConvolution_AsksItsEngineToFoldNothingBack_ForNobodyAsksHowTheImagesMovedTheLoss()
    {
        var engine = new NotingBackend(_backend);
        var stream = new RandomStream(7);
        var network = new LayerStack(
            new Conv2D(1, 2, new Window(3, 3), stream.Draw("initialise:0", 0, 0)), new Relu(), new Flatten(), new Dense(18, 1, stream.Draw("initialise:3", 0, 0)));

        network.Compile(new Adam(0.01), new MeanSquaredError())
            .Fit(Images(16), validation: null, new FitOptions(seed: 7) { Backend = engine, Epochs = 2, BatchSize = 8 });

        // Two epochs of two batches: every batch's images unfolded on the way forward, and nothing folded back onto them.
        Assert.Equal(4, engine.Count(nameof(ITensorBackend.Unfold)));
        Assert.Equal(0, engine.Count(nameof(ITensorBackend.Fold)));
    }

    [Fact]
    public void TheValidationRows_NeverMoveARunningStatistic()
    {
        var watched = Normalised();
        var unwatched = Normalised();

        watched.Network.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Rows(64), Rows(40, from: 64), new FitOptions(seed: 7) { Backend = _backend, Epochs = 3 });
        unwatched.Network.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Rows(64), validation: null, new FitOptions(seed: 7) { Backend = _backend, Epochs = 3 });

        Assert.Equal(unwatched.Norm.RunningMean.Value.Values.ToArray(), watched.Norm.RunningMean.Value.Values.ToArray());
        Assert.Equal(unwatched.Norm.RunningVariance.Value.Values.ToArray(), watched.Norm.RunningVariance.Value.Values.ToArray());
    }

    [Fact]
    public void TheHistory_HasARowPerEpoch_WithItsRate_AndSaysTheSeedAndTheBestEpoch()
    {
        var network = new LayerStack(new Dense(2, 1, Draws()));

        var history = network.Compile(new Sgd(0.1), new MeanSquaredError(), new StepDecay(every: 1, factor: 0.5))
            .Fit(Rows(64), Rows(16, from: 64), new FitOptions(seed: 11) { Backend = _backend, Epochs = 3 });

        Assert.Equal(11, history.Seed);
        Assert.Equal([0, 1, 2], history.Epochs.Select(epoch => epoch.Number));
        Assert.Equal([0.1, 0.05, 0.025], history.Epochs.Select(epoch => epoch.LearningRate));
        Assert.All(history.Epochs, epoch => Assert.True(double.IsFinite(epoch.Loss) && epoch.ValidationLoss is not null));
        Assert.Equal(history.Epochs.MinBy(epoch => epoch.ValidationLoss)!.Number, history.BestEpoch);
        Assert.Equal(Stopping.AllEpochsRan, history.Stopped);
    }

    [Fact]
    public void WithoutValidationRows_AnEpochHasNoValidationLoss_AndThereIsNoBestEpoch()
    {
        var history = new LayerStack(new Dense(2, 1, Draws()))
            .Compile(new Sgd(0.1), new MeanSquaredError())
            .Fit(Rows(64), validation: null, new FitOptions(seed: 11) { Backend = _backend, Epochs = 2 });

        Assert.All(history.Epochs, epoch => Assert.Null(epoch.ValidationLoss));
        Assert.Null(history.BestEpoch);
    }

    [Fact]
    public void EarlyStoppingByKerassDefaults_StopsAtTheFirstEpochThatDoesNotImprove()
    {
        // The training rows teach twice the input and the validation rows want minus twice it, so every epoch that learns the
        // one is worse at the other: the second epoch does not improve, and with no patience the run stops there.
        var history = new LayerStack(new Dense(1, 1, Draws()))
            .Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), Sloped(16, -2), new FitOptions(seed: 3) { Backend = _backend, Epochs = 10, EarlyStopping = new EarlyStopping() });

        Assert.Equal(2, history.Epochs.Count);
        Assert.Equal(0, history.BestEpoch);
        Assert.Equal(Stopping.NoLongerImproving, history.Stopped);
    }

    [Fact]
    public void EarlyStopping_WaitsAsManyEpochsAsItsPatience_BeforeItStops()
    {
        var history = new LayerStack(new Dense(1, 1, Draws()))
            .Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), Sloped(16, -2), new FitOptions(seed: 3) { Backend = _backend, Epochs = 10, EarlyStopping = new EarlyStopping { Patience = 3 } });

        Assert.Equal(4, history.Epochs.Count);
    }

    [Fact]
    public void AFallSmallerThanTheLeastThatCounts_IsNoImprovement()
    {
        // Both parts want twice the input, so the validation loss falls every epoch, by less and less; with a least fall that
        // counts, the run stops once the fall is smaller than it.
        var history = new LayerStack(new Dense(1, 1, Draws()))
            .Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), Sloped(16, 2), new FitOptions(seed: 3) { Backend = _backend, Epochs = 50, EarlyStopping = new EarlyStopping { MinDelta = 0.01 } });

        Assert.True(history.Epochs.Count < 50);
        Assert.Equal(Stopping.NoLongerImproving, history.Stopped);
    }

    [Fact]
    public void RestoringTheBestEpoch_BringsBackEverySlot_TheRunningStatisticsToo()
    {
        var stopped = Normalised();
        var once = Normalised();
        var options = new FitOptions(seed: 3) { Backend = _backend, Epochs = 10, EarlyStopping = new EarlyStopping { RestoreBest = true } };

        var history = stopped.Network.Compile(new Sgd(0.05), new MeanSquaredError()).Fit(Sloped(64, 2, width: 2), Sloped(16, -2, width: 2), options);
        once.Network.Compile(new Sgd(0.05), new MeanSquaredError()).Fit(Sloped(64, 2, width: 2), Sloped(16, -2, width: 2), new FitOptions(seed: 3) { Backend = _backend });

        Assert.Equal(0, history.BestEpoch);
        Assert.True(history.Epochs.Count > 1);
        Assert.Equal(
            once.Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray()),
            stopped.Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
    }

    [Fact]
    public void ALossThatIsNotAFiniteNumber_IsRefused_NamingTheEpochAndTheBatch()
    {
        var huge = new TrainingData(Tensor.From(new Shape(4, 1), [1f, 1f, 1f, 1f]), Tensor.From(new Shape(4, 1), [1e30f, 1e30f, 1e30f, 1e30f]));

        var wrong = Assert.Throws<InvalidOperationException>(
            () => new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new MeanSquaredError()).Fit(huge, validation: null, new FitOptions(seed: 1) { Backend = _backend }));

        Assert.Contains("epoch 1", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("batch 1", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AValidationLossThatIsNotAFiniteNumber_IsRefused_NamingTheEpoch_ForNoEpochCouldBeJudgedByIt()
    {
        var train = new TrainingData(Tensor.From(new Shape(2, 1), [0.5f, -0.5f]), Tensor.From(new Shape(2, 1), [1f, -1f]));
        var huge = new TrainingData(Tensor.From(new Shape(1, 1), [3e38f]), Tensor.From(new Shape(1, 1), [0f]));
        var network = new LayerStack(new Dense(Tensor.From(new Shape(1, 1), [2f]), Tensor.Zeros(new Shape(1))));

        var wrong = Assert.Throws<InvalidOperationException>(
            () => network.Compile(new Sgd(), new MeanSquaredError()).Fit(train, huge, new FitOptions(seed: 1) { Backend = _backend }));

        Assert.Contains("validation loss of epoch 1", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("not a finite number", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpointOfAnotherNetwork_IsRefused_ForItsSlotsAreNotThisOnes()
    {
        var kept = new List<Checkpoint>();
        new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 1) { Backend = _backend, Checkpoints = new Checkpoints(kept.Add) });

        var wrong = Assert.Throws<ArgumentException>(() => new LayerStack(new Dense(1, 2, Draws()), new Dense(2, 1, Draws())).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 1) { Backend = _backend, Epochs = 2, ResumeFrom = kept[0] }));

        Assert.Contains("another network", wrong.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => new Checkpoints(null!));

        // As many slots, under other paths, are other slots too.
        Assert.Contains("another network", Assert.Throws<ArgumentException>(() => new LayerStack(new Relu(), new Dense(1, 1, Draws())).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 1) { Backend = _backend, Epochs = 2, ResumeFrom = kept[0] })).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpointWhoseSlotsHaveOtherShapes_IsRefused_BeforeAnySlotIsReplaced()
    {
        var kept = new List<Checkpoint>();
        new LayerStack(new Dense(2, 1, Draws())).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Rows(8, width: 2), validation: null, new FitOptions(seed: 1) { Backend = _backend, Checkpoints = new Checkpoints(kept.Add) });
        var other = new LayerStack(new Dense(3, 1, Draws()));
        var before = other.Slots().Select(slot => slot.Slot.Value).ToArray();

        var wrong = Assert.Throws<ArgumentException>(() => other.Compile(new Sgd(), new MeanSquaredError())
            .Fit(Rows(8, width: 3), validation: null, new FitOptions(seed: 1) { Backend = _backend, Epochs = 2, ResumeFrom = kept[0] }));

        Assert.Contains("another network", wrong.Message, StringComparison.Ordinal);
        Assert.Equal(before, other.Slots().Select(slot => slot.Slot.Value));
    }

    [Fact]
    public void ARunJudgedByNoValidationRows_GoesOnJudgedByNone_EvenFromACheckpointThatWasJudged()
    {
        var kept = new List<Checkpoint>();
        var first = new LayerStack(new Dense(1, 1, Draws()));
        first.Compile(new Sgd(), new MeanSquaredError()).Fit(Sloped(8, 2), Sloped(4, 2), new FitOptions(seed: 1) { Backend = _backend, Checkpoints = new Checkpoints(kept.Add) });

        var history = first.Compile(new Sgd(), new MeanSquaredError()).Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 1) { Backend = _backend, Epochs = 2, ResumeFrom = kept[0] });

        Assert.NotNull(kept[0].History[0].ValidationLoss);
        Assert.Null(history.BestEpoch);
        Assert.Null(history.Epochs[1].ValidationLoss);
    }

    [Fact]
    public void ARunResumedAtItsLastEpoch_TrainsNothing_AndJudgesNothingItNeverTrained()
    {
        // Judged by validation rows now, and by none when the checkpoint was taken: no epoch is left to train, so none is judged.
        var kept = new List<Checkpoint>();
        var network = new LayerStack(new Dense(1, 1, Draws()));
        network.Compile(new Sgd(), new MeanSquaredError()).Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 1) { Backend = _backend, Checkpoints = new Checkpoints(kept.Add) });
        var before = network.Slots().Select(slot => slot.Slot.Value.Values.ToArray()).ToArray();

        var history = network.Compile(new Sgd(), new MeanSquaredError()).Fit(Sloped(8, 2), Sloped(4, 2), new FitOptions(seed: 1) { Backend = _backend, ResumeFrom = kept[0] });

        Assert.Single(history.Epochs);
        Assert.Null(history.BestEpoch);
        Assert.Equal(before, network.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
    }

    [Fact]
    public void AResumeInBatchesOfSixteen_OfACheckpointTakenInBatchesOfThirtyTwo_IsRefused_NamingBoth_BeforeAnySlotIsPutBack()
    {
        var kept = new List<Checkpoint>();
        var network = new LayerStack(new Dense(1, 1, Draws()));
        network.Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), validation: null, new FitOptions(seed: 1) { Backend = _backend, Epochs = 2, Checkpoints = new Checkpoints(kept.Add) });
        var before = network.Slots().Select(slot => slot.Slot.Value).ToArray();

        var wrong = Assert.Throws<ArgumentException>(() => network.Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), validation: null, new FitOptions(seed: 1) { Backend = _backend, Epochs = 4, BatchSize = 16, ResumeFrom = kept[0] }));

        Assert.Equal(
            "The checkpoint was taken of a run in batches of 32, and going on in batches of 16 would take other rows into every step. (Parameter 'options')",
            wrong.Message);
        Assert.Equal(before, network.Slots().Select(slot => slot.Slot.Value));
    }

    [Theory]
    [InlineData("10 0 best", "3 0 best", "whose early stopping had a patience of 10, and going on with a patience of 3 would stop it by another rule.")]
    [InlineData("10 0 best", "10 0.01 best",
        "whose early stopping counted a fall of more than 0 as better, and going on counting a fall of more than 0.01 as better would stop it by another rule.")]
    [InlineData("10 0 best", "10 0 last", "that ends holding its best epoch's weights, and going on without restoring them would end it holding its last.")]
    [InlineData("10 0 last", "10 0 best", "that ends holding its last epoch's weights, and going on restoring the best would end it holding its best epoch's.")]
    [InlineData("10 0 best", "none", "with early stopping, and going on without it would never stop it early.")]
    [InlineData("none", "10 0 best", "with no early stopping, and going on with early stopping would stop it by a rule it never ran under.")]
    public void AResumeUnderOtherEarlyStopping_IsRefused_NamingWhatDiffers(string taken, string handed, string says)
    {
        var kept = new List<Checkpoint>();
        var network = new LayerStack(new Dense(1, 1, Draws()));
        network.Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), Sloped(16, 2), new FitOptions(seed: 3) { Backend = _backend, Epochs = 2, EarlyStopping = EarlyStoppingOf(taken), Checkpoints = new Checkpoints(kept.Add) });

        var wrong = Assert.Throws<ArgumentException>(() => network.Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), Sloped(16, 2), new FitOptions(seed: 3) { Backend = _backend, Epochs = 4, EarlyStopping = EarlyStoppingOf(handed), ResumeFrom = kept[0] }));

        Assert.Equal($"The checkpoint was taken of a run {says} (Parameter 'options')", wrong.Message);
    }

    [Fact]
    public void AResumeUnderAnotherSeed_OtherBatches_AnotherPatience_AndOnAnotherEngine_IsRefused_NamingEachAtOnce()
    {
        var kept = new List<Checkpoint>();
        new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), Sloped(16, 2), new FitOptions(seed: 1) { Backend = _backend, EarlyStopping = new EarlyStopping { Patience = 10 }, Checkpoints = new Checkpoints(kept.Add) });

        var wrong = Assert.Throws<ArgumentException>(() => new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), Sloped(16, 2), new FitOptions(seed: 2)
            {
                Backend = new ElsewhereBackend(_backend), Epochs = 2, BatchSize = 16, EarlyStopping = new EarlyStopping { Patience = 3 }, ResumeFrom = kept[0],
            }));

        Assert.Equal(
            "The checkpoint was taken of a run seeded 1, and going on under 2 would draw other numbers. "
            + "The checkpoint was taken of a run in batches of 32, and going on in batches of 16 would take other rows into every step. "
            + "The checkpoint was taken of a run whose early stopping had a patience of 10, and going on with a patience of 3 would stop it by another rule. "
            + $"The checkpoint was taken of a run on the engine {ElsewhereBackend.NamedAs(_backend)}, and going on under the engine {ElsewhereBackend.Named} would round every step otherwise. (Parameter 'options')",
            wrong.Message);
    }

    [Fact]
    public void AResumeOnAnotherEngine_IsRefused_NamingTheEngineTheCheckpointWasTakenOnAndTheOneHanded_BeforeAnySlotIsPutBack()
    {
        // The other engine works every step out on this one: it is refused for what the checkpoint recorded of the engine
        // its run was on — this one's name, and its version and device where it names them — and for nothing it would compute.
        var kept = new List<Checkpoint>();
        var network = Rich();
        network.Compile(new Adam(0.01), new MeanSquaredError())
            .Fit(Sloped(96, 2, width: 3), validation: null, new FitOptions(seed: 5) { Backend = _backend, Epochs = 2, BatchSize = 16, Checkpoints = new Checkpoints(kept.Add) });
        var before = network.Slots().Select(slot => slot.Slot.Value).ToArray();

        var wrong = Assert.Throws<ArgumentException>(() => network.Compile(new Adam(0.01), new MeanSquaredError())
            .Fit(Sloped(96, 2, width: 3), validation: null, new FitOptions(seed: 5) { Backend = new ElsewhereBackend(_backend), Epochs = 4, BatchSize = 16, ResumeFrom = kept[0] }));

        Assert.Equal(
            $"The checkpoint was taken of a run on the engine {ElsewhereBackend.NamedAs(_backend)}, and going on under the engine {ElsewhereBackend.Named} would round every step otherwise. (Parameter 'options')",
            wrong.Message);
        Assert.Equal(before, network.Slots().Select(slot => slot.Slot.Value));
    }

    [Fact]
    public void AResumeUnderTheSameValues_GoesOnBitForBit_ItsEarlyStoppingHandedAsANewOne()
    {
        // Early stopping is compared by what it says, not by which object says it: every run here is handed one of its own.
        var straight = Rich();
        var stopped = Rich();
        var resumed = Rich();
        var kept = new List<Checkpoint>();

        var whole = straight.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), Options(epochs: 8));
        stopped.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), Options(epochs: 3, checkpoints: new Checkpoints(kept.Add)));
        var rest = resumed.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), Options(epochs: 8, from: kept[^1]));

        Assert.Equal(whole.Epochs, rest.Epochs);
        Assert.Equal(whole.BestEpoch, rest.BestEpoch);
        Assert.Equal(whole.Stopped, rest.Stopped);
        Assert.Equal(
            straight.Slots().Select(slot => slot.Slot.Value.Values.ToArray()),
            resumed.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));

        FitOptions Options(int epochs, Checkpoints? checkpoints = null, Checkpoint? from = null) => new(seed: 5)
        {
            Backend = _backend,
            Epochs = epochs,
            BatchSize = 16,
            EarlyStopping = new EarlyStopping { Patience = 2, MinDelta = 0.001, RestoreBest = true },
            Checkpoints = checkpoints,
            ResumeFrom = from,
        };
    }

    [Fact]
    public void ARowWithNoNameGiven_IsNamedByItsPlace_FromOne()
    {
        var compiled = new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new BinaryCrossEntropy());
        var rows = new TrainingData(Tensor.From(new Shape(2, 1), [1f, 2f]), Tensor.From(new Shape(2, 1), [0f, 1.5f]));
        string[] names = ["first", "second"];

        var wrong = Assert.Throws<ArgumentException>(() => compiled.Fit(rows, validation: null, new FitOptions(seed: 1) { Backend = _backend }));

        Assert.Contains("row 2 cannot be trained on", wrong.Message, StringComparison.Ordinal);
        Assert.Null(rows.RowNames);
        Assert.Equal(names, new TrainingData(rows.Features, rows.Answers) { RowNames = names }.RowNames);
    }

    [Fact]
    public void SixEpochs_AreThreeEpochsACheckpointAndThreeMore_BitForBit()
    {
        var straight = Rich();
        var stopped = Rich();
        var resumed = Rich();
        var kept = new List<Checkpoint>();

        var whole = straight.Compile(new Adam(0.01), new MeanSquaredError(), new StepDecay(2, 0.5))
            .Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), new FitOptions(seed: 5) { Backend = _backend, Epochs = 6, BatchSize = 16 });
        stopped.Compile(new Adam(0.01), new MeanSquaredError(), new StepDecay(2, 0.5))
            .Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), new FitOptions(seed: 5) { Backend = _backend, Epochs = 3, BatchSize = 16, Checkpoints = new Checkpoints(kept.Add) });
        var rest = resumed.Compile(new Adam(0.01), new MeanSquaredError(), new StepDecay(2, 0.5))
            .Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), new FitOptions(seed: 5) { Backend = _backend, Epochs = 6, BatchSize = 16, ResumeFrom = kept[^1] });

        Assert.Equal(3, kept.Count);
        Assert.Equal(3, kept[^1].Epochs);
        Assert.Equal(
            straight.Slots().Select(slot => slot.Slot.Value.Values.ToArray()),
            resumed.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
        Assert.Equal(whole.Epochs, rest.Epochs);
    }

    [Fact]
    public void ACheckpointOfTheBestOnly_IsKeptWhenTheValidationLossImproves()
    {
        var kept = new List<Checkpoint>();

        var history = new LayerStack(new Dense(1, 1, Draws()))
            .Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), Sloped(16, -2), new FitOptions(seed: 3) { Backend = _backend, Epochs = 4, Checkpoints = new Checkpoints(kept.Add) { BestOnly = true } });

        Assert.Single(kept);
        Assert.Equal(1, kept[0].Epochs);
        Assert.Equal(4, history.Epochs.Count);
    }

    [Fact]
    public void Predicting_RunsAnEvaluationPass_ThroughTheLossesActivation()
    {
        var compiled = new LayerStack(new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias()), new Relu(), new Dropout(0.5),
                new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias()))
            .Compile(new Adam(), new BinaryCrossEntropy());

        var predictions = compiled.Predict(WalkedRows.Passengers(), _backend);

        Assert.Equal(0.5498339533805847, predictions.Values[0], 1e-6);
        Assert.Equal(0.5561374425888062, predictions.Values[2], 1e-6);
    }

    [Fact]
    public void PredictingAHundredRows_AChunkAtATime_ComesToWhatOnePassOverAllOfThemDoes_WithinTheEnginesOwnAgreement()
    {
        // The property TrainedNetwork.Answered's chunking rests on, held to this engine alone: a row's answer never depends
        // on which other rows share its pass, so working a hundred rows out four chunks at a time (thirty-two rows each, the
        // last holding four) comes to what one whole pass over all of them does — engine-independent of the report or a
        // pipeline, which is where the chunk sizes themselves (the run's batch size, and thirty-two for serving) are chosen.
        var compiled = new LayerStack(new Dense(2, 4, Draws()), new Relu(), new Dense(4, 1, Draws())).Compile(new Adam(), new BinaryCrossEntropy());
        var rows = Rows(100).Features;

        var noted = new NotingBackend(_backend);
        var whole = compiled.Predict(rows, noted).Values.ToArray();
        var onePass = noted.Count(nameof(ITensorBackend.MatMul));

        var counted = new NotingBackend(_backend);
        var chunked = PredictedAChunkAtATime(compiled, rows, counted, 32);

        // A hundred rows chunked at thirty-two: four passes, as many MatMuls as four whole passes make.
        Assert.Equal(4 * onePass, counted.Count(nameof(ITensorBackend.MatMul)));
        WithinAgreement(whole, chunked);
    }

    [Fact]
    public void WhatALoopCannotTrainOn_IsRefusedBeforeItStarts()
    {
        var compiled = new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new BinaryCrossEntropy());
        var rows = Sloped(8, 0.1);

        Assert.Throws<ArgumentException>(() => compiled.Fit(Rows(0, width: 1), validation: null, new FitOptions(seed: 1) { Backend = _backend }));
        Assert.Throws<ArgumentException>(() => compiled.Fit(rows, validation: null, new FitOptions(seed: 1) { Backend = _backend, EarlyStopping = new EarlyStopping() }));
        Assert.Throws<ArgumentException>(() => compiled.Fit(rows, validation: null, new FitOptions(seed: 1) { Backend = _backend, Checkpoints = new Checkpoints(_ => { }) { BestOnly = true } }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FitOptions(seed: 1) { Backend = _backend, Epochs = 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new FitOptions(seed: 1) { Backend = _backend, BatchSize = 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new EarlyStopping { Patience = -1 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new EarlyStopping { MinDelta = -0.1 });
    }

    [Fact]
    public void ARowTheLossCouldNotHaveMeant_IsRefused_NamedAsTheRowsNameIt()
    {
        var compiled = new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new BinaryCrossEntropy());
        var rows = new TrainingData(Tensor.From(new Shape(2, 1), [1f, 2f]), Tensor.From(new Shape(2, 1), [0f, 1.5f]))
        {
            RowNames = ["row 12 of titanic.csv", "row 13 of titanic.csv"],
        };

        var wrong = Assert.Throws<ArgumentException>(() => compiled.Fit(rows, validation: null, new FitOptions(seed: 1) { Backend = _backend }));

        Assert.Contains("row 13 of titanic.csv", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("1.5", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AResumeUnderAnotherSeed_IsRefused_ForItWouldDrawOtherNumbers()
    {
        var kept = new List<Checkpoint>();
        new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 1) { Backend = _backend, Checkpoints = new Checkpoints(kept.Add) });

        Assert.Throws<ArgumentException>(() => new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 2) { Backend = _backend, Epochs = 2, ResumeFrom = kept[0] }));
    }

    [Fact]
    public void TrainingDataWhoseFeaturesAndAnswersDisagree_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new TrainingData(Tensor.Zeros(new Shape(3, 2)), Tensor.Zeros(new Shape(2, 1))));
        Assert.Throws<ArgumentException>(() => new TrainingData(Tensor.Zeros(new Shape(3, 2)), Tensor.Zeros(new Shape(3))));
        Assert.Throws<ArgumentException>(() => new TrainingData(Tensor.Zeros(new Shape(3)), Tensor.Zeros(new Shape(3, 1))));
        Assert.Throws<ArgumentException>(() => new TrainingData(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(2, 1))) { RowNames = ["one"] });
    }

    [Fact]
    public void TitanicsPassengers_TrainedByTheLoop_AreJudgedBetterThanByTheMostCommonAnswer()
    {
        // The pipeline's own handover, a hidden layer of sixteen, Adam, and Keras's early stopping keeping its best epoch:
        // PyTorch reached 0.805 to 0.820 on these validation rows; the most common answer alone gets 0.615 of them right.
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("pclass", "sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .Build()
            .Run();
        var train = Handed(prepared.Batch(Part.Train, Needs.OneScale));
        var validation = Handed(prepared.Batch(Part.Validation, Needs.OneScale));
        var stream = new RandomStream(20260929);
        var compiled = new LayerStack(new Dense(14, 16, stream.Draw("initialise:0", 0, 0)), new Relu(), new Dense(16, 1, stream.Draw("initialise:2", 0, 0)))
            .Compile(new Adam(0.01), new BinaryCrossEntropy());

        var history = compiled.Fit(train, validation, new FitOptions(seed: 20260929)
        {
            Backend = _backend,
            Epochs = 200,
            EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true },
        });
        var predictions = compiled.Predict(validation.Features, _backend).Values.ToArray();
        var answers = validation.Answers.Values.ToArray();
        var right = predictions.Zip(answers).Count(pair => (pair.First >= 0.5f ? 1f : 0f) == pair.Second);

        Assert.True(history.Epochs.Count < 200);
        Assert.True(right / (double)answers.Length > 0.75, $"{right} of {answers.Length}");
    }

    [Fact]
    public void AClippedStep_ScalesEveryGradientByTheMostNormOverTheirWholeNorm_AsPyTorchsClipGradNormDoes()
    {
        // One step of SGD at a rate of a tenth over the four passengers, their gradients' whole norm — 0.2446 — clipped to
        // 0.05, as Fixtures/optimizers-pytorch.py printed PyTorch's clip_grad_norm_ and step.
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());

        new LayerStack(hidden, new Relu(), output).Compile(new Sgd(0.1), new BinaryCrossEntropy())
            .Fit(new TrainingData(WalkedRows.Passengers(), WalkedRows.Survived()), validation: null, new FitOptions(seed: 1)
            {
                Backend = _backend, BatchSize = 4, GradientClip = new GradientClip(0.05),
            });

        AssertClose([-0.30000001192092896, 0.20028923451900482, 0.0026806776877492666, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.20405973494052887], output.Bias.Value);
        AssertClose([0.10000000149011612, -0.09954631328582764, 0.05000000074505806, 0.0], hidden.Bias.Value);
        AssertClose([0.10000000149011612, 0.24954630434513092, -0.15000000596046448, 0.0], Tensor.From(new Shape(4), hidden.Weight.Value.Values[4..8]));
        AssertClose([0.0, 0.15045370161533356, -0.25, -0.10000000149011612], Tensor.From(new Shape(4), hidden.Weight.Value.Values[28..32]));
    }

    [Fact]
    public void AClipTheGradientsNeverReach_LeavesEveryStepAsItIsWithout_BitForBit()
    {
        // Clipped to ten, the walk's gradients pass untouched — PyTorch's step without a clip — and a run whose gradients
        // never reach the most norm is the run without one, to the last bit.
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());

        new LayerStack(hidden, new Relu(), output).Compile(new Sgd(0.1), new BinaryCrossEntropy())
            .Fit(new TrainingData(WalkedRows.Passengers(), WalkedRows.Survived()), validation: null, new FitOptions(seed: 1)
            {
                Backend = _backend, BatchSize = 4, GradientClip = new GradientClip(10),
            });

        AssertClose([-0.30000001192092896, 0.20141485333442688, 0.01311309915035963, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.2198590189218521], output.Bias.Value);

        var clipped = Rich();
        var plain = Rich();
        var never = clipped.Compile(new Adam(0.01), new MeanSquaredError())
            .Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), new FitOptions(seed: 5) { Backend = _backend, Epochs = 4, BatchSize = 16, GradientClip = new GradientClip(1e6) });
        var without = plain.Compile(new Adam(0.01), new MeanSquaredError())
            .Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), new FitOptions(seed: 5) { Backend = _backend, Epochs = 4, BatchSize = 16 });

        Assert.Equal(without.Epochs, never.Epochs);
        Assert.Equal(Bits(plain), Bits(clipped));
    }

    [Fact]
    public void AClipTheGradientsDoReach_MovesTheRunOtherwise_AndARunGoneOnFromItsCheckpointUnderTheSameClip_GoesOnBitForBit()
    {
        var straight = Rich();
        var stopped = Rich();
        var resumed = Rich();
        var plain = Rich();
        var kept = new List<Checkpoint>();

        var whole = straight.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), Options(6));
        stopped.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), Options(3, new Checkpoints(kept.Add)));
        var rest = resumed.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), Options(6, from: kept[^1]));
        plain.Compile(new Adam(0.01), new MeanSquaredError())
            .Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), new FitOptions(seed: 5) { Backend = _backend, Epochs = 6, BatchSize = 16 });

        Assert.Equal(whole.Epochs, rest.Epochs);
        Assert.Equal(Bits(straight), Bits(resumed));
        Assert.NotEqual(Bits(plain), Bits(straight));

        FitOptions Options(int epochs, Checkpoints? checkpoints = null, Checkpoint? from = null) => new(seed: 5)
        {
            Backend = _backend, Epochs = epochs, BatchSize = 16, GradientClip = new GradientClip(0.01), Checkpoints = checkpoints, ResumeFrom = from,
        };
    }

    [Theory]
    [InlineData(0.5, 0.25, "whose gradients were clipped to a norm of 0.5, and going on clipping them to a norm of 0.25 would move every step otherwise.")]
    [InlineData(0.5, null, "whose gradients were clipped to a norm of 0.5, and going on without clipping them would move every step otherwise.")]
    [InlineData(null, 0.5, "whose gradients were never clipped, and going on clipping them to a norm of 0.5 would move every step otherwise.")]
    public void AResumeUnderOtherClipping_IsRefused_NamingWhatDiffers(double? taken, double? handed, string says)
    {
        var kept = new List<Checkpoint>();
        var network = new LayerStack(new Dense(1, 1, Draws()));
        network.Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), validation: null, new FitOptions(seed: 3) { Backend = _backend, Epochs = 2, GradientClip = Clip(taken), Checkpoints = new Checkpoints(kept.Add) });

        var wrong = Assert.Throws<ArgumentException>(() => network.Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), validation: null, new FitOptions(seed: 3) { Backend = _backend, Epochs = 4, GradientClip = Clip(handed), ResumeFrom = kept[0] }));

        Assert.Equal($"The checkpoint was taken of a run {says} (Parameter 'options')", wrong.Message);

        static GradientClip? Clip(double? norm) => norm is { } most ? new GradientClip(most) : null;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void AClipToANormOfNothing_BelowIt_OrOfNoFiniteNumber_IsRefused(double norm)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new GradientClip(norm));
        Assert.Equal(0.5, new GradientClip(0.5).MaxNorm);
    }

    [Fact]
    public void ARunThatCompletes_IsTheSameToTheLastBit_HandedACallbackThatLooksAtEverythingAndATokenOrNeither()
    {
        // A callback reads what the run hands it and draws nothing; a token is read and draws nothing: neither moves a number.
        var watched = Rich();
        var plain = Rich();
        var watchedKept = new List<Checkpoint>();
        var plainKept = new List<Checkpoint>();
        using var source = new CancellationTokenSource();
        var looked = 0f;

        var with = watched.Compile(new Adam(0.01), new MeanSquaredError(), new StepDecay(2, 0.5)).Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), Options(watchedKept, watching: true));
        var without = plain.Compile(new Adam(0.01), new MeanSquaredError(), new StepDecay(2, 0.5)).Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), Options(plainKept));

        Assert.Equal(without.Epochs, with.Epochs);
        Assert.Equal(without.BestEpoch, with.BestEpoch);
        Assert.Equal(without.Stopped, with.Stopped);
        Assert.Equal(Bits(plain), Bits(watched));
        Assert.Equal(plainKept.Count, watchedKept.Count);
        Assert.NotEqual(0f, looked);

        FitOptions Options(List<Checkpoint> kept, bool watching = false) => new(seed: 5)
        {
            Backend = _backend,
            Epochs = 6,
            BatchSize = 16,
            EarlyStopping = new EarlyStopping { Patience = 2, RestoreBest = true },
            Checkpoints = new Checkpoints(kept.Add),
            Cancellation = watching ? source.Token : default,
            OnEpoch = watching ? epoch => looked += watched.Forward(Sloped(24, 2, width: 3).Features, Pass.Evaluation(_backend)).Values[0] + (float)epoch.Loss : null,
        };
    }

    [Fact]
    public void TheCallback_IsHandedEveryEpoch_InOrder_OnTheThreadTheRunTrainsOn_AfterItsCheckpoint_BeforeTheNextEpochBegins()
    {
        var watcher = new Watcher();
        var kept = new List<Checkpoint>();
        var handed = new List<Epoch>();
        var thread = Environment.CurrentManagedThreadId;

        var history = new LayerStack(watcher, new Dense(2, 1, Draws())).Compile(new Sgd(0.01), new MeanSquaredError())
            .Fit(Rows(64), Rows(16, from: 64), new FitOptions(seed: 7)
            {
                Backend = _backend,
                Epochs = 3,
                BatchSize = 16,
                Checkpoints = new Checkpoints(kept.Add),
                OnEpoch = epoch =>
                {
                    Assert.Equal(thread, Environment.CurrentManagedThreadId);
                    Assert.Equal(epoch.Number + 1, kept.Count);
                    Assert.All(watcher.Batches, batch => Assert.True(batch.Epoch <= epoch.Number));
                    Assert.Equal(4 * (epoch.Number + 1), watcher.Batches.Count);
                    handed.Add(epoch);
                },
            });

        Assert.Equal(history.Epochs, handed);
    }

    [Fact]
    public void ACancelAfterAnEpoch_ThrowsTheTokensCancel_ReturningNothing_AndLeavesTheNetworkAsTheEpochsItFinishedLeftIt()
    {
        // Cancelled as the second epoch is handed over, the run stops before the third's first batch: the network holds what
        // two epochs left it, exactly as a run of two epochs leaves it, and no history comes back.
        var cancelled = Rich();
        var two = Rich();
        using var source = new CancellationTokenSource();

        var stopped = Assert.Throws<OperationCanceledException>(() => cancelled.Compile(new Adam(0.01), new MeanSquaredError())
            .Fit(Sloped(96, 2, width: 3), validation: null, new FitOptions(seed: 5)
            {
                Backend = _backend, Epochs = 6, BatchSize = 16, Cancellation = source.Token, OnEpoch = epoch => Cancel(source, epoch.Number == 1),
            }));
        two.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Sloped(96, 2, width: 3), validation: null, new FitOptions(seed: 5) { Backend = _backend, Epochs = 2, BatchSize = 16 });

        Assert.Equal(source.Token, stopped.CancellationToken);
        Assert.Equal(Bits(two), Bits(cancelled));
    }

    [Fact]
    public void ACancelDuringABatch_LetsThatBatchFinish_AndStopsTheRunBeforeTheNext()
    {
        // Cancelled while the third batch of the first epoch is worked out: that batch's step is taken, and no fourth batch
        // is — the network holds what the third left it.
        using var source = new CancellationTokenSource();
        var watcher = new Watcher { During = (seen, count) => Cancel(source, seen.Epoch == 0 && count == 3) };
        var network = new LayerStack(watcher, new Dense(2, 1, Draws()));
        var before = network.Slots().Select(slot => slot.Slot.Value).ToArray();

        var stopped = Assert.Throws<OperationCanceledException>(() => network.Compile(new Sgd(0.01), new MeanSquaredError())
            .Fit(Rows(96), Rows(16, from: 96), new FitOptions(seed: 7) { Backend = _backend, Epochs = 2, BatchSize = 16, Cancellation = source.Token }));

        Assert.Equal(source.Token, stopped.CancellationToken);
        Assert.Equal(3, watcher.Batches.Count);
        Assert.Equal(0, watcher.Evaluations);
        Assert.NotEqual(before, network.Slots().Select(slot => slot.Slot.Value));
    }

    [Fact]
    public void ACancelDuringTheLastBatchOfAnEpoch_StopsTheRunBeforeItsValidationRowsAreLookedAt()
    {
        using var source = new CancellationTokenSource();
        var watcher = new Watcher { During = (seen, count) => Cancel(source, count == 6) };
        var network = new LayerStack(watcher, new Dense(2, 1, Draws()));
        var handed = new List<Epoch>();

        Assert.Throws<OperationCanceledException>(() => network.Compile(new Sgd(0.01), new MeanSquaredError())
            .Fit(Rows(96), Rows(16, from: 96), new FitOptions(seed: 7) { Backend = _backend, Epochs = 2, BatchSize = 16, Cancellation = source.Token, OnEpoch = handed.Add }));

        Assert.Equal(6, watcher.Batches.Count);
        Assert.Equal(0, watcher.Evaluations);
        Assert.Empty(handed);
    }

    [Fact]
    public void ATokenCancelledBeforeTheRun_TrainsNothing_AndMovesNothing()
    {
        using var source = new CancellationTokenSource();
        var watcher = new Watcher();
        var network = new LayerStack(watcher, new Dense(2, 1, Draws()));
        var before = network.Slots().Select(slot => slot.Slot.Value).ToArray();

        source.Cancel();

        Assert.Throws<OperationCanceledException>(() => network.Compile(new Sgd(0.01), new MeanSquaredError())
            .Fit(Rows(32), validation: null, new FitOptions(seed: 7) { Backend = _backend, Cancellation = source.Token }));

        Assert.Empty(watcher.Batches);
        Assert.Equal(before, network.Slots().Select(slot => slot.Slot.Value));
    }

    [Fact]
    public void ACallbackThatThrows_AbandonsTheRun_ItsExceptionReachingTheCallerAsItWasThrown()
    {
        var watcher = new Watcher();
        var thrown = new InvalidOperationException("Enough.");

        var caught = Assert.Throws<InvalidOperationException>(() => new LayerStack(watcher, new Dense(2, 1, Draws())).Compile(new Sgd(0.01), new MeanSquaredError())
            .Fit(Rows(64), validation: null, new FitOptions(seed: 7) { Backend = _backend, Epochs = 5, BatchSize = 16, OnEpoch = epoch => Throw(epoch.Number == 1, thrown) }));

        Assert.Same(thrown, caught);
        Assert.Equal(8, watcher.Batches.Count);
    }

    [Fact]
    public void ACheckpointTakenBeforeACancel_IsAsItWas_AndARunGoneOnFromIt_IsTheRunThatNeverStopped()
    {
        var straight = Rich();
        var cancelled = Rich();
        var resumed = Rich();
        var kept = new List<Checkpoint>();
        using var source = new CancellationTokenSource();

        var whole = straight.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), Options(6));
        Assert.Throws<OperationCanceledException>(() => cancelled.Compile(new Adam(0.01), new MeanSquaredError())
            .Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), Options(6, new Checkpoints(kept.Add), source.Token, epoch => Cancel(source, epoch.Number == 2))));
        var rest = resumed.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), Options(6, from: kept[^1]));

        Assert.Equal(3, kept.Count);
        Assert.Equal(whole.Epochs, rest.Epochs);
        Assert.Equal(Bits(straight), Bits(resumed));

        FitOptions Options(int epochs, Checkpoints? checkpoints = null, CancellationToken cancellation = default, Action<Epoch>? onEpoch = null, Checkpoint? from = null) =>
            new(seed: 5)
            {
                Backend = _backend, Epochs = epochs, BatchSize = 16, Checkpoints = checkpoints, Cancellation = cancellation, OnEpoch = onEpoch, ResumeFrom = from,
            };
    }

    // What a network predicts, worked out a chunk of rows at a time and brought back together in order: the core property
    // TrainedNetwork.Answered's chunking rests on, without needing the bridge package at all.
    private static float[] PredictedAChunkAtATime(CompiledNetwork compiled, Tensor rows, ITensorBackend engine, int size)
    {
        var width = rows.Shape[1];
        List<float> answers = [];

        for (var start = 0; start < rows.Shape[0]; start += size)
        {
            var count = Math.Min(size, rows.Shape[0] - start);

            answers.AddRange(compiled.Predict(Tensor.From(new Shape(count, width), rows.Values.Slice(start * width, count * width)), engine).Values.ToArray());
        }

        return [.. answers];
    }

    // Chunked or not, an evaluation pass on one engine agrees with its own whole pass within the same elementwise tolerance
    // every engine is held to against the light one (R7-F5).
    private static void WithinAgreement(float[] whole, float[] chunked)
    {
        Assert.Equal(whole.Length, chunked.Length);

        for (var at = 0; at < whole.Length; at++)
        {
            Assert.True(
                Math.Abs((double)chunked[at] - whole[at]) <= AgreementContract.AbsoluteTolerance + (AgreementContract.RelativeTolerance * Math.Abs(whole[at])),
                string.Create(CultureInfo.InvariantCulture, $"answer[{at}]: {chunked[at]} chunked against {whole[at]} in one pass"));
        }
    }

    // The rows a pipeline hands over, as the tensors a network takes.
    private static TrainingData Handed(Batch batch) => new(
        Tensor.From(new Shape(batch.RowCount, batch.Width), [.. batch.Features.SelectMany(row => row.Select(value => (float)value))]),
        Tensor.From(new Shape(batch.RowCount, batch.AnswerNames!.Count), [.. batch.Answers!.SelectMany(row => row.Select(value => (float)value))]));

    private static Draws Draws() => new RandomStream(42).Draw("initialise:test", 0, 0);

    private static int[][] Bits(Layer network) =>
        [.. network.Slots().Select(slot => slot.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray())];

    private static void AssertClose(double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual.Values[at], Math.Max(1e-9, Math.Abs(expected[at]) * 2e-5));
        }
    }

    // Early stopping as a theory writes it: none, or its patience, its least fall that counts, and whether the run ends
    // holding its best epoch or its last.
    private static EarlyStopping? EarlyStoppingOf(string said) => said.Split(' ') is [var patience, var minDelta, var end]
        ? new EarlyStopping
        {
            Patience = int.Parse(patience, CultureInfo.InvariantCulture),
            MinDelta = double.Parse(minDelta, CultureInfo.InvariantCulture),
            RestoreBest = end == "best",
        }
        : null;

    // Rows marked by their number in the first column, over 1024 so no step overshoots, a second column of noise, and an
    // answer made of both.
    private static TrainingData Rows(int count, int from = 0, int width = 2)
    {
        var features = new float[count * width];
        var answers = new float[count];

        for (var row = 0; row < count; row++)
        {
            features[row * width] = (from + row) / 1024f;
            for (var column = 1; column < width; column++)
            {
                features[(row * width) + column] = MathF.Sin(from + row);
            }

            answers[row] = MathF.Cos(from + row);
        }

        return new TrainingData(Tensor.From(new Shape(count, width), features), Tensor.From(new Shape(count, 1), answers));
    }

    // Rows of evenly spread inputs whose answer is the slope times the input's first column.
    private static TrainingData Sloped(int count, double slope, int width = 1)
    {
        var features = new float[count * width];
        var answers = new float[count];

        for (var row = 0; row < count; row++)
        {
            for (var column = 0; column < width; column++)
            {
                features[(row * width) + column] = MathF.Sin((row * width) + column + 1);
            }

            answers[row] = (float)(slope * features[row * width]);
        }

        return new TrainingData(Tensor.From(new Shape(count, width), features), Tensor.From(new Shape(count, 1), answers));
    }

    // Images of five by five with one channel, each holding a stretch of a sine, and an answer made of its first value.
    private static TrainingData Images(int count)
    {
        var images = new float[count * 25];
        var answers = new float[count];

        for (var at = 0; at < images.Length; at++)
        {
            images[at] = MathF.Sin(at * 0.37f);
        }

        for (var image = 0; image < count; image++)
        {
            answers[image] = MathF.Cos(images[image * 25]);
        }

        return new TrainingData(Tensor.From(new Shape(count, 5, 5, 1), images), Tensor.From(new Shape(count, 1), answers));
    }

    private static NormalisedStack Normalised()
    {
        var norm = new BatchNorm(2);

        return new NormalisedStack(new LayerStack(norm, new Dense(2, 1, Draws())), norm);
    }

    private static LayerStack Rich() =>
        new(new Dense(3, 8, Draws()), new BatchNorm(8), new Relu(), new Dropout(0.25), new Dense(8, 1, new RandomStream(43).Draw("initialise:test", 0, 0)));

    private readonly record struct NormalisedStack(LayerStack Network, BatchNorm Norm);

    private sealed record Seen(int Epoch, int Rows, ITensorBackend Backend, int[] Marks);

    // Cancels the source when told to, as a run's caller would from wherever it watches the run.
    private static void Cancel(CancellationTokenSource source, bool now)
    {
        if (now)
        {
            source.Cancel();
        }
    }

    private static void Throw(bool now, Exception thrown)
    {
        if (now)
        {
            throw thrown;
        }
    }

    // Passes its input on, and writes down every training batch it sees: its epoch, its rows, the backend it came through,
    // and the numbers the rows carry in their first column — handing each to what it is told to do during a batch — and
    // counts the evaluation passes it sees.
    private sealed class Watcher : Layer
    {
        public List<Seen> Batches { get; } = [];

        public int Evaluations { get; private set; }

        public Action<Seen, int>? During { get; init; }

        protected override Tensor Compute(Tensor input, Pass pass)
        {
            if (pass.Mode == PassMode.Training)
            {
                var width = input.Shape[1];
                var marks = Enumerable.Range(0, input.Shape[0]).Select(row => (int)MathF.Round(input.Values[row * width] * 1024)).ToArray();
                var seen = new Seen(pass.Epoch, input.Shape[0], pass.Backend, marks);

                Batches.Add(seen);
                During?.Invoke(seen, Batches.Count);
            }
            else
            {
                Evaluations++;
            }

            return input;
        }
    }
}
