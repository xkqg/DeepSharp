// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The loop that trains a compiled network: the training rows shuffled afresh every epoch and taken a batch at a time,
/// each batch through a recording pass of its own and an optimizer's step, the validation rows looked at once an epoch and
/// never trained on, early stopping as Keras has it, and a checkpoint that a run resumes from as if it had never stopped.
/// </summary>
public class LoopTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

    [Fact]
    public void SixHundredAndTwentyThreeRows_MakeTwentyBatchesOfThirtyTwo_TheLastHoldingFifteen()
    {
        var watcher = new Watcher();
        var network = new LayerStack(watcher, new Dense(2, 1, Draws()));

        network.Compile(new Sgd(0.01), new MeanSquaredError()).Fit(Rows(623), validation: null, new FitOptions(seed: 7));

        Assert.Equal([.. Enumerable.Repeat(32, 19), 15], watcher.Batches.Select(batch => batch.Rows));
    }

    [Fact]
    public void EveryEpoch_ShufflesTheTrainingRowsAfresh_AndNoValidationRowEverReachesATrainingPass()
    {
        var watcher = new Watcher();
        var network = new LayerStack(watcher, new Dense(2, 1, Draws()));

        network.Compile(new Sgd(0.01), new MeanSquaredError())
            .Fit(Rows(100), Rows(20, from: 100), new FitOptions(seed: 7) { Epochs = 2, BatchSize = 16 });

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

        network.Compile(new Sgd(0.01), new MeanSquaredError()).Fit(Rows(100), validation: null, new FitOptions(seed: 7) { Epochs = 2 });

        Assert.All(watcher.Batches, batch => Assert.IsType<RecordingBackend>(batch.Backend));
        Assert.Equal(watcher.Batches.Count, watcher.Batches.Select(batch => batch.Backend).Distinct(ReferenceEqualityComparer.Instance).Count());
    }

    [Fact]
    public void TheValidationRows_NeverMoveARunningStatistic()
    {
        var watched = Normalised();
        var unwatched = Normalised();

        watched.Network.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Rows(64), Rows(40, from: 64), new FitOptions(seed: 7) { Epochs = 3 });
        unwatched.Network.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Rows(64), validation: null, new FitOptions(seed: 7) { Epochs = 3 });

        Assert.Equal(unwatched.Norm.RunningMean.Value.Values.ToArray(), watched.Norm.RunningMean.Value.Values.ToArray());
        Assert.Equal(unwatched.Norm.RunningVariance.Value.Values.ToArray(), watched.Norm.RunningVariance.Value.Values.ToArray());
    }

    [Fact]
    public void TheHistory_HasARowPerEpoch_WithItsRate_AndSaysTheSeedAndTheBestEpoch()
    {
        var network = new LayerStack(new Dense(2, 1, Draws()));

        var history = network.Compile(new Sgd(0.1), new MeanSquaredError(), new StepDecay(every: 1, factor: 0.5))
            .Fit(Rows(64), Rows(16, from: 64), new FitOptions(seed: 11) { Epochs = 3 });

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
            .Fit(Rows(64), validation: null, new FitOptions(seed: 11) { Epochs = 2 });

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
            .Fit(Sloped(64, 2), Sloped(16, -2), new FitOptions(seed: 3) { Epochs = 10, EarlyStopping = new EarlyStopping() });

        Assert.Equal(2, history.Epochs.Count);
        Assert.Equal(0, history.BestEpoch);
        Assert.Equal(Stopping.NoLongerImproving, history.Stopped);
    }

    [Fact]
    public void EarlyStopping_WaitsAsManyEpochsAsItsPatience_BeforeItStops()
    {
        var history = new LayerStack(new Dense(1, 1, Draws()))
            .Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), Sloped(16, -2), new FitOptions(seed: 3) { Epochs = 10, EarlyStopping = new EarlyStopping { Patience = 3 } });

        Assert.Equal(4, history.Epochs.Count);
    }

    [Fact]
    public void AFallSmallerThanTheLeastThatCounts_IsNoImprovement()
    {
        // Both parts want twice the input, so the validation loss falls every epoch, by less and less; with a least fall that
        // counts, the run stops once the fall is smaller than it.
        var history = new LayerStack(new Dense(1, 1, Draws()))
            .Compile(new Sgd(0.05), new MeanSquaredError())
            .Fit(Sloped(64, 2), Sloped(16, 2), new FitOptions(seed: 3) { Epochs = 50, EarlyStopping = new EarlyStopping { MinDelta = 0.01 } });

        Assert.True(history.Epochs.Count < 50);
        Assert.Equal(Stopping.NoLongerImproving, history.Stopped);
    }

    [Fact]
    public void RestoringTheBestEpoch_BringsBackEverySlot_TheRunningStatisticsToo()
    {
        var stopped = Normalised();
        var once = Normalised();
        var options = new FitOptions(seed: 3) { Epochs = 10, EarlyStopping = new EarlyStopping { RestoreBest = true } };

        var history = stopped.Network.Compile(new Sgd(0.05), new MeanSquaredError()).Fit(Sloped(64, 2, width: 2), Sloped(16, -2, width: 2), options);
        once.Network.Compile(new Sgd(0.05), new MeanSquaredError()).Fit(Sloped(64, 2, width: 2), Sloped(16, -2, width: 2), new FitOptions(seed: 3));

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
            () => new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new MeanSquaredError()).Fit(huge, validation: null, new FitOptions(seed: 1)));

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
            () => network.Compile(new Sgd(), new MeanSquaredError()).Fit(train, huge, new FitOptions(seed: 1)));

        Assert.Contains("validation loss of epoch 1", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("not a finite number", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpointOfAnotherNetwork_IsRefused_ForItsSlotsAreNotThisOnes()
    {
        var kept = new List<Checkpoint>();
        new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 1) { Checkpoints = new Checkpoints(kept.Add) });

        var wrong = Assert.Throws<ArgumentException>(() => new LayerStack(new Dense(1, 2, Draws()), new Dense(2, 1, Draws())).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 1) { Epochs = 2, ResumeFrom = kept[0] }));

        Assert.Contains("another network", wrong.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => new Checkpoints(null!));

        // As many slots, under other paths, are other slots too.
        Assert.Contains("another network", Assert.Throws<ArgumentException>(() => new LayerStack(new Relu(), new Dense(1, 1, Draws())).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 1) { Epochs = 2, ResumeFrom = kept[0] })).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpointWhoseSlotsHaveOtherShapes_IsRefused_BeforeAnySlotIsReplaced()
    {
        var kept = new List<Checkpoint>();
        new LayerStack(new Dense(2, 1, Draws())).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Rows(8, width: 2), validation: null, new FitOptions(seed: 1) { Checkpoints = new Checkpoints(kept.Add) });
        var other = new LayerStack(new Dense(3, 1, Draws()));
        var before = other.Slots().Select(slot => slot.Slot.Value).ToArray();

        var wrong = Assert.Throws<ArgumentException>(() => other.Compile(new Sgd(), new MeanSquaredError())
            .Fit(Rows(8, width: 3), validation: null, new FitOptions(seed: 1) { Epochs = 2, ResumeFrom = kept[0] }));

        Assert.Contains("another network", wrong.Message, StringComparison.Ordinal);
        Assert.Equal(before, other.Slots().Select(slot => slot.Slot.Value));
    }

    [Fact]
    public void ARunJudgedByNoValidationRows_GoesOnJudgedByNone_EvenFromACheckpointThatWasJudged()
    {
        var kept = new List<Checkpoint>();
        var first = new LayerStack(new Dense(1, 1, Draws()));
        first.Compile(new Sgd(), new MeanSquaredError()).Fit(Sloped(8, 2), Sloped(4, 2), new FitOptions(seed: 1) { Checkpoints = new Checkpoints(kept.Add) });

        var history = first.Compile(new Sgd(), new MeanSquaredError()).Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 1) { Epochs = 2, ResumeFrom = kept[0] });

        Assert.NotNull(kept[0].History[0].ValidationLoss);
        Assert.Null(history.BestEpoch);
        Assert.Null(history.Epochs[1].ValidationLoss);
    }

    [Fact]
    public void ARunResumedAtItsLastEpoch_TrainsNothing_AndRestoresNothingItNeverJudged()
    {
        var kept = new List<Checkpoint>();
        var network = new LayerStack(new Dense(1, 1, Draws()));
        network.Compile(new Sgd(), new MeanSquaredError()).Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 1) { Checkpoints = new Checkpoints(kept.Add) });
        var before = network.Slots().Select(slot => slot.Slot.Value.Values.ToArray()).ToArray();

        var history = network.Compile(new Sgd(), new MeanSquaredError()).Fit(Sloped(8, 2), Sloped(4, 2), new FitOptions(seed: 1)
        {
            EarlyStopping = new EarlyStopping { RestoreBest = true }, ResumeFrom = kept[0],
        });

        Assert.Single(history.Epochs);
        Assert.Null(history.BestEpoch);
        Assert.Equal(before, network.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
    }

    [Fact]
    public void ARowWithNoNameGiven_IsNamedByItsPlace_FromOne()
    {
        var compiled = new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new BinaryCrossEntropy());
        var rows = new TrainingData(Tensor.From(new Shape(2, 1), [1f, 2f]), Tensor.From(new Shape(2, 1), [0f, 1.5f]));
        string[] names = ["first", "second"];

        var wrong = Assert.Throws<ArgumentException>(() => compiled.Fit(rows, validation: null, new FitOptions(seed: 1)));

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
            .Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), new FitOptions(seed: 5) { Epochs = 6, BatchSize = 16 });
        stopped.Compile(new Adam(0.01), new MeanSquaredError(), new StepDecay(2, 0.5))
            .Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), new FitOptions(seed: 5) { Epochs = 3, BatchSize = 16, Checkpoints = new Checkpoints(kept.Add) });
        var rest = resumed.Compile(new Adam(0.01), new MeanSquaredError(), new StepDecay(2, 0.5))
            .Fit(Sloped(96, 2, width: 3), Sloped(24, 2, width: 3), new FitOptions(seed: 5) { Epochs = 6, BatchSize = 16, ResumeFrom = kept[^1] });

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
            .Fit(Sloped(64, 2), Sloped(16, -2), new FitOptions(seed: 3) { Epochs = 4, Checkpoints = new Checkpoints(kept.Add) { BestOnly = true } });

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
    public void WhatALoopCannotTrainOn_IsRefusedBeforeItStarts()
    {
        var compiled = new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new BinaryCrossEntropy());
        var rows = Sloped(8, 0.1);

        Assert.Throws<ArgumentException>(() => compiled.Fit(Rows(0, width: 1), validation: null, new FitOptions(seed: 1)));
        Assert.Throws<ArgumentException>(() => compiled.Fit(rows, validation: null, new FitOptions(seed: 1) { EarlyStopping = new EarlyStopping() }));
        Assert.Throws<ArgumentException>(() => compiled.Fit(rows, validation: null, new FitOptions(seed: 1) { Checkpoints = new Checkpoints(_ => { }) { BestOnly = true } }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FitOptions(seed: 1) { Epochs = 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new FitOptions(seed: 1) { BatchSize = 0 });
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

        var wrong = Assert.Throws<ArgumentException>(() => compiled.Fit(rows, validation: null, new FitOptions(seed: 1)));

        Assert.Contains("row 13 of titanic.csv", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("1.5", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AResumeUnderAnotherSeed_IsRefused_ForItWouldDrawOtherNumbers()
    {
        var kept = new List<Checkpoint>();
        new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 1) { Checkpoints = new Checkpoints(kept.Add) });

        Assert.Throws<ArgumentException>(() => new LayerStack(new Dense(1, 1, Draws())).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Sloped(8, 2), validation: null, new FitOptions(seed: 2) { Epochs = 2, ResumeFrom = kept[0] }));
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
            Epochs = 200,
            EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true },
        });
        var predictions = compiled.Predict(validation.Features, _backend).Values.ToArray();
        var answers = validation.Answers.Values.ToArray();
        var right = predictions.Zip(answers).Count(pair => (pair.First >= 0.5f ? 1f : 0f) == pair.Second);

        Assert.True(history.Epochs.Count < 200);
        Assert.True(right / (double)answers.Length > 0.75, $"{right} of {answers.Length}");
    }

    // The rows a pipeline hands over, as the tensors a network takes.
    private static TrainingData Handed(Batch batch) => new(
        Tensor.From(new Shape(batch.RowCount, batch.Width), [.. batch.Features.SelectMany(row => row.Select(value => (float)value))]),
        Tensor.From(new Shape(batch.RowCount, batch.AnswerNames!.Count), [.. batch.Answers!.SelectMany(row => row.Select(value => (float)value))]));

    private static Draws Draws() => new RandomStream(42).Draw("initialise:test", 0, 0);

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

    private static NormalisedStack Normalised()
    {
        var norm = new BatchNorm(2);

        return new NormalisedStack(new LayerStack(norm, new Dense(2, 1, Draws())), norm);
    }

    private static LayerStack Rich() =>
        new(new Dense(3, 8, Draws()), new BatchNorm(8), new Relu(), new Dropout(0.25), new Dense(8, 1, new RandomStream(43).Draw("initialise:test", 0, 0)));

    private readonly record struct NormalisedStack(LayerStack Network, BatchNorm Norm);

    private sealed record Seen(int Epoch, int Rows, ITensorBackend Backend, int[] Marks);

    // Passes its input on, and writes down every training batch it sees: its epoch, its rows, the backend it came through,
    // and the numbers the rows carry in their first column.
    private sealed class Watcher : Layer
    {
        public List<Seen> Batches { get; } = [];

        protected override Tensor Compute(Tensor input, Pass pass)
        {
            if (pass.Mode == PassMode.Training)
            {
                var width = input.Shape[1];
                var marks = Enumerable.Range(0, input.Shape[0]).Select(row => (int)MathF.Round(input.Values[row * width] * 1024)).ToArray();

                Batches.Add(new Seen(pass.Epoch, input.Shape[0], pass.Backend, marks));
            }

            return input;
        }
    }
}
