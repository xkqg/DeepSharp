// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Networks;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// What a run does around its optimizer: each gradient value clipped before the whole norm, a step taken once every so many
/// batches, and an average of the weights kept beside them and written over them at the end. The walked rows through the walked
/// network, in full batches, land where PyTorch's land — <c>clip_grad_value_</c> and <c>clip_grad_norm_</c> before its step, and
/// its exponential moving average of the weights after it — and a run that stops and goes on is the run that never stopped.
/// </summary>
/// <param name="engine">The backend the runs are on.</param>
public abstract class RunOptionsContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    // Fixtures/optimizers-options-pytorch.py: the same walked rows, three or four full-batch steps of SGD at a tenth.
    private static readonly JsonElement Runs = JsonDocument
        .Parse(File.ReadAllText(Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "optimizers-options-pytorch.json")))
        .RootElement.GetProperty("runs");

    public static TheoryData<string> EveryRun { get; } = [.. Runs.EnumerateArray().Select(each => each.GetProperty("name").GetString()!)];

    [Theory]
    [MemberData(nameof(EveryRun), MemberType = typeof(RunOptionsContract))]
    public void ARunWithClippedValuesOrAnAverage_LandsWhereTheSameRunInPyTorchLands(string name)
    {
        var run = Runs.EnumerateArray().Single(each => each.GetProperty("name").GetString() == name);
        var (hidden, output, network) = Walked();

        network.Compile(new Sgd(run.GetProperty("rate").GetDouble()), new BinaryCrossEntropy())
            .Fit(Rows(), validation: null, new FitOptions(seed: 1)
            {
                Backend = _backend,
                Epochs = run.GetProperty("steps").GetInt32(),
                BatchSize = 4,
                ClipValue = Number(run, "clipValue"),
                GradientClip = Number(run, "clipNorm") is { } norm ? new GradientClip(norm) : null,
                Average = Number(run, "average"),
            });

        // A run that keeps an average ends holding it; one that does not, the weights.
        var expected = run.TryGetProperty("averaged", out var averaged) && averaged.ValueKind == JsonValueKind.Object ? averaged : run.GetProperty("parameters");

        AssertClose(Doubles(expected, "hidden.weight"), hidden.Weight.Value);
        AssertClose(Doubles(expected, "hidden.bias"), hidden.Bias!.Value);
        AssertClose(Doubles(expected, "output.weight"), output.Weight.Value);
        AssertClose(Doubles(expected, "output.bias"), output.Bias!.Value);
    }

    [Fact]
    public void AStepTakenOnceEveryTwoBatches_OfTwoRows_IsTheStepOfTheFourRowsTogether_AsTheFullBatchRunIs()
    {
        var plain = Runs.EnumerateArray().Single(each => each.GetProperty("name").GetString() == "plain");
        var (hidden, output, network) = Walked();

        network.Compile(new Sgd(0.1), new BinaryCrossEntropy())
            .Fit(Rows(), validation: null, new FitOptions(seed: 1) { Backend = _backend, Epochs = 3, BatchSize = 2, StepEvery = 2 });

        var parameters = plain.GetProperty("parameters");

        AssertClose(Doubles(parameters, "hidden.weight"), hidden.Weight.Value);
        AssertClose(Doubles(parameters, "output.weight"), output.Weight.Value);
        AssertClose(Doubles(parameters, "output.bias"), output.Bias!.Value);
    }

    [Fact]
    public void AStepTakenEveryBatch_IsTheRunWithoutTheSetting_AndAnAverageOfNothingLeftOutIsTheRunWithout_BitForBit()
    {
        var plain = Walked();
        var every = Walked();

        plain.Network.Compile(new Adam(0.01), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Backend = _backend, Epochs = 4, BatchSize = 2 });
        every.Network.Compile(new Adam(0.01), new BinaryCrossEntropy())
            .Fit(Rows(), validation: null, new FitOptions(seed: 1) { Backend = _backend, Epochs = 4, BatchSize = 2, StepEvery = 1, ClipValue = null, Average = null });

        Assert.Equal(Bits(plain.Network), Bits(every.Network));
    }

    [Fact]
    public void ARunThatStopsAndGoesOn_UnderClippedValuesAWindowOfBatchesAndAnAverage_IsTheRunThatNeverStopped_BitForBit()
    {
        var straight = Walked();
        var stopped = Walked();
        var resumed = Walked();
        var kept = new List<Checkpoint>();

        straight.Network.Compile(new Adam(0.05), new BinaryCrossEntropy()).Fit(Rows(), validation: null, Options(6));
        stopped.Network.Compile(new Adam(0.05), new BinaryCrossEntropy()).Fit(Rows(), validation: null, Options(3, new Checkpoints(kept.Add)));
        resumed.Network.Compile(new Adam(0.05), new BinaryCrossEntropy()).Fit(Rows(), validation: null, Options(6, from: kept[^1]));

        Assert.Equal(Bits(straight.Network), Bits(resumed.Network));

        FitOptions Options(int epochs, Checkpoints? checkpoints = null, Checkpoint? from = null) => new(seed: 5)
        {
            Backend = _backend, Epochs = epochs, BatchSize = 1, StepEvery = 3, ClipValue = 0.02, Average = 0.9, Checkpoints = checkpoints, ResumeFrom = from,
        };
    }

    [Fact]
    public void AnAverageOfTheWeights_IsWrittenOverThemAtTheEnd_AndOnlyAtTheEnd()
    {
        var with = Walked();
        var without = Walked();
        var epochs = new List<double>();

        with.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy())
            .Fit(Rows(), Rows(), new FitOptions(seed: 1) { Backend = _backend, Epochs = 3, BatchSize = 4, Average = 0.9, OnEpoch = epoch => epochs.Add(epoch.ValidationLoss!.Value) });
        var raw = without.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy())
            .Fit(Rows(), Rows(), new FitOptions(seed: 1) { Backend = _backend, Epochs = 3, BatchSize = 4 });

        // Every epoch is judged by the weights as they are, so the run's history is the run without an average; only its end differs.
        Assert.Equal(raw.Epochs.Select(epoch => epoch.ValidationLoss!.Value), epochs);
        Assert.NotEqual(Bits(without.Network), Bits(with.Network));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.5)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void AClipOfValuesToNothingOrBelowIt_OrToNoFiniteNumber_IsRefused(double wrong) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new FitOptions(1) { ClipValue = wrong });

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public void AStepOnceEveryNoBatchesOrFewer_IsRefused(int wrong) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new FitOptions(1) { StepEvery = wrong });

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    public void AnAverageThatIsNoShareBetweenNothingAndOne_IsRefused(double wrong) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new FitOptions(1) { Average = wrong });

    [Fact]
    public void AnAverageAndTheBestEpochsWeights_BothSayWhatTheRunEndsHolding_AndAreRefusedTogether_WhileEpochsAreJudgedByTheWeights()
    {
        var network = Walked().Network.Compile(new Sgd(0.1), new BinaryCrossEntropy());

        var refused = Assert.Throws<ArgumentException>(() => network.Fit(
            Rows(), Rows(), new FitOptions(seed: 1) { Backend = _backend, Average = 0.9, EarlyStopping = new EarlyStopping { Patience = 1, RestoreBest = true } }));

        Assert.Contains("average", refused.Message, StringComparison.Ordinal);
        Assert.Contains("best", refused.Message, StringComparison.Ordinal);
        Assert.Contains(nameof(FitOptions.JudgedByAverage), refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EpochsJudgedByTheAverage_WithoutAnAverage_AreRefused()
    {
        var network = Walked().Network.Compile(new Sgd(0.1), new BinaryCrossEntropy());

        var refused = Assert.Throws<ArgumentException>(() => network.Fit(
            Rows(), Rows(), new FitOptions(seed: 1) { Backend = _backend, JudgedByAverage = true }));

        Assert.Contains(nameof(FitOptions.Average), refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEpochJudgedByTheAverage_IsJudgedByTheAveragedWeights_AndTheRunTrainsOnTheWeightsAsTheyStand()
    {
        // The last epoch is judged by the averages the run then ends holding, so its loss is the loss of the network it ends
        // as; and judging reads the validation rows and changes nothing, so the weights the run trains are the ones the run
        // that judges by them trains, to the bit.
        var judged = Walked();
        var raw = Walked();
        var compiled = judged.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy());
        var history = compiled.Fit(Rows(), Rows(), new FitOptions(seed: 1) { Backend = _backend, Epochs = 3, BatchSize = 4, Average = 0.9, JudgedByAverage = true });
        var plain = raw.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy())
            .Fit(Rows(), Rows(), new FitOptions(seed: 1) { Backend = _backend, Epochs = 3, BatchSize = 4, Average = 0.9 });
        var rows = Rows();
        var heldAtTheEnd = new BinaryCrossEntropy().Of(judged.Network.Forward(rows.Features, Pass.Evaluation(_backend)), rows.Answers, _backend).Values[0];

        Assert.Equal(heldAtTheEnd, history.Epochs[^1].ValidationLoss!.Value, 6);
        Assert.NotEqual(plain.Epochs.Select(epoch => epoch.ValidationLoss), history.Epochs.Select(epoch => epoch.ValidationLoss));
        Assert.Equal(plain.Epochs.Select(epoch => epoch.Loss), history.Epochs.Select(epoch => epoch.Loss));
        Assert.Equal(Bits(raw.Network), Bits(judged.Network));
    }

    [Fact]
    public void RestoringTheBestEpoch_WhenEpochsAreJudgedByTheAverage_EndsHoldingThatEpochsAverage()
    {
        // The run that stops at the best epoch it was judged by ends holding what a run of exactly that many epochs ends
        // holding: the averages at that epoch's end.
        var restored = Walked();
        var history = restored.Network.Compile(new Sgd(0.5), new BinaryCrossEntropy()).Fit(Rows(), Rows(), new FitOptions(seed: 1)
        {
            Backend = _backend, Epochs = 8, BatchSize = 2, Average = 0.7, JudgedByAverage = true,
            EarlyStopping = new EarlyStopping { Patience = 100, RestoreBest = true },
        });
        var best = history.BestEpoch!.Value;
        var shorter = Walked();

        shorter.Network.Compile(new Sgd(0.5), new BinaryCrossEntropy())
            .Fit(Rows(), Rows(), new FitOptions(seed: 1) { Backend = _backend, Epochs = best + 1, BatchSize = 2, Average = 0.7, JudgedByAverage = true });

        Assert.Equal(Bits(shorter.Network), Bits(restored.Network));
    }

    [Fact]
    public void ARunJudgedByTheAverageThatStopsAndGoesOn_RestoringItsBest_IsTheRunThatNeverStopped_BitForBit()
    {
        var straight = Walked();
        var stopped = Walked();
        var resumed = Walked();
        var kept = new List<Checkpoint>();

        straight.Network.Compile(new Adam(0.05), new BinaryCrossEntropy()).Fit(Rows(), Rows(), Options(6));
        stopped.Network.Compile(new Adam(0.05), new BinaryCrossEntropy()).Fit(Rows(), Rows(), Options(3, new Checkpoints(kept.Add)));
        resumed.Network.Compile(new Adam(0.05), new BinaryCrossEntropy()).Fit(Rows(), Rows(), Options(6, from: kept[^1]));

        Assert.Equal(Bits(straight.Network), Bits(resumed.Network));

        FitOptions Options(int epochs, Checkpoints? checkpoints = null, Checkpoint? from = null) => new(seed: 5)
        {
            Backend = _backend, Epochs = epochs, BatchSize = 1, Average = 0.8, JudgedByAverage = true,
            EarlyStopping = new EarlyStopping { Patience = 100, RestoreBest = true }, Checkpoints = checkpoints, ResumeFrom = from,
        };
    }

    [Theory]
    [InlineData("clipValue", "whose gradients were clipped value by value to 0.02, and going on without clipping them that way would move every step otherwise.")]
    [InlineData("stepEvery", "that took a step once every 3 batches, and going on taking one every batch would take other rows into every step.")]
    [InlineData("average", "that kept an average of its weights at 0.9, and going on without one would end it holding other weights.")]
    [InlineData("judgedByAverage", "whose epochs were judged by the average of its weights, and going on judging them by the weights as they stand would stop it and keep its best otherwise.")]
    public void AResumeUnderOtherOptionsOfTheRun_IsRefused_NamingWhatDiffers(string option, string says)
    {
        var kept = new List<Checkpoint>();
        var taken = new FitOptions(seed: 3)
        {
            Backend = _backend,
            Epochs = 2,
            BatchSize = 1,
            ClipValue = option == "clipValue" ? 0.02 : null,
            StepEvery = option == "stepEvery" ? 3 : 1,
            Average = option is "average" or "judgedByAverage" ? 0.9 : null,
            JudgedByAverage = option == "judgedByAverage",
            Checkpoints = new Checkpoints(kept.Add),
        };

        Walked().Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, taken);

        var refused = Assert.Throws<ArgumentException>(
            () => Walked().Network.Compile(new Sgd(0.1), new BinaryCrossEntropy())
                .Fit(Rows(), validation: null, new FitOptions(seed: 3)
                {
                    Backend = _backend, Epochs = 4, BatchSize = 1, Average = option == "judgedByAverage" ? 0.9 : null, ResumeFrom = kept[0],
                }));

        Assert.Contains(says, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AResumeJudgingByTheAverage_OfARunJudgedByItsWeights_IsRefused_NamingWhatDiffers()
    {
        var kept = new List<Checkpoint>();

        Walked().Network.Compile(new Sgd(0.1), new BinaryCrossEntropy())
            .Fit(Rows(), validation: null, new FitOptions(seed: 3) { Backend = _backend, Epochs = 2, BatchSize = 1, Average = 0.9, Checkpoints = new Checkpoints(kept.Add) });

        var refused = Assert.Throws<ArgumentException>(
            () => Walked().Network.Compile(new Sgd(0.1), new BinaryCrossEntropy())
                .Fit(Rows(), validation: null, new FitOptions(seed: 3) { Backend = _backend, Epochs = 4, BatchSize = 1, Average = 0.9, JudgedByAverage = true, ResumeFrom = kept[0] }));

        Assert.Contains(
            "whose epochs were judged by the weights as they stand, and going on judging them by the average of its weights would stop it and keep its best otherwise.",
            refused.Message,
            StringComparison.Ordinal);
    }

    private (Dense Hidden, Dense Output, LayerStack Network) Walked()
    {
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());

        return (hidden, output, new LayerStack(hidden, new Relu(), output));
    }

    private static TrainingData Rows() => new(WalkedRows.Passengers(), WalkedRows.Survived());

    private static double? Number(JsonElement run, string key) => run.TryGetProperty(key, out var written) && written.ValueKind == JsonValueKind.Number ? written.GetDouble() : null;

    private static double[] Doubles(JsonElement written, string key) => [.. written.GetProperty(key).EnumerateArray().Select(value => value.GetDouble())];

    private static int[][] Bits(Layer network) =>
        [.. network.Parameters().Select(parameter => parameter.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray())];

    private static void AssertClose(double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual.Values[at], Math.Max(1e-9, Math.Abs(expected[at]) * 2e-5));
        }
    }
}
