// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// A dense layer with a low-rank pair beside its held kernel answers as Keras's <c>Dense</c> does once <c>enable_lora</c> has
/// been called, and a step of training moves the pair and the bias and leaves the kernel where it was: the answers to four rows
/// and the numbers after one step of plain gradient descent on the mean squared error land where Keras's land
/// (Fixtures/lowrank-keras.py), on any engine.
/// </summary>
/// <param name="engine">The backend the arithmetic runs on.</param>
public abstract class LowRankContract(ITensorBackend engine)
{
    private static readonly JsonElement Keras = JsonDocument
        .Parse(File.ReadAllText(Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "lowrank-keras.json")))
        .RootElement;

    private readonly ITensorBackend _backend = engine;

    [Fact]
    public void TheAnswers_AreTheKernelsProductPlusThePairsScaledByAlphaOverRank_AsKerasDenseAnswers()
    {
        var (dense, _) = Walked();
        var answered = dense.Forward(Rows(), Pass.Evaluation(_backend)).Values.ToArray();
        var expected = Numbers("answers");

        Assert.Equal(expected.Length, answered.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], answered[at], Math.Max(1e-5, Math.Abs(expected[at]) * 1e-5));
        }
    }

    [Fact]
    public void AStepOfGradientDescent_MovesThePairAndTheBias_AndLeavesTheKernel_WhereKerasLeavesThem()
    {
        var (dense, network) = Walked();
        var after = Keras.GetProperty("after");

        network.Compile(new Sgd(Keras.GetProperty("rate").GetDouble()), new MeanSquaredError())
            .Fit(new TrainingData(Rows(), Targets()), validation: null, new FitOptions(seed: 1) { Backend = _backend, Epochs = 1, BatchSize = 4 });

        Close(Doubles(after, "kernel"), dense.Weight.Value);
        Close(Doubles(after, "bias"), dense.Bias!.Value);
        Close(Doubles(after, "a"), dense.A!.Value);
        Close(Doubles(after, "b"), dense.B!.Value);
    }

    [Fact]
    public void TheLossOfTheFirstEpoch_IsTheLossKerasReportsBeforeItsFirstStep()
    {
        var (_, network) = Walked();
        var history = network.Compile(new Sgd(Keras.GetProperty("rate").GetDouble()), new MeanSquaredError())
            .Fit(new TrainingData(Rows(), Targets()), validation: null, new FitOptions(seed: 1) { Backend = _backend, Epochs = 1, BatchSize = 4 });

        Assert.Equal(Keras.GetProperty("loss").GetDouble(), history.Epochs[0].Loss, 1e-4);
    }

    // The layer Keras was walked with: its kernel, bias and pair put in by their paths, as a file of a network puts them in.
    private (Dense Dense, LayerStack Network) Walked()
    {
        var rank = Keras.GetProperty("rank").GetInt32();
        var dense = new Dense(3, 2, new Beginning(new RandomStream(1).Draw("initialise:0", 0, 0)) { LowRank = new LowRank(rank, Keras.GetProperty("alpha").GetDouble()) });
        var network = new LayerStack(dense);

        network.Load(
        [
            new SlotEntry("0.weight", Tensor.From(new Shape(3, 2), Floats("kernel")), "kernel"),
            new SlotEntry("0.bias", Tensor.From(new Shape(2), Floats("bias")), "bias"),
            new SlotEntry("0.lora_a", Tensor.From(new Shape(3, rank), Floats("a")), "a"),
            new SlotEntry("0.lora_b", Tensor.From(new Shape(rank, 2), Floats("b")), "b"),
        ]);

        return (dense, network);
    }

    private static Tensor Rows() => Tensor.From(new Shape(4, 3), Floats("rows"));

    private static Tensor Targets() => Tensor.From(new Shape(4, 2), Floats("targets"));

    private static float[] Floats(string key) => [.. Numbers(key).Select(value => (float)value)];

    private static double[] Numbers(string key) => Doubles(Keras, key);

    private static double[] Doubles(JsonElement written, string key) => [.. written.GetProperty(key).EnumerateArray().Select(value => value.GetDouble())];

    private static void Close(double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual.Values[at], Math.Max(1e-6, Math.Abs(expected[at]) * 2e-5));
        }
    }
}
