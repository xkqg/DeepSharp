// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Networks;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// What the learning does to the numbers of a slot beside stepping them: a penalty the loss adds — Keras's L1 and L2
/// regularizers — a bound each unit's numbers are held under after a step — Keras's MaxNorm — and a layer held still. The walked
/// rows through the walked network land where PyTorch's land when its loss adds the same terms, its steps are followed by the
/// same bound, and the same parameters are left out of its optimizer; and a held normalisation keeps its running statistics.
/// </summary>
/// <param name="engine">The backend the runs are on.</param>
public abstract class RulesContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    // Fixtures/optimizers-options-pytorch.py: the same walked rows, three full-batch steps of SGD at a tenth under each set of rules.
    private static readonly JsonElement Rules = JsonDocument
        .Parse(File.ReadAllText(Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "optimizers-options-pytorch.json")))
        .RootElement.GetProperty("rules");

    public static TheoryData<string> EveryRun { get; } = [.. Rules.EnumerateArray().Select(each => each.GetProperty("name").GetString()!)];

    [Theory]
    [MemberData(nameof(EveryRun), MemberType = typeof(RulesContract))]
    public void APenaltyABoundAndAHeldLayer_LandWhereTheSameRunInPyTorchLands_TheLossesItReportsIncludingThePenalty(string name)
    {
        var run = Rules.EnumerateArray().Single(each => each.GetProperty("name").GetString() == name);
        var (hidden, output, network) = Walked();
        var named = new Dictionary<string, Parameter>
        {
            ["hidden.weight"] = hidden.Weight,
            ["hidden.bias"] = hidden.Bias!,
            ["output.weight"] = output.Weight,
            ["output.bias"] = output.Bias!,
        };

        if (run.TryGetProperty("penalties", out var penalties))
        {
            foreach (var penalty in penalties.EnumerateObject())
            {
                named[penalty.Name].Rules = named[penalty.Name].Rules with { L1 = penalty.Value[0].GetDouble(), L2 = penalty.Value[1].GetDouble() };
            }
        }

        if (run.TryGetProperty("bounds", out var bounds))
        {
            foreach (var bound in bounds.EnumerateObject())
            {
                named[bound.Name].Rules = named[bound.Name].Rules with { MaxNorm = bound.Value.GetDouble() };
            }
        }

        if (run.TryGetProperty("held", out var held))
        {
            foreach (var each in held.EnumerateArray())
            {
                named[each.GetString()!].Rules = named[each.GetString()!].Rules with { Held = true };
            }
        }

        var history = network.Compile(new Sgd(0.1), new BinaryCrossEntropy())
            .Fit(Rows(), validation: null, new FitOptions(seed: 1) { Backend = _backend, Epochs = run.GetProperty("steps").GetInt32(), BatchSize = 4 });

        var losses = run.GetProperty("losses").EnumerateArray().Select(loss => loss.GetDouble()).ToArray();

        Assert.Equal(losses.Length, history.Epochs.Count);

        for (var epoch = 0; epoch < losses.Length; epoch++)
        {
            Assert.Equal(losses[epoch], history.Epochs[epoch].Loss, 1e-5);
        }

        var parameters = run.GetProperty("parameters");

        AssertClose(Doubles(parameters, "hidden.weight"), hidden.Weight.Value);
        AssertClose(Doubles(parameters, "hidden.bias"), hidden.Bias!.Value);
        AssertClose(Doubles(parameters, "output.weight"), output.Weight.Value);
        AssertClose(Doubles(parameters, "output.bias"), output.Bias!.Value);
    }

    [Fact]
    public void TheValidationLoss_IncludesThePenalty_AsTheTrainingLossDoes()
    {
        var plain = Walked();
        var penalised = Walked();

        penalised.Output.Weight.Rules = new SlotRules { L2 = 0.5 };

        var options = new FitOptions(seed: 1) { Backend = _backend, Epochs = 1, BatchSize = 4 };
        var without = plain.Network.Compile(new Sgd(0.0001), new BinaryCrossEntropy()).Fit(Rows(), Rows(), options);
        var with = penalised.Network.Compile(new Sgd(0.0001), new BinaryCrossEntropy()).Fit(Rows(), Rows(), options);

        // The penalty is half the sum of the squares of the output weights as they stand when the epoch is judged.
        var squares = penalised.Output.Weight.Value.Values.ToArray().Sum(value => (double)value * value);

        Assert.Equal(without.Epochs[0].ValidationLoss!.Value + (0.5 * squares), with.Epochs[0].ValidationLoss!.Value, 1e-4);
        Assert.True(with.Epochs[0].Loss > without.Epochs[0].Loss + 0.05);
    }

    [Fact]
    public void AMaxNormOfADenseLayer_HoldsEachUnitsNumbersUnderTheBound_AsKerasFormulaDoes_AndLeavesTheOnesUnderItAlone()
    {
        // Two units over three inputs: the first unit's numbers (3, 4, 0) have a norm of 5, the second's (0.3, 0.4, 0) of 0.5. Held
        // under 1, the first is scaled by 1 / (1e-7 + 5) and the second by 0.5 / (1e-7 + 0.5), which is one to a millionth, as
        // Keras's w x clip(n, 0, m) / (1e-7 + n) has it.
        var dense = new Dense(Tensor.From(new Shape(3, 2), [3f, 0.3f, 4f, 0.4f, 0f, 0f]), Tensor.Zeros(new Shape(2)));

        dense.Weight.Rules = new SlotRules { MaxNorm = 1 };
        new Sgd(0.1).Step([dense.Weight], ZeroGradients(dense.Weight), 0.1, _backend);

        AssertClose([0.6, 0.3, 0.8, 0.4, 0, 0], dense.Weight.Value);
    }

    [Fact]
    public void AMaxNormOfAVector_HoldsTheWholeVectorsNormUnderTheBound()
    {
        var dense = new Dense(Tensor.Zeros(new Shape(1, 2)), Tensor.From(new Shape(2), [3f, 4f]));

        dense.Bias!.Rules = new SlotRules { MaxNorm = 1 };
        new Sgd(0.1).Step([dense.Bias], ZeroGradients(dense.Bias), 0.1, _backend);

        AssertClose([0.6, 0.8], dense.Bias.Value);
    }

    [Fact]
    public void ANormalisationWhoseNumbersAreHeld_UsesItsRunningStatisticsInATrainingPass_AndMovesNone()
    {
        var norm = new BatchNorm(2);
        var rows = Tensor.From(new Shape(4, 2), [1f, 5f, 2f, 6f, 3f, 7f, 4f, 8f]);
        var training = Pass.Training(_backend, new RandomStream(1), 0, 0);

        norm.Weight!.Rules = new SlotRules { Held = true };
        norm.Bias!.Rules = new SlotRules { Held = true };

        var held = norm.Forward(rows, training);

        // The statistics start at a mean of nought and a variance of one, and a held normalisation answers from them as it
        // does when it is not training — and measures nothing.
        Assert.Equal<float>([0f, 0f], norm.RunningMean.Value.Values.ToArray());
        Assert.Equal<float>([1f, 1f], norm.RunningVariance.Value.Values.ToArray());
        Assert.Equal(new BatchNorm(2).Forward(rows, Pass.Evaluation(_backend)).Values.ToArray(), held.Values.ToArray());
    }

    [Fact]
    public void ANormalisationWhoseNumbersLearn_MeasuresTheBatchAsItAlwaysDid()
    {
        var norm = new BatchNorm(2);

        norm.Forward(Tensor.From(new Shape(4, 2), [1f, 5f, 2f, 6f, 3f, 7f, 4f, 8f]), Pass.Training(_backend, new RandomStream(1), 0, 0));

        Assert.NotEqual<float>([0f, 0f], norm.RunningMean.Value.Values.ToArray());
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void APenaltyBelowNothingOrOfNoFiniteNumber_AndABoundOfNothingOrNoFiniteNumber_AreRefused(double wrong)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SlotRules { L1 = wrong });
        Assert.Throws<ArgumentOutOfRangeException>(() => new SlotRules { L2 = wrong });
        Assert.Throws<ArgumentOutOfRangeException>(() => new SlotRules { MaxNorm = wrong });
        Assert.Throws<ArgumentOutOfRangeException>(() => new SlotRules { MaxNorm = 0 });
    }

    private (Dense Hidden, Dense Output, LayerStack Network) Walked()
    {
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());

        return (hidden, output, new LayerStack(hidden, new Relu(), output));
    }

    private static TrainingData Rows() => new(WalkedRows.Passengers(), WalkedRows.Survived());

    // Gradients of nothing for the slots handed: a step then moves them by nothing, and a bound shows alone.
    private Gradients ZeroGradients(params Parameter[] parameters)
    {
        var recording = new RecordingBackend(_backend);
        var total = parameters.Select(parameter => recording.Mean(parameter.Value)).Aggregate((all, each) => recording.Add(all, each));

        return recording.GradientsOf(recording.Scale(total, recording.Fill(new Shape(), 0f)), parameters.Select(parameter => parameter.Value));
    }

    private static double[] Doubles(JsonElement written, string key) => [.. written.GetProperty(key).EnumerateArray().Select(value => value.GetDouble())];

    private static void AssertClose(double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual.Values[at], Math.Max(1e-9, Math.Abs(expected[at]) * 2e-5));
        }
    }
}
