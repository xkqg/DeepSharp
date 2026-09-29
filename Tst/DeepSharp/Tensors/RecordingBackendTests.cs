// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// A network learns by knowing which way to move each of its numbers. The recording backend is handed in for one pass,
/// writes down every operation as it runs it on the backend it wraps, and then works back from the loss to how much
/// each number the pass read moved it. Every gradient is checked against the loss nudged a little either way.
/// </summary>
public class RecordingBackendTests
{
    private const float Nudge = 1e-2f;

    private const double Tolerance = 1e-3;

    private readonly ITensorBackend _backend = new CpuBackend();

    // Two days of the published price series on minus one to one, a column of two weights, one bias, and the return
    // one day ahead that each day should have predicted.
    private static Tensor Days() => Tensor.From(new Shape(2, 2), [0.7993437f, -0.5730045f, 0.7866853f, -0.6739606f]);

    private static Tensor Weights() => Tensor.From(new Shape(2, 1), [0.1f, -0.2f]);

    private static Tensor Bias() => Tensor.From(new Shape(1), [0f]);

    private static Tensor Returns() => Tensor.From(new Shape(2, 1), [-0.0020976f, 0.0081744f]);

    [Fact]
    public void TheGradientOfTheSquaredMisses_ForTheWeightsAndTheBias_IsWhatTheLossNudgedEitherWaySays()
    {
        var weights = Weights();
        var bias = Bias();
        var pass = new RecordingBackend(_backend);

        var answers = pass.AddRow(pass.MatMul(Days(), weights), bias);
        var misses = pass.Subtract(answers, Returns());
        var loss = pass.Mean(pass.Multiply(misses, misses));

        var gradients = pass.GradientsOf(loss, [weights, bias]);

        // Worked out in double precision by nudging each number a millionth either way.
        Assert.Equal(new Shape(2, 1), gradients[weights].Shape);
        Assert.Equal(0.3186729327, gradients[weights].Values[0], 6);
        Assert.Equal(-0.2510263867, gradients[weights].Values[1], 6);
        Assert.Equal(new Shape(1), gradients[bias].Shape);
        Assert.Equal(0.4019191364, gradients[bias].Values[0], 6);
    }

    [Fact]
    public void ANumberReadTwice_MovesTheLossByBothReadings()
    {
        var three = Tensor.From(new Shape(), [3f]);
        var pass = new RecordingBackend(_backend);

        var loss = pass.Multiply(three, three);

        Assert.Equal(6f, pass.GradientsOf(loss, [three])[three].Values[0]);
    }

    [Fact]
    public void AddingTwoTensors_PassesTheGradientToBoth() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Add(inputs[0], inputs[1]), Random(2, 3), Random(2, 3));

    [Fact]
    public void Subtracting_PassesTheGradientOnAndItsOppositeBack() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Subtract(inputs[0], inputs[1]), Random(2, 3), Random(2, 3));

    [Fact]
    public void Multiplying_PassesEachSideTheGradientTimesTheOther() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Multiply(inputs[0], inputs[1]), Random(2, 3), Random(2, 3));

    [Fact]
    public void MultiplyingMatrices_PassesEachSideTheGradientThroughTheOtherTurned() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.MatMul(inputs[0], inputs[1]), Random(2, 3), Random(3, 4));

    [Fact]
    public void Transposing_TurnsTheGradientBack() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Transpose(inputs[0]), Random(2, 3));

    [Fact]
    public void AddingARowToEveryRow_PassesTheRowTheSumOfTheRows() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.AddRow(inputs[0], inputs[1]), Random(3, 2), Random(2));

    [Fact]
    public void SummingTheRows_PassesEveryRowTheSameGradient() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.SumRows(inputs[0]), Random(3, 2));

    [Fact]
    public void TheMean_PassesEveryValueAnEqualShare() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Mean(inputs[0]), Random(2, 3));

    [Fact]
    public void Scaling_PassesTheValuesTheFactorAndTheFactorTheValues() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Scale(inputs[0], inputs[1]), Random(2, 3), Random());

    [Fact]
    public void ScalingNoValues_LeavesTheFactorAGradientOfNothing()
    {
        var factor = Tensor.From(new Shape(), [2f]);
        var pass = new RecordingBackend(_backend);

        var loss = pass.Mean(pass.SumRows(pass.Scale(Tensor.Zeros(new Shape(0, 1)), factor)));

        Assert.Equal(0f, pass.GradientsOf(loss, [factor])[factor].Values[0]);
    }

    [Fact]
    public void ALossWithAxes_IsRefused()
    {
        var weights = Weights();
        var pass = new RecordingBackend(_backend);

        var answers = pass.MatMul(Days(), weights);

        var wrong = Assert.Throws<ArgumentException>(() => pass.GradientsOf(answers, [weights]));
        Assert.Contains("2x1", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALossThisPassDidNotWorkOut_IsRefused()
    {
        var weights = Weights();
        var elsewhere = _backend.Mean(_backend.MatMul(Days(), weights));

        Assert.Throws<ArgumentException>(() => new RecordingBackend(_backend).GradientsOf(elsewhere, [weights]));
    }

    [Fact]
    public void ANumberThePassNeverRead_IsRefusedRatherThanGivenNothing()
    {
        var weights = Weights();
        var unread = Bias();
        var pass = new RecordingBackend(_backend);

        var loss = pass.Mean(pass.MatMul(Days(), weights));

        var wrong = Assert.Throws<ArgumentException>(() => pass.GradientsOf(loss, [weights, unread]));
        Assert.Contains("never read", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANumberReadWhereItDoesNotReachTheLoss_HasAGradientOfNothing()
    {
        var weights = Weights();
        var aside = Bias();
        var pass = new RecordingBackend(_backend);

        _ = pass.Add(aside, aside);
        var loss = pass.Mean(pass.MatMul(Days(), weights));

        Assert.Equal<float[]>([0f], pass.GradientsOf(loss, [weights, aside])[aside].Values.ToArray());
    }

    [Fact]
    public void AGradientNotAskedFor_IsRefused()
    {
        var weights = Weights();
        var pass = new RecordingBackend(_backend);

        var gradients = pass.GradientsOf(pass.Mean(pass.MatMul(Days(), weights)), [weights]);

        Assert.Throws<ArgumentException>(() => gradients[Bias()]);
    }

    [Fact]
    public void RecordingChangesNothingTheWrappedBackendWorksOut()
    {
        var pass = new RecordingBackend(_backend);

        Assert.Equal(_backend.Name, pass.Name);
        Assert.Equal<float[]>(
            _backend.MatMul(Days(), Weights()).Values.ToArray(),
            pass.MatMul(Days(), Weights()).Values.ToArray());
        Assert.Equal<float[]>([0.5f, 0.5f], pass.Fill(new Shape(2), 0.5f).Values.ToArray());
    }

    // The loss is the operation's result weighted by fixed numbers and averaged, so every value of it counts, each by
    // its own weight. Each input's gradient is then compared with the loss nudged a little either way, one value at a
    // time, on the wrapped backend alone: the recording is never used to check itself.
    private void AssertGradientsMatchTheLossNudged(Func<ITensorBackend, Tensor[], Tensor> operation, params Tensor[] inputs)
    {
        var weighing = Random(operation(_backend, inputs).Shape);

        Tensor Loss(ITensorBackend backend, Tensor[] values) => backend.Mean(backend.Multiply(operation(backend, values), weighing));

        var pass = new RecordingBackend(_backend);
        var gradients = pass.GradientsOf(Loss(pass, inputs), inputs);

        for (var input = 0; input < inputs.Length; input++)
        {
            for (var at = 0; at < inputs[input].Values.Length; at++)
            {
                var up = Loss(_backend, Nudged(inputs, input, at, Nudge)).Values[0];
                var down = Loss(_backend, Nudged(inputs, input, at, -Nudge)).Values[0];

                Assert.Equal((up - down) / (2 * Nudge), gradients[inputs[input]].Values[at], Tolerance);
            }
        }
    }

    private static Tensor[] Nudged(Tensor[] inputs, int input, int at, float by)
    {
        var values = inputs[input].Values.ToArray();
        values[at] += by;

        return [.. inputs.Select((each, place) => place == input ? Tensor.From(each.Shape, values) : each)];
    }

    private static Tensor Random(params int[] axes) => Random(new Shape(axes));

    private static Tensor Random(Shape shape)
    {
        var generator = new Random(shape.Count + 20260929);

        return Tensor.From(shape, [.. Enumerable.Range(0, shape.Count).Select(_ => (generator.NextSingle() * 2) - 1)]);
    }
}
