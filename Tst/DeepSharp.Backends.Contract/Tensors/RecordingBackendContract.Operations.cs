// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// Every operation a layer, a loss or an optimizer is made of sends its gradient back by a rule of its own, and each
/// rule is held to the loss nudged a little either way. The values are drawn where the operation is smooth: a
/// logarithm, a root and a division away from nothing, a step and a rectifier a nudge or more either side of it.
/// </summary>
public abstract partial class RecordingBackendContract
{
    [Fact]
    public void Relu_PassesTheGradientWhereTheValueWasAboveNothing() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Relu(inputs[0]), AwayFromNothing(3, 4));

    [Fact]
    public void Positive_PassesNothingBack_ForAStepIsFlatEitherSideOfIt() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Positive(inputs[0]), AwayFromNothing(3, 4));

    [Fact]
    public void Tanh_PassesTheGradientTimesOneLessItsSquare() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Tanh(inputs[0]), Random(3, 4));

    [Fact]
    public void Sigmoid_PassesTheGradientTimesItselfTimesOneLessItself() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Sigmoid(inputs[0]), Random(3, 4));

    [Fact]
    public void Exp_PassesTheGradientTimesItself() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Exp(inputs[0]), Between(-3.5f, 3.5f, 3, 4));

    [Fact]
    public void Log_PassesTheGradientOverTheValue() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Log(inputs[0]), Between(0.3f, 3f, 3, 4));

    [Fact]
    public void Sqrt_PassesTheGradientOverTwiceTheRoot() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Sqrt(inputs[0]), Between(0.3f, 3f, 3, 4));

    [Fact]
    public void Softplus_PassesTheGradientTimesTheSigmoid() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Softplus(inputs[0]), Between(-3f, 3f, 3, 4));

    [Fact]
    public void Dividing_PassesTheTopTheGradientOverTheBottom_AndTheBottomMinusTheQuotientOverItself() =>
        AssertGradientsMatchTheLossNudged(
            (backend, inputs) => backend.Divide(inputs[0], inputs[1]), Between(0.3f, 3f, 2, 3), Between(0.3f, 3f, 2, 3).Drawn(anew: 1));

    [Fact]
    public void LogSoftmax_PassesTheGradientLessEachShareTimesTheRowsGradient() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.LogSoftmax(inputs[0]), Random(3, 4));

    [Fact]
    public void Reshaping_PassesTheGradientBackInTheOldShape() =>
        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Reshape(inputs[0], new Shape(3, 2)), Random(2, 3));

    [Fact]
    public void Unfolding_PassesEveryPatchsGradientBackToWhereItsValueCameFrom() =>
        AssertGradientsMatchTheLossNudged(
            (backend, inputs) => backend.Unfold(inputs[0], new Window(3, 3) { Stride = 2, Padding = 1 }), Random(2, 4, 3, 2));

    [Fact]
    public void Unfolding_PaddedAsSame_PassesEveryPatchsGradientBackToWhereItsValueCameFrom() =>
        AssertGradientsMatchTheLossNudged(
            (backend, inputs) => backend.Unfold(inputs[0], new Window(3, 2) { Stride = 2, PaddingMode = PaddingMode.Same }), Random(2, 4, 5, 2));

    [Fact]
    public void Folding_ByAWindowPaddedAsSame_PassesEachPlacesGradientToEveryPatchThatCoveredIt()
    {
        var window = new Window(2, 3) { PaddingMode = PaddingMode.Same };
        var image = new Shape(1, 3, 2, 2);

        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Fold(inputs[0], image, window), Random(6, 12));
    }

    [Fact]
    public void Folding_PassesEachPlacesGradientToEveryPatchThatCoveredIt()
    {
        var window = new Window(2, 2) { Padding = 1 };
        var image = new Shape(1, 3, 2, 2);

        AssertGradientsMatchTheLossNudged((backend, inputs) => backend.Fold(inputs[0], image, window), Random(12, 8));
    }

    [Fact]
    public void RecordingChangesNothingTheWrappedBackendWorksOut_ForTheNewOperationsEither()
    {
        var pass = new RecordingBackend(_backend);
        var values = Between(0.3f, 3f, 2, 3);

        foreach (var (recorded, plain) in new (Func<Tensor, Tensor> Recorded, Func<Tensor, Tensor> Plain)[]
                 {
                     (pass.Relu, _backend.Relu), (pass.Positive, _backend.Positive), (pass.Tanh, _backend.Tanh),
                     (pass.Sigmoid, _backend.Sigmoid), (pass.Exp, _backend.Exp), (pass.Log, _backend.Log), (pass.Sqrt, _backend.Sqrt),
                     (pass.Softplus, _backend.Softplus), (pass.LogSoftmax, _backend.LogSoftmax),
                 })
        {
            Assert.Equal<float[]>(plain(values).Values.ToArray(), recorded(values).Values.ToArray());
        }
    }

    // Values between minus one and one, none nearer to nothing than a twentieth, so a nudge never crosses it.
    private static Tensor AwayFromNothing(params int[] axes)
    {
        var drawn = Random(axes);

        return Tensor.From(drawn.Shape, [.. drawn.Values.ToArray().Select(value => MathF.CopySign(0.05f + (0.95f * MathF.Abs(value)), value))]);
    }

    private static Tensor Between(float low, float high, params int[] axes)
    {
        var drawn = Random(axes);

        return Tensor.From(drawn.Shape, [.. drawn.Values.ToArray().Select(value => low + ((high - low) * (value + 1) / 2))]);
    }
}

/// <summary>Draws a tensor's values again, so two inputs of one shape are not the same numbers.</summary>
internal static class DrawnTensorExtensions
{
    public static Tensor Drawn(this Tensor tensor, int anew)
    {
        var generator = new Random(anew);
        var low = tensor.Values.ToArray().Min();
        var high = tensor.Values.ToArray().Max();

        return Tensor.From(tensor.Shape, [.. Enumerable.Range(0, tensor.Shape.Count).Select(_ => low + ((high - low) * generator.NextSingle()))]);
    }
}
