// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// The operations that bend a straight line: what an activation, a loss and an optimizer are made of. Every expected
/// value is PyTorch's for the same input, to the last digit a float carries or within the rounding its vector
/// arithmetic allows.
/// </summary>
public abstract partial class TensorBackendContract
{
    // A value either side of nothing, nothing itself, and three above it.
    private static Tensor Mixed() => Tensor.From(new Shape(2, 3), [-2f, -0.5f, 0f, 0.25f, 1f, 3f]);

    // Values a logarithm, a root and a division are defined for.
    private static Tensor Positives() => Tensor.From(new Shape(2, 3), [0.5f, 1f, 4f, 0.1f, 2f, 9f]);

    [Fact]
    public void Relu_KeepsWhatIsAboveNothing_AndMakesTheRestNothing()
    {
        var kept = _backend.Relu(Mixed());

        Assert.Equal(new Shape(2, 3), kept.Shape);
        Assert.Equal<float[]>([0f, 0f, 0f, 0.25f, 1f, 3f], kept.Values.ToArray());
    }

    [Fact]
    public void Positive_IsOneWhereAValueIsAboveNothing_AndNoughtElsewhere_NothingIncluded()
    {
        var steps = _backend.Positive(Mixed());

        Assert.Equal<float[]>([0f, 0f, 0f, 1f, 1f, 1f], steps.Values.ToArray());
    }

    [Fact]
    public void FirstLargest_IsOneWhereAValueIsTheLargestOfItsRow_AndNoughtElsewhere()
    {
        var picked = _backend.FirstLargest(Tensor.From(new Shape(3, 3), [1f, 3f, 2f, -1f, -2f, -3f, 4f, 0f, 9f]));

        Assert.Equal(new Shape(3, 3), picked.Shape);
        Assert.Equal<float[]>([0f, 1f, 0f, 1f, 0f, 0f, 0f, 0f, 1f], picked.Values.ToArray());
    }

    [Fact]
    public void FirstLargest_PicksTheFirstOfValuesThatTie_SoAGradientGoesToOnePlaceAlone()
    {
        var picked = _backend.FirstLargest(Tensor.From(new Shape(3, 4), [2f, 2f, 2f, 2f, 1f, 5f, 5f, 0f, -1f, -1f, -3f, -1f]));

        Assert.Equal<float[]>([1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 1f, 0f, 0f, 0f], picked.Values.ToArray());
    }

    [Fact]
    public void FirstLargest_TakesAValueThatIsNotANumberAsTheLargest_AsPyTorchsArgmaxDoes()
    {
        var picked = _backend.FirstLargest(Tensor.From(new Shape(3, 3), [1f, float.NaN, 5f, float.NaN, float.NaN, 7f, float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity]));

        Assert.Equal<float[]>([0f, 1f, 0f, 1f, 0f, 0f, 1f, 0f, 0f], picked.Values.ToArray());
    }

    [Fact]
    public void FirstLargest_OfARowOfOneValue_IsOne_AndOfNoRows_IsNoRows()
    {
        Assert.Equal<float[]>([1f, 1f, 1f], _backend.FirstLargest(Tensor.From(new Shape(3, 1), [-5f, 0f, 9f])).Values.ToArray());
        Assert.Equal(new Shape(0, 4), _backend.FirstLargest(Tensor.Zeros(new Shape(0, 4))).Shape);
    }

    [Fact]
    public void FirstLargest_MultipliedIntoTheValues_AndAddedUpAlongTheRow_IsTheLargestOfEachRow()
    {
        var values = Tensor.From(new Shape(2, 4), [3f, -1f, 3f, 2f, -4f, -2f, -9f, -2f]);

        var largest = _backend.MatMul(_backend.Multiply(values, _backend.FirstLargest(values)), _backend.Fill(new Shape(4, 1), 1f));

        Assert.Equal(new Shape(2, 1), largest.Shape);
        Assert.Equal<float[]>([3f, -2f], largest.Values.ToArray());
    }

    [Fact]
    public void Tanh_IsPyTorchsTanh()
    {
        AssertClose([-0.9640275835990906, -0.46211716532707214, 0.0, 0.24491865932941437, 0.7615941762924194, 0.9950547814369202], _backend.Tanh(Mixed()));
    }

    [Fact]
    public void Sigmoid_IsPyTorchsSigmoid()
    {
        AssertClose([0.11920291930437088, 0.3775406777858734, 0.5, 0.562176525592804, 0.7310585975646973, 0.9525741338729858], _backend.Sigmoid(Mixed()));
    }

    [Fact]
    public void Exp_IsPyTorchsExponential()
    {
        AssertClose([0.1353352814912796, 0.6065306663513184, 1.0, 1.2840254306793213, 2.7182817459106445, 20.08553695678711], _backend.Exp(Mixed()));
    }

    [Fact]
    public void Log_IsPyTorchsNaturalLogarithm()
    {
        AssertClose([-0.6931471824645996, 0.0, 1.3862943649291992, -2.3025851249694824, 0.6931471824645996, 2.1972246170043945], _backend.Log(Positives()));
    }

    [Fact]
    public void Sqrt_IsPyTorchsSquareRoot()
    {
        AssertClose([0.7071067690849304, 1.0, 2.0, 0.3162277638912201, 1.4142135381698608, 3.0], _backend.Sqrt(Positives()));
    }

    [Fact]
    public void Dividing_DividesValueByValue()
    {
        AssertClose([-4.0, -0.5, 0.0, 2.5, 0.5, 0.3333333432674408], _backend.Divide(Mixed(), Positives()));
    }

    [Fact]
    public void DividingTensorsOfDifferentShapes_IsRefusedAsAddingThemIs()
    {
        Assert.Throws<ArgumentException>(() => _backend.Divide(Tensor.Zeros(new Shape(2)), Tensor.Zeros(new Shape(3))));
    }

    [Fact]
    public void Softplus_IsTheLogOfOnePlusTheExponential_AtEveryScale()
    {
        // At minus forty the answer is four quintillionths, not nothing: a sum of one and a number that small, taken as
        // it is, loses it entirely. At forty it is forty, not the infinity an exponential taken first would give.
        var softplus = _backend.Softplus(Tensor.From(new Shape(6), [-40f, -30f, -1f, 0f, 2f, 40f]));

        Assert.Equal(4.24835413113866e-18, softplus.Values[0], 4.24835413113866e-18 * 1e-6);
        Assert.Equal(9.357622912219837e-14, softplus.Values[1], 9.357622912219837e-14 * 1e-6);
        Assert.Equal(0.3132616877555847, softplus.Values[2], 1e-7);
        Assert.Equal(0.6931471824645996, softplus.Values[3], 1e-7);
        Assert.Equal(2.1269280910491943, softplus.Values[4], 1e-6);
        Assert.Equal(40f, softplus.Values[5]);
    }

    [Fact]
    public void LogSoftmax_IsEachRowMinusTheLogOfTheSumOfItsExponentials()
    {
        var logs = _backend.LogSoftmax(Mixed());

        Assert.Equal(new Shape(2, 3), logs.Shape);
        AssertClose([-2.5549569129943848, -1.0549569129943848, -0.55495685338974, -2.9317073822021484, -2.1817073822021484, -0.18170727789402008], logs);
    }

    [Fact]
    public void LogSoftmax_OfValuesInTheThousands_IsStillTheLogOfTheirShares()
    {
        // Taken as the exponential first, a thousand is an infinity and every share is not a number. Each row is shifted by
        // its largest value before anything is exponentiated.
        var logs = _backend.LogSoftmax(Tensor.From(new Shape(1, 3), [1000f, 1001f, 1002f]));

        AssertClose([-2.4076058864593506, -1.4076058864593506, -0.40760594606399536], logs);
    }

    [Fact]
    public void OnlyTheRowsOfAMatrix_TakeALogSoftmax()
    {
        Assert.Throws<ArgumentException>(() => _backend.LogSoftmax(Tensor.Zeros(new Shape(3))));
    }

    [Fact]
    public void TheElementwiseOperations_LeaveTheirInputAsItWas()
    {
        var values = Mixed();
        Func<Tensor, Tensor>[] operations = [_backend.Relu, _backend.Positive, _backend.Tanh, _backend.Sigmoid, _backend.Exp, _backend.Softplus, _backend.LogSoftmax, _backend.FirstLargest];

        foreach (var result in operations.Select(operation => operation(values)))
        {
            Assert.Equal(values.Shape, result.Shape);
        }

        Assert.Equal<float[]>([-2f, -0.5f, 0f, 0.25f, 1f, 3f], values.Values.ToArray());
    }

    // Within a millionth of the value, or of one where the value is smaller: the processor's vector instructions take
    // their own route to a transcendental, and tanh of nothing comes out as six hundred-millionths below it.
    private static void AssertClose(double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual.Values[at], Math.Max(1e-6, Math.Abs(expected[at]) * 1e-6));
        }
    }
}
