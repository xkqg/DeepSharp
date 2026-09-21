// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// Every arithmetic a network performs goes through a backend, so that the same model can later run its
/// heavy work somewhere else without a single line of the model changing. This is the one that ships: .NET's
/// own SIMD tensor primitives, no native library behind it.
/// </summary>
public class CpuBackendTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

    [Fact]
    public void AddingTwoTensors_AddsThemValueByValueAndKeepsTheShape()
    {
        var sum = _backend.Add(Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]),
                               Tensor.From(new Shape(2, 2), [10f, 20f, 30f, 40f]));

        Assert.Equal(new Shape(2, 2), sum.Shape);
        Assert.Equal<float[]>([11f, 22f, 33f, 44f], sum.Values.ToArray());
    }

    [Fact]
    public void MultiplyingTwoTensors_MultipliesThemValueByValue()
    {
        var product = _backend.Multiply(Tensor.From(new Shape(3), [2f, 3f, 4f]),
                                        Tensor.From(new Shape(3), [5f, 6f, 7f]));

        Assert.Equal<float[]>([10f, 18f, 28f], product.Values.ToArray());
    }

    [Fact]
    public void TensorsOfDifferentShapes_AreRefusedAndTheMessageNamesBoth()
    {
        var wrong = Assert.Throws<ArgumentException>(
            () => _backend.Add(Tensor.Zeros(new Shape(2, 3)), Tensor.Zeros(new Shape(3, 2))));

        Assert.Contains("2x3", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("3x2", wrong.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(1000)]
    public void EveryValue_IsAddedWhateverTheLength(int length)
    {
        // SIMD works a register at a time and the tail is handled separately, so a length that is not a whole
        // number of registers is exactly where an off-by-one hides — and it hides quietly, because the first
        // few thousand values are right.
        float[] left = [.. Enumerable.Range(0, length).Select(i => (float)i)];
        float[] right = [.. Enumerable.Range(0, length).Select(i => i * 2f)];

        var sum = _backend.Add(Tensor.From(new Shape(length), left), Tensor.From(new Shape(length), right));

        for (int i = 0; i < length; i++)
        {
            Assert.Equal(i * 3f, sum.Values[i]);
        }
    }

    [Fact]
    public void AddingDoesNotChangeEitherSide()
    {
        var left = Tensor.From(new Shape(2), [1f, 2f]);
        var right = Tensor.From(new Shape(2), [3f, 4f]);

        _backend.Add(left, right);

        Assert.Equal<float[]>([1f, 2f], left.Values.ToArray());
        Assert.Equal<float[]>([3f, 4f], right.Values.ToArray());
    }

    [Fact]
    public void AnEmptyTensor_AddsToAnEmptyTensor()
    {
        var sum = _backend.Add(Tensor.Zeros(new Shape(0)), Tensor.Zeros(new Shape(0)));

        Assert.Equal(0, sum.Values.Length);
    }

    [Fact]
    public void TheBackend_SaysWhichOneItIs()
    {
        Assert.Equal("cpu", _backend.Name);
    }
}
