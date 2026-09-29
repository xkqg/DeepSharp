// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// Every arithmetic a network performs goes through a backend, so that the same model can later run its
/// heavy work somewhere else without a single line of the model changing. This is the one that ships: .NET's
/// own SIMD tensor primitives, no native library behind it.
/// </summary>
public partial class CpuBackendTests
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

    [Fact]
    public void MultiplyingMatrices_SumsEachRowOfTheLeftAgainstEachColumnOfTheRight()
    {
        var product = _backend.MatMul(
            Tensor.From(new Shape(2, 3), [1f, 2f, 3f, 4f, 5f, 6f]),
            Tensor.From(new Shape(3, 2), [7f, 8f, 9f, 10f, 11f, 12f]));

        Assert.Equal(new Shape(2, 2), product.Shape);
        Assert.Equal<float[]>([58f, 64f, 139f, 154f], product.Values.ToArray());
    }

    [Fact]
    public void MultiplyingTwoPricesOnMinusOneToOne_ByAColumnOfWeights_GivesOneAnswerARow()
    {
        // Two days of the published price series, closing price and volume brought onto minus one to one by the
        // training rows' own extremes, times a column of two weights.
        var rows = Tensor.From(new Shape(2, 2), [0.7993437f, -0.5730045f, 0.7866853f, -0.6739606f]);
        var weights = Tensor.From(new Shape(2, 1), [0.1f, -0.2f]);

        var answers = _backend.MatMul(rows, weights);

        Assert.Equal(new Shape(2, 1), answers.Shape);
        Assert.Equal(0.1945353, answers.Values[0], 6);
        Assert.Equal(0.2134607, answers.Values[1], 6);
    }

    [Fact]
    public void MatricesWhoseInnerLengthsDiffer_AreRefusedAndTheMessageNamesBoth()
    {
        var wrong = Assert.Throws<ArgumentException>(
            () => _backend.MatMul(Tensor.Zeros(new Shape(2, 3)), Tensor.Zeros(new Shape(2, 3))));

        Assert.Contains("2x3", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void OnlyMatrices_AreMultipliedAsMatrices()
    {
        Assert.Throws<ArgumentException>(() => _backend.MatMul(Tensor.Zeros(new Shape(3)), Tensor.Zeros(new Shape(3, 1))));
        Assert.Throws<ArgumentException>(() => _backend.MatMul(Tensor.Zeros(new Shape(1, 3)), Tensor.Zeros(new Shape(3))));
    }

    [Fact]
    public void TransposingAMatrix_TurnsItsRowsIntoColumns()
    {
        var turned = _backend.Transpose(Tensor.From(new Shape(2, 3), [1f, 2f, 3f, 4f, 5f, 6f]));

        Assert.Equal(new Shape(3, 2), turned.Shape);
        Assert.Equal<float[]>([1f, 4f, 2f, 5f, 3f, 6f], turned.Values.ToArray());
    }

    [Fact]
    public void OnlyAMatrix_IsTransposed()
    {
        Assert.Throws<ArgumentException>(() => _backend.Transpose(Tensor.Zeros(new Shape(3))));
    }

    [Fact]
    public void AddingARowToAMatrix_AddsItToEveryRow()
    {
        var shifted = _backend.AddRow(
            Tensor.From(new Shape(2, 3), [1f, 2f, 3f, 4f, 5f, 6f]),
            Tensor.From(new Shape(3), [10f, 20f, 30f]));

        Assert.Equal(new Shape(2, 3), shifted.Shape);
        Assert.Equal<float[]>([11f, 22f, 33f, 14f, 25f, 36f], shifted.Values.ToArray());
    }

    [Fact]
    public void ARowAsLongAsTheMatrixIsNotWide_IsRefusedAndTheMessageNamesBoth()
    {
        var wrong = Assert.Throws<ArgumentException>(
            () => _backend.AddRow(Tensor.Zeros(new Shape(2, 3)), Tensor.Zeros(new Shape(2))));

        Assert.Contains("2x3", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("2", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARowMustBeARow_AndTheMatrixAMatrix()
    {
        Assert.Throws<ArgumentException>(() => _backend.AddRow(Tensor.Zeros(new Shape(3)), Tensor.Zeros(new Shape(3))));
        Assert.Throws<ArgumentException>(() => _backend.AddRow(Tensor.Zeros(new Shape(2, 3)), Tensor.Zeros(new Shape(1, 3))));
    }

    [Fact]
    public void SummingTheRows_GivesOneTotalAColumn()
    {
        var totals = _backend.SumRows(Tensor.From(new Shape(2, 3), [1f, 2f, 3f, 4f, 5f, 6f]));

        Assert.Equal(new Shape(3), totals.Shape);
        Assert.Equal<float[]>([5f, 7f, 9f], totals.Values.ToArray());
    }

    [Fact]
    public void SummingTheRowsOfAColumn_GivesTheOneTotal()
    {
        var total = _backend.SumRows(Tensor.From(new Shape(2, 1), [0.19663289f, 0.20528625f]));

        Assert.Equal(new Shape(1), total.Shape);
        Assert.Equal(0.4019191, total.Values[0], 6);
    }

    [Fact]
    public void OnlyTheRowsOfAMatrix_AreSummed()
    {
        Assert.Throws<ArgumentException>(() => _backend.SumRows(Tensor.Zeros(new Shape(3))));
    }

    [Fact]
    public void SubtractingTwoTensors_SubtractsThemValueByValue()
    {
        var difference = _backend.Subtract(Tensor.From(new Shape(2), [5f, 7f]), Tensor.From(new Shape(2), [2f, 3f]));

        Assert.Equal<float[]>([3f, 4f], difference.Values.ToArray());
    }

    [Fact]
    public void SubtractingTensorsOfDifferentShapes_IsRefusedAsAddingThemIs()
    {
        Assert.Throws<ArgumentException>(() => _backend.Subtract(Tensor.Zeros(new Shape(2)), Tensor.Zeros(new Shape(3))));
    }

    [Fact]
    public void TheMean_IsOneValueWithNoAxes()
    {
        // The squared misses of two answers: what a loss is, a single value.
        var misses = Tensor.From(new Shape(2, 1), [0.19663289f, 0.20528625f]);

        var loss = _backend.Mean(_backend.Multiply(misses, misses));

        Assert.Equal(new Shape(), loss.Shape);
        Assert.Equal(0.04040347, loss.Values[0], 7);
    }

    [Fact]
    public void TheMeanOfTenMillionTenths_IsATenth()
    {
        // Added up in single precision, the running total stops taking small values in once it is large and the
        // mean comes out as 0.1087937; the total is kept in double precision so it does not.
        var tenths = _backend.Fill(new Shape(10_000_000), 0.1f);

        Assert.Equal(0.1f, _backend.Mean(tenths).Values[0]);
    }

    [Fact]
    public void TheMeanOfNothing_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => _backend.Mean(Tensor.Zeros(new Shape(0))));
    }

    [Fact]
    public void ScalingByOneValue_MultipliesEveryValueByIt()
    {
        var scaled = _backend.Scale(Tensor.From(new Shape(3), [1f, 2f, 3f]), Tensor.From(new Shape(), [2f]));

        Assert.Equal(new Shape(3), scaled.Shape);
        Assert.Equal<float[]>([2f, 4f, 6f], scaled.Values.ToArray());
    }

    [Fact]
    public void AFactorOfMoreThanOneValue_IsRefused()
    {
        var wrong = Assert.Throws<ArgumentException>(
            () => _backend.Scale(Tensor.Zeros(new Shape(3)), Tensor.Zeros(new Shape(1))));

        Assert.Contains("1", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Filling_GivesEveryValueTheSameOne()
    {
        var filled = _backend.Fill(new Shape(2, 2), 0.5f);

        Assert.Equal(new Shape(2, 2), filled.Shape);
        Assert.Equal<float[]>([0.5f, 0.5f, 0.5f, 0.5f], filled.Values.ToArray());
    }

    [Fact]
    public void FillingNoAxes_GivesOneValue()
    {
        // Where a backward pass starts: the loss's own gradient, one.
        var seed = _backend.Fill(new Shape(), 1f);

        Assert.Equal(new Shape(), seed.Shape);
        Assert.Equal<float[]>([1f], seed.Values.ToArray());
    }
}
