// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// Matrices stacked on their rows as so many blocks, multiplied block by block: what attention needs of a batch of series, which a product
/// by a vector for each series makes cost the square of the batch. Whichever engine it is asked of, and whether the engine has the product
/// itself or the library works it out from the products every engine has, block <c>i</c> of the answer is block <c>i</c> of the first, turned
/// or not, times block <c>i</c> of the second, turned or not.
/// </summary>
/// <param name="engine">The engine the arithmetic runs on.</param>
public abstract class BlockProductContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    public static TheoryData<int, int, int, int, bool, bool> Products => new()
    {
        { 1, 3, 4, 2, false, false },
        { 3, 2, 4, 3, false, false },
        { 3, 2, 4, 3, false, true },
        { 3, 2, 4, 3, true, false },
        { 3, 2, 4, 3, true, true },
        { 4, 1, 5, 1, false, true },
        { 2, 5, 17, 5, false, true },
        { 5, 3, 1, 4, false, false },
        { 2, 9, 8, 33, true, true },
    };

    [Theory]
    [MemberData(nameof(Products))]
    public void EveryBlockOfTheAnswer_IsTheProductOfTheBlocksItIsMadeOf(int count, int rows, int inner, int columns, bool turnLeft, bool turnRight)
    {
        var (left, right) = Operands(count, rows, inner, columns, turnLeft, turnRight);
        var how = new BlockProduct(count) { TurnLeft = turnLeft, TurnRight = turnRight };

        var made = _backend.MatMulInBlocks(left, right, how);

        Assert.Equal(new Shape(count * rows, columns), made.Shape);
        AssertAlike(Reference(left, right, how), made.Values.ToArray());
    }

    [Theory]
    [MemberData(nameof(Products))]
    public void AnEngineWithoutTheProduct_IsAskedForTheProductsItHas_AndTheAnswerIsTheSame(int count, int rows, int inner, int columns, bool turnLeft, bool turnRight)
    {
        var (left, right) = Operands(count, rows, inner, columns, turnLeft, turnRight);
        var how = new BlockProduct(count) { TurnLeft = turnLeft, TurnRight = turnRight };
        var without = new NotingBackend(_backend);

        var made = without.MatMulInBlocks(left, right, how);

        Assert.Equal(new Shape(count * rows, columns), made.Shape);
        AssertAlike(Reference(left, right, how), made.Values.ToArray());
        Assert.True(without.Count(nameof(ITensorBackend.MatMul)) >= count);
    }

    [Fact]
    public void ABatchTheRowsDoNotDivideInto_IsRefused_AndTheMessageNamesTheShapes()
    {
        var refused = Assert.Throws<ArgumentException>(() => _backend.MatMulInBlocks(Tensor.Zeros(new Shape(5, 2)), Tensor.Zeros(new Shape(6, 2)), new BlockProduct(3)));

        Assert.Contains("5x2", refused.Message, StringComparison.Ordinal);
        Assert.Contains("3", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BlocksThatDoNotMeetInTheMiddle_AreRefused_AndTheMessageNamesTheShapes()
    {
        var refused = Assert.Throws<ArgumentException>(() => _backend.MatMulInBlocks(Tensor.Zeros(new Shape(6, 4)), Tensor.Zeros(new Shape(9, 3)), new BlockProduct(3)));

        Assert.Contains("2x4", refused.Message, StringComparison.Ordinal);
        Assert.Contains("3x3", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void LessThanOneBlock_IsRefused() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => _backend.MatMulInBlocks(Tensor.Zeros(new Shape(2, 2)), Tensor.Zeros(new Shape(2, 2)), new BlockProduct(0)));

    [Fact]
    public void ATensorThatIsNotAMatrix_IsRefused() =>
        Assert.Throws<ArgumentException>(() => _backend.MatMulInBlocks(Tensor.Zeros(new Shape(2, 2, 2)), Tensor.Zeros(new Shape(2, 2)), new BlockProduct(1)));

    // Block-wise stacked operands: the blocks of the first are rows by inner (or inner by rows, turned), the second's inner by columns (or the reverse).
    private static (Tensor Left, Tensor Right) Operands(int count, int rows, int inner, int columns, bool turnLeft, bool turnRight) => (
        Numbers(new Shape(count * (turnLeft ? inner : rows), turnLeft ? rows : inner), 0.37),
        Numbers(new Shape(count * (turnRight ? columns : inner), turnRight ? inner : columns), 0.91));

    private static Tensor Numbers(Shape shape, double step) =>
        Tensor.From(shape, [.. Enumerable.Range(0, shape.Count).Select(at => (float)Math.Sin((at * step) + 0.2))]);

    private static float[] Reference(Tensor left, Tensor right, BlockProduct how)
    {
        var leftRows = left.Shape[0] / how.Count;
        var rightRows = right.Shape[0] / how.Count;
        var rows = how.TurnLeft ? left.Shape[1] : leftRows;
        var inner = how.TurnLeft ? leftRows : left.Shape[1];
        var columns = how.TurnRight ? rightRows : right.Shape[1];
        var made = new float[how.Count * rows * columns];

        for (var block = 0; block < how.Count; block++)
        {
            for (var row = 0; row < rows; row++)
            {
                for (var column = 0; column < columns; column++)
                {
                    double total = 0;

                    for (var at = 0; at < inner; at++)
                    {
                        var first = how.TurnLeft ? left.Values[(((block * leftRows) + at) * left.Shape[1]) + row] : left.Values[(((block * leftRows) + row) * left.Shape[1]) + at];
                        var second = how.TurnRight ? right.Values[(((block * rightRows) + column) * right.Shape[1]) + at] : right.Values[(((block * rightRows) + at) * right.Shape[1]) + column];

                        total += (double)first * second;
                    }

                    made[(((block * rows) + row) * columns) + column] = (float)total;
                }
            }
        }

        return made;
    }

    private static void AssertAlike(float[] wanted, float[] got)
    {
        Assert.Equal(wanted.Length, got.Length);

        for (var at = 0; at < wanted.Length; at++)
        {
            Assert.InRange(Math.Abs(wanted[at] - got[at]), 0, 1e-5 * Math.Max(1, Math.Abs(wanted[at])));
        }
    }
}
