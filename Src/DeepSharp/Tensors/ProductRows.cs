// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace DeepSharp.Tensors;

/// <summary>How big a matrix product is: the rows of the left matrix, the length each row is added up over, and the columns of the right one.</summary>
/// <param name="Rows">The rows of the left matrix, and of the product.</param>
/// <param name="Inner">How many terms each total adds up.</param>
/// <param name="Columns">The columns of the right matrix, and of the product.</param>
internal readonly record struct ProductShape(int Rows, int Inner, int Columns);

/// <summary>
/// The rows of a matrix product, worked out as the light engine works out every total: a row at a time, each total kept in
/// double precision, the terms of a total added in the order they stand.
/// </summary>
/// <remarks>
/// The one place that arithmetic is written. A product is this over all its rows, and a product spread over several threads
/// is this over a range of rows each, so what carries it never changes what it comes to. A term is a float times a float,
/// which a double holds exactly, so the totals are the sums the rows' definition gives; the order they are added in is what the
/// bits depend on, and it is never changed, whatever the width of the processor's vectors.
/// </remarks>
internal readonly ref struct ProductRows
{
    private readonly ReadOnlySpan<float> _left;
    private readonly ReadOnlySpan<float> _right;
    private readonly ProductShape _shape;
    private readonly bool _turnedLeft;

    /// <summary>The rows of the product of two matrices.</summary>
    /// <param name="left">The left matrix, row-major.</param>
    /// <param name="right">The right matrix, row-major.</param>
    /// <param name="shape">How big the product is.</param>
    /// <param name="turnedLeft">Whether the left matrix lies turned on its side: its rows are the product's columns of terms, read down instead of across.</param>
    public ProductRows(ReadOnlySpan<float> left, ReadOnlySpan<float> right, ProductShape shape, bool turnedLeft = false)
    {
        _turnedLeft = turnedLeft;
        _left = left;
        _right = right;
        _shape = shape;
    }

    /// <summary>Works out some rows of the product.</summary>
    /// <param name="result">The product, row-major, as long as the whole of it; only the rows asked for are written.</param>
    /// <param name="firstRow">The first row, counted from nought.</param>
    /// <param name="lastRow">One past the last row.</param>
    /// <remarks>
    /// A row's totals are added to a vector's width of columns at a time: each total is still the sum of its own terms, in the
    /// order the terms stand, so the lanes change how many totals one instruction carries and nothing about any one of them. A
    /// total that comes out as no number is worked out again by the plain loop, because which of two not-a-numbers a sum
    /// keeps is the instruction's to say, and the plain loop is the one the bits were defined by.
    /// </remarks>
    public void Work(Span<float> result, int firstRow, int lastRow)
    {
        var totals = new double[_shape.Columns];

        for (var row = firstRow; row < lastRow; row++)
        {
            Row(totals, row);

            if (Vector.IsHardwareAccelerated && HoldsNoNumber(totals))
            {
                Row(totals, row, lanes: false);
            }

            for (var column = 0; column < _shape.Columns; column++)
            {
                result[(row * _shape.Columns) + column] = (float)totals[column];
            }
        }
    }

    // One row's totals: every term of every total added in the order the terms stand, from a total of positive nothing.
    private void Row(double[] totals, int row, bool lanes = true)
    {
        Array.Clear(totals);

        var across = Vector.IsHardwareAccelerated && lanes ? Vector<float>.Count : 0;
        var wide = across > 0 ? _shape.Columns - (_shape.Columns % across) : 0;

        for (var step = 0; step < _shape.Inner; step++)
        {
            var weight = (double)_left[_turnedLeft ? (step * _shape.Rows) + row : (row * _shape.Inner) + step];
            var terms = _right.Slice(step * _shape.Columns, _shape.Columns);
            var column = 0;

            if (wide > 0)
            {
                var factor = new Vector<double>(weight);
                var half = Vector<double>.Count;

                for (; column < wide; column += across)
                {
                    Vector.Widen(new Vector<float>(terms.Slice(column, across)), out var low, out var high);

                    (new Vector<double>(totals.AsSpan(column, half)) + (factor * low)).CopyTo(totals.AsSpan(column, half));
                    (new Vector<double>(totals.AsSpan(column + half, half)) + (factor * high)).CopyTo(totals.AsSpan(column + half, half));
                }
            }

            for (; column < _shape.Columns; column++)
            {
                totals[column] += weight * terms[column];
            }
        }
    }

    // Whether any total is no number: a vector equals itself in every lane but those.
    private static bool HoldsNoNumber(double[] totals)
    {
        var half = Vector<double>.Count;
        var column = 0;

        for (; column + half <= totals.Length; column += half)
        {
            var lanes = new Vector<double>(totals.AsSpan(column, half));

            if (!Vector.EqualsAll(lanes, lanes))
            {
                return true;
            }
        }

        for (; column < totals.Length; column++)
        {
            if (double.IsNaN(totals[column]))
            {
                return true;
            }
        }

        return false;
    }
}
