// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Numerics.Tensors;

namespace DeepSharp.Tensors;

/// <summary>
/// The backend that ships: this machine's own vector registers, through .NET's tensor primitives.
/// </summary>
/// <remarks>
/// There is no native library behind this and nothing to install. That is the whole point — a model built
/// here runs inside an ordinary .NET application, on any operating system the runtime reaches, without a
/// hundred megabytes of platform-specific binaries travelling with it.
/// <para>
/// The tensor primitives hold no matrix product, so that one is written here. Every total — a product's inner sum, a
/// column's sum, a mean — is kept in double precision while it is added up: a single-precision total stops taking small
/// values in once it is large, and ten million tenths came to a mean of 0.1087937 rather than a tenth.
/// </para>
/// </remarks>
public sealed class CpuBackend : ITensorBackend
{
    /// <inheritdoc />
    public string Name => "cpu";

    /// <inheritdoc />
    public Tensor Add(Tensor left, Tensor right) => Elementwise(left, right, TensorPrimitives.Add, nameof(Add));

    /// <inheritdoc />
    public Tensor Subtract(Tensor left, Tensor right) =>
        Elementwise(left, right, TensorPrimitives.Subtract, nameof(Subtract));

    /// <inheritdoc />
    public Tensor Multiply(Tensor left, Tensor right) =>
        Elementwise(left, right, TensorPrimitives.Multiply, nameof(Multiply));

    /// <inheritdoc />
    public Tensor MatMul(Tensor left, Tensor right)
    {
        RequireMatrix(left, nameof(MatMul), nameof(left));
        RequireMatrix(right, nameof(MatMul), nameof(right));

        var rows = left.Shape[0];
        var inner = left.Shape[1];
        var columns = right.Shape[1];

        if (right.Shape[0] != inner)
        {
            throw new ArgumentException(
                $"MatMul needs the left matrix as wide as the right one is tall, and was given {left.Shape} and {right.Shape}.",
                nameof(right));
        }

        var a = left.Values;
        var b = right.Values;
        var result = new float[rows * columns];
        var totals = new double[columns];

        for (var row = 0; row < rows; row++)
        {
            Array.Clear(totals);

            // Row by row of the right, so both are read in the order they are laid out.
            for (var step = 0; step < inner; step++)
            {
                var weight = (double)a[(row * inner) + step];
                var across = b.Slice(step * columns, columns);

                for (var column = 0; column < columns; column++)
                {
                    totals[column] += weight * across[column];
                }
            }

            for (var column = 0; column < columns; column++)
            {
                result[(row * columns) + column] = (float)totals[column];
            }
        }

        return Tensor.Wrap(new Shape(rows, columns), result);
    }

    /// <inheritdoc />
    public Tensor Transpose(Tensor matrix)
    {
        RequireMatrix(matrix, nameof(Transpose), nameof(matrix));

        var rows = matrix.Shape[0];
        var columns = matrix.Shape[1];
        var values = matrix.Values;
        var result = new float[values.Length];

        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                result[(column * rows) + row] = values[(row * columns) + column];
            }
        }

        return Tensor.Wrap(new Shape(columns, rows), result);
    }

    /// <inheritdoc />
    public Tensor AddRow(Tensor matrix, Tensor row)
    {
        RequireMatrix(matrix, nameof(AddRow), nameof(matrix));
        ArgumentNullException.ThrowIfNull(row);

        if (row.Shape.Rank != 1)
        {
            throw new ArgumentException($"AddRow needs a row, a tensor of one axis, and was given a {row.Shape} one.", nameof(row));
        }

        var rows = matrix.Shape[0];
        var columns = matrix.Shape[1];

        if (row.Shape[0] != columns)
        {
            throw new ArgumentException(
                $"AddRow needs a row as long as the matrix is wide, and was given {matrix.Shape} and {row.Shape}.", nameof(row));
        }

        var result = new float[matrix.Values.Length];

        for (var at = 0; at < rows; at++)
        {
            TensorPrimitives.Add(matrix.Values.Slice(at * columns, columns), row.Values, result.AsSpan(at * columns, columns));
        }

        return Tensor.Wrap(matrix.Shape, result);
    }

    /// <inheritdoc />
    public Tensor SumRows(Tensor matrix)
    {
        RequireMatrix(matrix, nameof(SumRows), nameof(matrix));

        var rows = matrix.Shape[0];
        var columns = matrix.Shape[1];
        var values = matrix.Values;
        var totals = new double[columns];

        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                totals[column] += values[(row * columns) + column];
            }
        }

        return Tensor.Wrap(new Shape(columns), [.. totals.Select(total => (float)total)]);
    }

    /// <inheritdoc />
    public Tensor Mean(Tensor values)
    {
        ArgumentNullException.ThrowIfNull(values);

        if (values.Shape.Count == 0)
        {
            throw new ArgumentException($"A mean needs at least one value, and a {values.Shape} tensor holds none.", nameof(values));
        }

        var total = 0d;

        foreach (var value in values.Values)
        {
            total += value;
        }

        return Tensor.Wrap(new Shape(), [(float)(total / values.Shape.Count)]);
    }

    /// <inheritdoc />
    public Tensor Scale(Tensor values, Tensor factor)
    {
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(factor);

        if (factor.Shape.Rank != 0)
        {
            throw new ArgumentException(
                $"Scale multiplies by one value, a tensor with no axes, and was given a {factor.Shape} one.", nameof(factor));
        }

        var result = new float[values.Values.Length];
        TensorPrimitives.Multiply(values.Values, factor.Values[0], result);

        return Tensor.Wrap(values.Shape, result);
    }

    /// <inheritdoc />
    public Tensor Fill(Shape shape, float value)
    {
        var result = new float[shape.Count];
        Array.Fill(result, value);

        return Tensor.Wrap(shape, result);
    }

    /// <inheritdoc />
    public Tensor Relu(Tensor values) => Elementwise(values, (source, destination) => TensorPrimitives.Max(source, 0f, destination));

    /// <inheritdoc />
    public Tensor Positive(Tensor values) => Elementwise(values, (source, destination) =>
    {
        for (var at = 0; at < source.Length; at++)
        {
            destination[at] = source[at] > 0f ? 1f : 0f;
        }
    });

    /// <inheritdoc />
    /// <remarks>The processor's vector instructions take their own route to a tangent: tanh of nothing is six hundred-millionths below it.</remarks>
    public Tensor Tanh(Tensor values) => Elementwise(values, TensorPrimitives.Tanh);

    /// <inheritdoc />
    public Tensor Sigmoid(Tensor values) => Elementwise(values, TensorPrimitives.Sigmoid);

    /// <inheritdoc />
    public Tensor Exp(Tensor values) => Elementwise(values, TensorPrimitives.Exp);

    /// <inheritdoc />
    public Tensor Log(Tensor values) => Elementwise(values, TensorPrimitives.Log);

    /// <inheritdoc />
    public Tensor Sqrt(Tensor values) => Elementwise(values, TensorPrimitives.Sqrt);

    /// <inheritdoc />
    /// <remarks>
    /// Worked out in double precision as the largest of the value and nothing, plus the logarithm of one plus the
    /// exponential of minus its size — which never overflows, since that exponential is at most one.
    /// </remarks>
    public Tensor Softplus(Tensor values) => Elementwise(values, (source, destination) =>
    {
        for (var at = 0; at < source.Length; at++)
        {
            double value = source[at];

            destination[at] = (float)(Math.Max(value, 0) + LogOfOnePlus(Math.Exp(-Math.Abs(value))));
        }
    });

    /// <inheritdoc />
    public Tensor Divide(Tensor left, Tensor right) => Elementwise(left, right, TensorPrimitives.Divide, nameof(Divide));

    /// <inheritdoc />
    /// <remarks>Each row's sum of exponentials is kept in double precision while it is added up, like every total here.</remarks>
    public Tensor LogSoftmax(Tensor matrix)
    {
        RequireMatrix(matrix, nameof(LogSoftmax), nameof(matrix));

        var columns = matrix.Shape[1];
        var values = matrix.Values;
        var result = new float[values.Length];

        for (var start = 0; start < values.Length; start += columns)
        {
            var row = values.Slice(start, columns);
            var largest = double.NegativeInfinity;

            foreach (var value in row)
            {
                largest = Math.Max(largest, value);
            }

            var total = 0d;

            foreach (var value in row)
            {
                total += Math.Exp(value - largest);
            }

            var shift = largest + Math.Log(total);

            for (var column = 0; column < columns; column++)
            {
                result[start + column] = (float)(row[column] - shift);
            }
        }

        return Tensor.Wrap(matrix.Shape, result);
    }

    /// <inheritdoc />
    public Tensor Reshape(Tensor values, Shape shape)
    {
        ArgumentNullException.ThrowIfNull(values);

        if (shape.Count != values.Shape.Count)
        {
            throw new ArgumentException(
                $"Reshape keeps every value, and a {values.Shape} tensor holds {values.Shape.Count} where a {shape} one holds {shape.Count}.",
                nameof(shape));
        }

        return Tensor.Wrap(shape, values.Values.ToArray());
    }

    /// <inheritdoc />
    public Tensor Unfold(Tensor images, Window window)
    {
        ArgumentNullException.ThrowIfNull(images);
        RequireImages(images.Shape, window, nameof(Unfold), nameof(images));

        var walk = new ImageWalk(images.Shape, window);
        var source = images.Values;
        var result = new float[walk.Patches.Count];

        foreach (var cell in walk.Cells())
        {
            source.Slice(cell.Image, walk.Channels).CopyTo(result.AsSpan(cell.Patch, walk.Channels));
        }

        return Tensor.Wrap(walk.Patches, result);
    }

    /// <inheritdoc />
    /// <remarks>What lands on one place is added up in double precision, like every total here.</remarks>
    public Tensor Fold(Tensor patches, Shape images, Window window)
    {
        ArgumentNullException.ThrowIfNull(patches);
        RequireImages(images, window, nameof(Fold), nameof(images));

        var walk = new ImageWalk(images, window);

        if (patches.Shape != walk.Patches)
        {
            throw new ArgumentException(
                $"Fold puts back the {walk.Patches} patches a {window} takes of {images} images, and was given {patches.Shape}.",
                nameof(patches));
        }

        var source = patches.Values;
        var totals = new double[images.Count];

        foreach (var cell in walk.Cells())
        {
            for (var channel = 0; channel < walk.Channels; channel++)
            {
                totals[cell.Image + channel] += source[cell.Patch + channel];
            }
        }

        return Tensor.Wrap(images, [.. totals.Select(total => (float)total)]);
    }

    // log(1 + x) for x between nothing and one. Taken as it is written, one plus a number below a ten-thousand-billionth
    // is one and the number is lost — .NET's own LogP1 is written that way — so below a ten-thousandth the first three
    // terms of its series stand in, which are exact to the last digit a double carries.
    private static double LogOfOnePlus(double x) => x < 1e-4 ? x * (1 - (x * (0.5 - (x / 3)))) : Math.Log(1 + x);

    /// <summary>The value-by-value operations of one tensor share the buffer they write into.</summary>
    private static Tensor Elementwise(Tensor values, UnaryOperation operation)
    {
        ArgumentNullException.ThrowIfNull(values);

        var result = new float[values.Shape.Count];
        operation(values.Values, result);

        return Tensor.Wrap(values.Shape, result);
    }

    /// <summary>Refuses a shape that is not a batch of images, or a window that cannot stand on them.</summary>
    private static void RequireImages(Shape images, Window window, string operation, string parameter)
    {
        if (images.Rank != 4)
        {
            throw new ArgumentException(
                $"{operation} works on a batch of images, image by row by column by channel, and was given a {images} one.", parameter);
        }

        if (window.Height < 1 || window.Width < 1 || window.Stride < 1 || window.Padding < 0)
        {
            throw new ArgumentException(
                $"A {window} cannot stand anywhere: its sides and its stride are at least one, and its border at least nothing.", nameof(window));
        }

        if (window.RowsOver(images[1]) < 1 || window.ColumnsOver(images[2]) < 1)
        {
            throw new ArgumentException($"A {window} is larger than a {images} image with its border.", nameof(window));
        }
    }

    /// <summary>The shape check the value-by-value operations share, and the buffer they write into.</summary>
    private static Tensor Elementwise(
        Tensor left,
        Tensor right,
        ElementwiseOperation operation,
        string name)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.Shape != right.Shape)
        {
            throw new ArgumentException(
                $"{name} needs two tensors of the same shape, and was given {left.Shape} and {right.Shape}.",
                nameof(right));
        }

        var result = new float[left.Shape.Count];
        operation(left.Values, right.Values, result);
        return Tensor.Wrap(left.Shape, result);
    }

    /// <summary>Refuses a tensor that is not a matrix, in the words of the operation that needs one.</summary>
    private static void RequireMatrix(Tensor tensor, string operation, string parameter)
    {
        ArgumentNullException.ThrowIfNull(tensor, parameter);

        if (tensor.Shape.Rank != 2)
        {
            throw new ArgumentException($"{operation} works on matrices, and was given a {tensor.Shape} tensor.", parameter);
        }
    }

    /// <summary>One of the tensor primitives that reads two spans and writes a third.</summary>
    private delegate void ElementwiseOperation(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> destination);

    /// <summary>One of the tensor primitives that reads one span and writes another.</summary>
    private delegate void UnaryOperation(ReadOnlySpan<float> source, Span<float> destination);

    /// <summary>
    /// A window's walk over a batch of images, channels last: every place it stands, and every place of the image each of
    /// its cells covers there — which is what unfolding copies out and folding adds back.
    /// </summary>
    private sealed class ImageWalk(Shape images, Window window)
    {
        private readonly int _count = images[0];
        private readonly int _height = images[1];
        private readonly int _width = images[2];
        private readonly int _rows = window.RowsOver(images[1]);
        private readonly int _columns = window.ColumnsOver(images[2]);

        /// <summary>How many values each place of an image holds.</summary>
        public int Channels { get; } = images[3];

        /// <summary>The patches' shape: a row for every image and every place the window stands, each as long as the window holds.</summary>
        public Shape Patches => new(_count * _rows * _columns, window.Height * window.Width * Channels);

        /// <summary>
        /// Every cell of every patch that covers the image rather than its border: where its channels start in the patches,
        /// and where they start in the images.
        /// </summary>
        public IEnumerable<CoveredCell> Cells()
        {
            var patch = 0;

            for (var image = 0; image < _count; image++)
            {
                for (var row = 0; row < _rows; row++)
                {
                    for (var column = 0; column < _columns; column++)
                    {
                        for (var down = 0; down < window.Height; down++)
                        {
                            for (var across = 0; across < window.Width; across++)
                            {
                                var y = (row * window.Stride) - window.Padding + down;
                                var x = (column * window.Stride) - window.Padding + across;

                                if (y >= 0 && y < _height && x >= 0 && x < _width)
                                {
                                    yield return new CoveredCell(patch, ((((image * _height) + y) * _width) + x) * Channels);
                                }

                                patch += Channels;
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>Where a cell's channels start in the patches, and where the same values start in the images.</summary>
    private readonly record struct CoveredCell(int Patch, int Image);
}
