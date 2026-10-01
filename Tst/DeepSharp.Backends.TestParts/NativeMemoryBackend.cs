// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Parts;

/// <summary>
/// An engine outside the library that differs from the light one in the two ways libtorch does: what it makes lives in
/// memory allocated outside .NET, owned by a storage of its own, and it adds its totals up in single precision.
/// </summary>
/// <remarks>
/// It reaches only what DeepSharp publishes. Every operation asks the seam's refusals first; takes in a tensor on any other
/// storage once — the first time it is handed one, keeping what it took in for as long as that tensor lives, since a tensor
/// never changes; works its arithmetic out where its values live; and hands what it made back on a storage of its own. It
/// keeps nothing of what it made, so what stays alive is only what something else holds, and it counts how much that is.
/// <para>
/// Its totals are libtorch's: a product's inner sum is one fused multiply-add after another down the inner axis, as a
/// matrix kernel accumulates it; a column's sum goes down the rows; a mean adds its halves pairwise, as libtorch's cascade
/// does; what folds onto one place is added where it lands — every one of them in single precision. A value worked out
/// value by value is worked out one value at a time by .NET's single-precision functions, not the light engine's vector
/// route; a reshape stands on the storage it was handed, as a view of it.
/// </para>
/// </remarks>
public sealed class NativeMemoryBackend : INamesItsVersionAndDevice
{
    private readonly StorageTally _tally = new();
    private readonly ConditionalWeakTable<Tensor, NativeMemoryStorage> _takenIn = new();
    private readonly ConditionalWeakTable<Tensor, NativeMemoryStorage>.CreateValueCallback _takeIn;
    private int _made;
    private int _taken;

    /// <summary>An engine that has made nothing yet.</summary>
    public NativeMemoryBackend() => _takeIn = TakeIn;

    /// <inheritdoc />
    public string Name => "nativememory";

    /// <inheritdoc />
    /// <remarks>Its own assembly's, as an engine of somebody else's names the version of what works its arithmetic out.</remarks>
    public string Version { get; } = typeof(NativeMemoryBackend).Assembly.GetName().Version!.ToString(3);

    /// <inheritdoc />
    /// <remarks>This machine's processor, whose memory outside .NET it keeps its values in.</remarks>
    public string Device => "cpu";

    /// <summary>How many storages it has made for what its operations worked out.</summary>
    public int Made => Volatile.Read(ref _made);

    /// <summary>How many tensors on a storage other than its own it has taken into memory of its own, each once.</summary>
    public int TakenIn => Volatile.Read(ref _taken);

    /// <summary>How many of its storages, made or taken in, still hold their memory.</summary>
    public int Live => _tally.Live;

    /// <inheritdoc />
    public Tensor Add(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Add));

        return Both(left, right, static (a, b) => a + b);
    }

    /// <inheritdoc />
    public Tensor Subtract(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Subtract));

        return Both(left, right, static (a, b) => a - b);
    }

    /// <inheritdoc />
    public Tensor Multiply(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Multiply));

        return Both(left, right, static (a, b) => a * b);
    }

    /// <inheritdoc />
    public Tensor Divide(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Divide));

        return Both(left, right, static (a, b) => a / b);
    }

    /// <inheritdoc />
    public Tensor MatMul(Tensor left, Tensor right)
    {
        left.RequireMatrixProduct(right);

        var rows = left.Shape[0];
        var inner = left.Shape[1];
        var columns = right.Shape[1];
        var a = Held(left);
        var b = Held(right);
        var made = Fresh(rows * columns);
        var x = a.Values;
        var y = b.Values;
        var totals = made.Written;

        // Each total is one fused multiply-add after another down the inner axis, kept in single precision where it lands.
        for (var row = 0; row < rows; row++)
        {
            var into = totals.Slice(row * columns, columns);

            for (var step = 0; step < inner; step++)
            {
                var weight = x[(row * inner) + step];
                var across = y.Slice(step * columns, columns);

                for (var column = 0; column < columns; column++)
                {
                    into[column] = MathF.FusedMultiplyAdd(weight, across[column], into[column]);
                }
            }
        }

        GC.KeepAlive(a);
        GC.KeepAlive(b);

        return Tensor.On(new Shape(rows, columns), made);
    }

    /// <inheritdoc />
    public Tensor Transpose(Tensor matrix)
    {
        matrix.RequireMatrix(nameof(Transpose));

        var rows = matrix.Shape[0];
        var columns = matrix.Shape[1];
        var from = Held(matrix);
        var made = Fresh(rows * columns);
        var source = from.Values;
        var target = made.Written;

        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                target[(column * rows) + row] = source[(row * columns) + column];
            }
        }

        GC.KeepAlive(from);

        return Tensor.On(new Shape(columns, rows), made);
    }

    /// <inheritdoc />
    public Tensor AddRow(Tensor matrix, Tensor row)
    {
        matrix.RequireRow(row);

        var columns = matrix.Shape[1];
        var from = Held(matrix);
        var added = Held(row);
        var made = Fresh(matrix.Shape.Count);
        var source = from.Values;
        var shift = added.Values;
        var target = made.Written;

        for (var at = 0; at < target.Length; at++)
        {
            target[at] = source[at] + shift[at % columns];
        }

        GC.KeepAlive(from);
        GC.KeepAlive(added);

        return Tensor.On(matrix.Shape, made);
    }

    /// <inheritdoc />
    public Tensor SumRows(Tensor matrix)
    {
        matrix.RequireMatrix(nameof(SumRows));

        var rows = matrix.Shape[0];
        var columns = matrix.Shape[1];
        var from = Held(matrix);
        var made = Fresh(columns);
        var source = from.Values;
        var totals = made.Written;

        // Down the rows, each column's total in single precision.
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                totals[column] += source[(row * columns) + column];
            }
        }

        GC.KeepAlive(from);

        return Tensor.On(new Shape(columns), made);
    }

    /// <inheritdoc />
    public Tensor Mean(Tensor values)
    {
        values.RequireValues();

        var from = Held(values);
        var made = Fresh(1);

        made.Written[0] = PairwiseTotal(from.Values) / values.Shape.Count;
        GC.KeepAlive(from);

        return Tensor.On(new Shape(), made);
    }

    /// <inheritdoc />
    public Tensor Scale(Tensor values, Tensor factor)
    {
        values.RequireFactor(factor);

        var by = Held(factor);
        var scale = by.Values[0];
        GC.KeepAlive(by);

        return Each(values, value => value * scale);
    }

    /// <inheritdoc />
    public Tensor Fill(Shape shape, float value)
    {
        var made = Fresh(shape.Count);
        made.Written.Fill(value);

        return Tensor.On(shape, made);
    }

    /// <inheritdoc />
    public Tensor Relu(Tensor values) => Each(values, static value => MathF.Max(value, 0f));

    /// <inheritdoc />
    public Tensor Positive(Tensor values) => Each(values, static value => value > 0f ? 1f : 0f);

    /// <inheritdoc />
    public Tensor Tanh(Tensor values) => Each(values, MathF.Tanh);

    /// <inheritdoc />
    public Tensor Sigmoid(Tensor values) => Each(values, static value => 1f / (1f + MathF.Exp(-value)));

    /// <inheritdoc />
    public Tensor Exp(Tensor values) => Each(values, MathF.Exp);

    /// <inheritdoc />
    public Tensor Log(Tensor values) => Each(values, MathF.Log);

    /// <inheritdoc />
    public Tensor Sqrt(Tensor values) => Each(values, MathF.Sqrt);

    /// <inheritdoc />
    /// <remarks>As libtorch has it in single precision: the value itself above twenty, the logarithm of one plus its exponential below.</remarks>
    public Tensor Softplus(Tensor values) => Each(values, static value => value > 20f ? value : LogOfOnePlus(MathF.Exp(value)));

    /// <inheritdoc />
    /// <remarks>As libtorch has it: each row less its largest value, less the logarithm of its exponentials' single-precision sum.</remarks>
    public Tensor LogSoftmax(Tensor matrix)
    {
        matrix.RequireMatrix(nameof(LogSoftmax));

        var columns = matrix.Shape[1];
        var from = Held(matrix);
        var made = Fresh(matrix.Shape.Count);
        var source = from.Values;
        var target = made.Written;

        for (var start = 0; start < source.Length; start += columns)
        {
            var row = source.Slice(start, columns);
            var largest = float.NegativeInfinity;

            foreach (var value in row)
            {
                largest = MathF.Max(largest, value);
            }

            var total = 0f;

            foreach (var value in row)
            {
                total += MathF.Exp(value - largest);
            }

            var logTotal = MathF.Log(total);

            for (var column = 0; column < columns; column++)
            {
                target[start + column] = row[column] - largest - logTotal;
            }
        }

        GC.KeepAlive(from);

        return Tensor.On(matrix.Shape, made);
    }

    /// <inheritdoc />
    /// <remarks>Stands on the storage it was handed, as a view of it: the values are the same values, in their order.</remarks>
    public Tensor Reshape(Tensor values, Shape shape)
    {
        values.RequireSameCount(shape);

        return Tensor.On(shape, Held(values));
    }

    /// <inheritdoc />
    public Tensor Unfold(Tensor images, Window window)
    {
        images.RequireImagesFor(window);

        var channels = images.Shape[3];
        var from = Held(images);
        var patches = new Shape(
            images.Shape[0] * window.RowsOver(images.Shape[1]) * window.ColumnsOver(images.Shape[2]), window.Height * window.Width * channels);
        var made = Fresh(patches.Count);
        var source = from.Values;
        var target = made.Written;

        foreach (var cell in Cells(images.Shape, window))
        {
            source.Slice(cell.Image, channels).CopyTo(target.Slice(cell.Patch, channels));
        }

        GC.KeepAlive(from);

        return Tensor.On(patches, made);
    }

    /// <inheritdoc />
    public Tensor Fold(Tensor patches, Shape images, Window window)
    {
        patches.RequirePatchesOf(images, window);

        var channels = images[3];
        var from = Held(patches);
        var made = Fresh(images.Count);
        var source = from.Values;
        var totals = made.Written;

        // What lands on one place is added where it lands, in single precision, in the order the window walks.
        foreach (var cell in Cells(images, window))
        {
            for (var channel = 0; channel < channels; channel++)
            {
                totals[cell.Image + channel] += source[cell.Patch + channel];
            }
        }

        GC.KeepAlive(from);

        return Tensor.On(images, made);
    }

    // log(1 + y) in single precision without losing a small y: where one plus it is one, it is y itself; elsewhere the
    // logarithm of what one plus it rounded to, scaled by how far that rounding moved it.
    private static float LogOfOnePlus(float y)
    {
        var sum = 1f + y;

        return sum == 1f ? y : MathF.Log(sum) * (y / (sum - 1f));
    }

    // A single-precision total of the values, the two halves of each run added up apart and then together.
    private static float PairwiseTotal(ReadOnlySpan<float> values)
    {
        if (values.Length <= 8)
        {
            var total = 0f;

            foreach (var value in values)
            {
                total += value;
            }

            return total;
        }

        var half = values.Length / 2;

        return PairwiseTotal(values[..half]) + PairwiseTotal(values[half..]);
    }

    // Every cell of every place a window stands on a batch of images, channels last, that covers the image rather than its
    // border: where its channels start in the patches, and where they start in the images. The first place stands where the
    // window's border, as the library's one rule for it works it out, puts it.
    private static IEnumerable<CoveredCell> Cells(Shape images, Window window)
    {
        var height = images[1];
        var width = images[2];
        var channels = images[3];
        var borders = window.BordersOver(height, width);
        var patch = 0;

        for (var image = 0; image < images[0]; image++)
        {
            for (var row = 0; row < window.RowsOver(height); row++)
            {
                for (var column = 0; column < window.ColumnsOver(width); column++)
                {
                    for (var down = 0; down < window.Height; down++)
                    {
                        for (var across = 0; across < window.Width; across++)
                        {
                            var y = (row * window.Stride) - borders.Top + down;
                            var x = (column * window.Stride) - borders.Left + across;

                            if (y >= 0 && y < height && x >= 0 && x < width)
                            {
                                yield return new CoveredCell(patch, (((image * height) + y) * width + x) * channels);
                            }

                            patch += channels;
                        }
                    }
                }
            }
        }
    }

    // A tensor's values where this engine works on them: its own storage as it is, any other taken in once.
    private NativeMemoryStorage Held(Tensor tensor) => tensor.Storage as NativeMemoryStorage ?? _takenIn.GetValue(tensor, _takeIn);

    // A tensor on another storage, copied into memory of this engine's own: kept for as long as the tensor lives.
    private NativeMemoryStorage TakeIn(Tensor tensor)
    {
        var storage = new NativeMemoryStorage(tensor.Shape.Count, _tally);
        tensor.Values.CopyTo(storage.Written);
        Interlocked.Increment(ref _taken);

        return storage;
    }

    // A storage for what an operation works out, every value nought until it is written.
    private NativeMemoryStorage Fresh(int count)
    {
        Interlocked.Increment(ref _made);

        return new NativeMemoryStorage(count, _tally);
    }

    private Tensor Each(Tensor values, Func<float, float> operation)
    {
        ArgumentNullException.ThrowIfNull(values);

        var from = Held(values);
        var made = Fresh(values.Shape.Count);
        var source = from.Values;
        var target = made.Written;

        for (var at = 0; at < target.Length; at++)
        {
            target[at] = operation(source[at]);
        }

        GC.KeepAlive(from);

        return Tensor.On(values.Shape, made);
    }

    private Tensor Both(Tensor left, Tensor right, Func<float, float, float> operation)
    {
        var a = Held(left);
        var b = Held(right);
        var made = Fresh(left.Shape.Count);
        var x = a.Values;
        var y = b.Values;
        var target = made.Written;

        for (var at = 0; at < target.Length; at++)
        {
            target[at] = operation(x[at], y[at]);
        }

        GC.KeepAlive(a);
        GC.KeepAlive(b);

        return Tensor.On(left.Shape, made);
    }

    // Where one cell's channels start in the patches, and where the same values start in the images.
    private readonly record struct CoveredCell(int Patch, int Image);
}
