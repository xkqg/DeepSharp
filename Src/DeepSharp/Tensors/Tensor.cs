// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// A shape and the values that fill it.
/// </summary>
/// <remarks>
/// A tensor knows nothing about arithmetic. Everything a network does to one is done by an
/// <see cref="ITensorBackend"/>, which is what lets the same model run its heavy work somewhere else later
/// without a line of the model changing. The values are laid out row-major — the last axis moves fastest —
/// and a tensor never changes once it exists, so handing one to two layers is safe.
/// <para>
/// Its values live where the engine that made it keeps them, its <see cref="Storage"/>: this machine's memory for a tensor
/// made here, or an engine's own for one that engine made.
/// </para>
/// </remarks>
public sealed class Tensor
{
    // The values in this machine's memory, row-major: the array a tensor made here holds; for one standing on an engine's
    // storage, the one copy taken out of it — nothing, until they are read.
    private float[]? _values;

    // An engine's storage, handed in; for a tensor made here, the storage over its array, made the first time it is asked for.
    private TensorStorage? _storage;

    // What the one copy out of an engine's storage is taken under, so readers at the same moment wait for it rather than
    // each taking one of their own; made only when such a copy is taken.
    private object? _copying;

    private Tensor(Shape shape, float[] values)
    {
        Shape = shape;
        _values = values;
    }

    private Tensor(Shape shape, TensorStorage storage)
    {
        Shape = shape;
        _storage = storage;
        _values = (storage as ManagedStorage)?.Values;
    }

    /// <summary>The axes this tensor is laid out along.</summary>
    public Shape Shape { get; }

    /// <summary>Where the values live: this machine's memory for a tensor made here, or the storage of the engine that made it.</summary>
    public TensorStorage Storage => _storage ?? LazyInitializer.EnsureInitialized(ref _storage, () => new ManagedStorage(_values!));

    /// <summary>The values, row-major: the last axis moves fastest.</summary>
    /// <remarks>
    /// For a tensor standing on an engine's storage, the values are copied out of it into this machine's memory the first
    /// time they are read, and that copy is kept: never a view of memory the engine owns, which could be let go of while it
    /// was being read.
    /// </remarks>
    public ReadOnlySpan<float> Values => Volatile.Read(ref _values) ?? ReadBack();

    /// <summary>A tensor of the given shape with every value at zero.</summary>
    /// <param name="shape">The axes to lay the values out along.</param>
    public static Tensor Zeros(Shape shape) => new(shape, new float[shape.Count]);

    /// <summary>A tensor of the given shape holding a copy of the given values, row-major.</summary>
    /// <param name="shape">The axes to lay the values out along.</param>
    /// <param name="values">Exactly as many values as the shape holds.</param>
    /// <exception cref="ArgumentException">There are more or fewer values than the shape holds.</exception>
    public static Tensor From(Shape shape, ReadOnlySpan<float> values)
    {
        if (values.Length != shape.Count)
        {
            throw new ArgumentException(
                $"A {shape} tensor holds {shape.Count} values and {values.Length} were given.", nameof(values));
        }

        // Copied, not kept: a caller who reuses a scratch buffer would otherwise rewrite a tensor that has
        // already been handed to a layer, and that shows up as a wrong answer with no trace back to here.
        return new Tensor(shape, values.ToArray());
    }

    /// <summary>A tensor of the given shape standing on a storage, whose values stay where the storage keeps them.</summary>
    /// <param name="shape">The axes to lay the values out along.</param>
    /// <param name="storage">
    /// Exactly as many values as the shape holds, row-major: what an engine made, kept where it keeps what it makes. It must
    /// never change once a tensor stands on it.
    /// </param>
    /// <returns>The tensor; nothing is copied until its values are read.</returns>
    /// <exception cref="ArgumentNullException">No storage is given.</exception>
    /// <exception cref="ArgumentException">The storage holds more or fewer values than the shape does.</exception>
    /// <remarks>How an engine of its own hands back what it made, as <see cref="TensorStorage"/> describes.</remarks>
    public static Tensor On(Shape shape, TensorStorage storage)
    {
        ArgumentNullException.ThrowIfNull(storage);

        if (storage.Count != shape.Count)
        {
            throw new ArgumentException(
                $"A {shape} tensor holds {shape.Count} values and the storage given holds {storage.Count}.", nameof(storage));
        }

        return new Tensor(shape, storage);
    }

    /// <summary>Builds a tensor from values already laid out row-major, without copying them.</summary>
    /// <remarks>For a backend, or the reader of a network's file, that has just produced a buffer nobody else holds.</remarks>
    internal static Tensor Wrap(Shape shape, float[] values) => new(shape, values);

    /// <summary>The tensor as it is spoken: <c>Tensor 2x3</c>.</summary>
    public override string ToString() => $"Tensor {Shape}";

    // The values of a tensor standing on an engine's storage, copied out once, by whichever reader comes first, and kept.
    private float[] ReadBack() => LazyInitializer.EnsureInitialized(ref _values, ref _copying, () =>
    {
        var copy = new float[Shape.Count];
        _storage!.CopyTo(copy);

        return copy;
    });
}
