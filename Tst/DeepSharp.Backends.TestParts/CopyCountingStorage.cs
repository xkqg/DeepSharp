// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Parts;

/// <summary>
/// Values an engine outside the library keeps in a place of its own, handed out whole whenever they are asked for, and
/// every time counted — so a test sees how often the library copies them out.
/// </summary>
public sealed class CopyCountingStorage : TensorStorage
{
    private readonly float[] _values;
    private int _copies;

    /// <summary>Keeps its own copy of the given values.</summary>
    /// <param name="values">The values, row-major.</param>
    public CopyCountingStorage(ReadOnlySpan<float> values) => _values = values.ToArray();

    /// <summary>How many times its values have been copied out.</summary>
    public int Copies => Volatile.Read(ref _copies);

    /// <inheritdoc />
    public override int Count => _values.Length;

    /// <inheritdoc />
    public override void CopyTo(Span<float> destination)
    {
        Interlocked.Increment(ref _copies);
        _values.CopyTo(destination);
    }
}
