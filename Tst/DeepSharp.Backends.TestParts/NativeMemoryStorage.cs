// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Parts;

/// <summary>
/// Values an engine outside the library keeps in memory allocated outside .NET, row-major: written once, as the engine
/// makes them, and never after.
/// </summary>
/// <remarks>
/// The storage owns that memory, and nothing but the collector ends it: once nothing holds the storage, its memory is let go
/// of. The collector cannot see memory allocated outside .NET, so the storage tells it how many bytes it holds, as a
/// storage over libtorch's memory has to — or a run could leave thousands of them waiting for a collection that a small
/// managed heap never asks for.
/// </remarks>
public sealed unsafe class NativeMemoryStorage : TensorStorage
{
    private readonly float* _values;
    private readonly int _count;
    private readonly long _bytes;
    private readonly StorageTally _tally;

    internal NativeMemoryStorage(int count, StorageTally tally)
    {
        // Room for one value at least, so a storage of none still owns memory of its own to let go of.
        var room = Math.Max(count, 1);

        _values = (float*)NativeMemory.AllocZeroed((nuint)room, sizeof(float));
        _count = count;
        _bytes = (long)room * sizeof(float);
        _tally = tally;
        GC.AddMemoryPressure(_bytes);
        tally.Held();
    }

    /// <summary>Lets the memory go once nothing holds the storage, and tells the collector and the engine so.</summary>
    ~NativeMemoryStorage()
    {
        NativeMemory.Free(_values);
        GC.RemoveMemoryPressure(_bytes);
        _tally.LetGo();
    }

    /// <inheritdoc />
    public override int Count => _count;

    /// <summary>The values where they are, for the engine that works on them; it keeps the storage alive while it reads.</summary>
    internal ReadOnlySpan<float> Values => new(_values, _count);

    /// <summary>The values where they are, for the engine to write once, as it makes them.</summary>
    internal Span<float> Written => new(_values, _count);

    /// <inheritdoc />
    public override void CopyTo(Span<float> destination)
    {
        Values.CopyTo(destination);

        // A span keeps nothing alive: the storage must outlive the copy out of it.
        GC.KeepAlive(this);
    }
}
