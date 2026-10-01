// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tests.Backends.Parts;

/// <summary>
/// How many of an engine's storages hold memory now: counted up as each is made, and down as each lets its memory go —
/// which happens on the collector's own thread, so both are counted atomically.
/// </summary>
internal sealed class StorageTally
{
    private int _live;

    /// <summary>How many hold memory now.</summary>
    public int Live => Volatile.Read(ref _live);

    /// <summary>A storage has taken memory.</summary>
    public void Held() => Interlocked.Increment(ref _live);

    /// <summary>A storage has let its memory go.</summary>
    public void LetGo() => Interlocked.Decrement(ref _live);
}
