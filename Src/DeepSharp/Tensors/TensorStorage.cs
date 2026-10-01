// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// Where a tensor's values live: in this machine's memory, as a tensor made here keeps them, or wherever the engine that
/// made the tensor keeps what it makes.
/// </summary>
/// <remarks>
/// An engine that keeps its values somewhere of its own — memory it allocated outside .NET, a graphics card — derives from
/// this, and hands back what it makes as <see cref="Tensor.On(Shape, TensorStorage)"/>, a tensor standing on that
/// storage. When a tensor comes back to it, it knows its own storage by its type from <see cref="Tensor.Storage"/>, and
/// takes any other in where the operation asks for it — once for each tensor, if it keeps what it took in. Nothing on
/// <see cref="ITensorBackend"/> changes for it, so a model cannot tell which engine made its tensors.
/// <para>
/// A storage never changes once a tensor stands on it, as the tensor never does: what it copies out the first time is
/// what it would copy out every time, which is why <see cref="Tensor.Values"/> copies it out once and keeps the copy.
/// </para>
/// </remarks>
public abstract class TensorStorage
{
    /// <summary>How many values it holds.</summary>
    public abstract int Count { get; }

    /// <summary>Copies every value it holds into memory the caller owns, row-major: the last axis moves fastest.</summary>
    /// <param name="destination">Exactly <see cref="Count"/> values long.</param>
    public abstract void CopyTo(Span<float> destination);
}

/// <summary>Values in this machine's memory, as a tensor made here keeps them: an array nothing else writes to.</summary>
/// <param name="values">The values, row-major.</param>
internal sealed class ManagedStorage(float[] values) : TensorStorage
{
    /// <summary>The values themselves, which a tensor reads where they are.</summary>
    internal float[] Values { get; } = values;

    /// <inheritdoc />
    public override int Count => Values.Length;

    /// <inheritdoc />
    public override void CopyTo(Span<float> destination) => Values.CopyTo(destination);
}
