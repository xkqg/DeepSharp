// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using DeepSharp.Tensors;
using TorchSharp;

namespace DeepSharp.Backends.TorchSharp;

/// <summary>
/// Values libtorch keeps where the engine that made them runs — this machine's memory or a graphics card's — row-major,
/// written once, as libtorch works them out, and never after.
/// </summary>
/// <remarks>
/// The storage owns libtorch's tensor from the moment it exists. TorchSharp hands every tensor it makes to whatever dispose
/// scope is open on the thread that made it, and ends it with that scope; a storage takes its tensor out of any such scope
/// at once, so a scope of the caller's ends without touching what the engine made — the network's slots, what its optimizer
/// remembers, the best epoch and every checkpoint a caller keeps are all such tensors, and each is held by reference, as a
/// tensor never changes. Nothing but the collector ends it: once nothing holds the storage, its tensor is disposed. The
/// collector cannot see libtorch's memory, so the storage tells it how many bytes it holds — or a run could leave thousands
/// of them waiting for a collection that a small managed heap never asks for.
/// </remarks>
internal sealed class TorchStorage : TensorStorage
{
    private readonly torch.Tensor _native;
    private readonly string _device;
    private readonly Shape _laidOut;
    private readonly long _bytes;

    /// <summary>Takes a tensor libtorch made into a storage that owns it.</summary>
    /// <param name="native">The tensor, laid out row-major and never written again.</param>
    /// <param name="device">Where it lives, as libtorch names a device — <c>cpu</c>, <c>cuda:0</c>: the engine's that made it.</param>
    internal TorchStorage(torch.Tensor native, string device)
    {
        _native = native.DetachFromDisposeScope();
        _device = device;
        _laidOut = new Shape([.. native.shape.Select(length => checked((int)length))]);

        // At least one value's worth: a storage of none still holds a tensor of libtorch's.
        _bytes = Math.Max(_laidOut.Count, 1) * (long)sizeof(float);
        GC.AddMemoryPressure(_bytes);
    }

    /// <summary>Lets libtorch's tensor go once nothing holds the storage, and tells the collector so.</summary>
    ~TorchStorage()
    {
        _native.Dispose();
        GC.RemoveMemoryPressure(_bytes);
    }

    /// <inheritdoc />
    public override int Count => _laidOut.Count;

    /// <summary>Whether the values live on the device of that name.</summary>
    /// <param name="device">The device, as libtorch names it: <c>cpu</c>, <c>cuda:0</c>.</param>
    /// <returns>Whether it is this storage's device.</returns>
    internal bool IsOn(string device) => string.Equals(device, _device, StringComparison.Ordinal);

    /// <summary>
    /// The values laid out along the given axes, for an operation that reads them so: the tensor itself when it is laid out
    /// so already — as everything libtorch works out is — and otherwise, for a tensor another shape was put on without a
    /// copy, a view of it that belongs to the dispose scope the operation reads it in.
    /// </summary>
    /// <param name="shape">Axes that hold as many values as the storage does.</param>
    /// <returns>The tensor, or a view of it.</returns>
    internal torch.Tensor As(Shape shape) => shape == _laidOut ? _native : _native.reshape(shape.Lengths());

    /// <inheritdoc />
    /// <remarks>A graphics card's values are brought into this machine's memory first, and read from there.</remarks>
    public override void CopyTo(Span<float> destination)
    {
        using (torch.NewDisposeScope())
        {
            MemoryMarshal.Cast<byte, float>(_native.cpu().bytes).CopyTo(destination);
        }

        // A span keeps nothing alive: the storage, and with it libtorch's tensor, must outlive the copy out of it.
        GC.KeepAlive(this);
    }
}
