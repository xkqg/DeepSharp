// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// An engine that names, beside itself, the version of what works its arithmetic out and the device it works it out on:
/// what a checkpoint records of the engine its run was on, so that a run going on from it on any other is refused.
/// </summary>
/// <remarks>
/// Two engines that add their totals up in another order drift apart over many steps, as any two float engines do, so a
/// run goes on from its checkpoint as the same run only on the engine it was taken on: the same engine, in the same
/// version, on the same device. The light engine names DeepSharp's version, whose code works its arithmetic out, and the
/// processor; the libtorch engine names the libtorch it runs on and the device it was made for. An engine of your own that
/// names neither is recorded by its <see cref="ITensorBackend.Name"/> alone, and held to that.
/// </remarks>
public interface INamesItsVersionAndDevice : ITensorBackend
{
    /// <summary>The version of what works the arithmetic out, as its makers number it.</summary>
    string Version { get; }

    /// <summary>Where the arithmetic is worked out, as the library that works it out names the place: <c>cpu</c>, <c>cuda:0</c>.</summary>
    string Device { get; }
}
