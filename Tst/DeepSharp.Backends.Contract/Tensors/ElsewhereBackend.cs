// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// Another engine than the one under test, to a checkpoint: it works every operation out on the engine it wraps, and names
/// itself, its version and its device otherwise — so a run gone on under it is refused for what the checkpoint recorded of
/// the engine it was taken on, and for nothing its arithmetic did.
/// </summary>
/// <param name="engine">The engine the arithmetic runs on.</param>
internal sealed class ElsewhereBackend(ITensorBackend engine) : NotingBackend(engine), INamesItsVersionAndDevice
{
    /// <summary>How a checkpoint names it.</summary>
    public const string Named = "'elsewhere' 1 on somewhere else";

    public override string Name => "elsewhere";

    public string Version => "1";

    public string Device => "somewhere else";

    /// <summary>
    /// How a checkpoint names an engine: its name, and — for one that names them — its version and the device it works on.
    /// </summary>
    public static string NamedAs(ITensorBackend engine) =>
        engine is INamesItsVersionAndDevice named ? $"'{engine.Name}' {named.Version} on {named.Device}" : $"'{engine.Name}'";
}
