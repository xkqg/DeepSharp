// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tests.Serve;

/// <summary>
/// The tests that let a run left behind end. Such a run puts the process's console back as it found it when it began,
/// under whatever C# run is under way then — Verso's C# kernel does so at the end of every run — so these run on their
/// own, once every test run in parallel is done.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class RunsLeftBehind
{
    /// <summary>The collection's name.</summary>
    public const string Name = "runs left behind";
}
