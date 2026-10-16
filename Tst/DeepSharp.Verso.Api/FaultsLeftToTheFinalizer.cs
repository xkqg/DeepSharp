// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tests.Api;

/// <summary>
/// The tests that leave a failing task to the finalizer on purpose. Verso's engine hands such a failure to the notebook that is
/// running a cell when it surfaces, as that cell's output — measured: dropped while another test's cell printed, it printed
/// into that cell — so these run on their own, once every test run in parallel is done.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class FaultsLeftToTheFinalizer
{
    /// <summary>The collection's name.</summary>
    public const string Name = "faults left to the finalizer";
}
