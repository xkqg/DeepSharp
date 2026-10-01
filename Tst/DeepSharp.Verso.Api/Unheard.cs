// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;

namespace DeepSharp.Tests.Api;

/// <summary>A notebook a run made by a test is of, which hears nothing the run tells it, and whose stop ends at once.</summary>
internal sealed class Unheard : IRunListener
{
    /// <summary>The one such notebook.</summary>
    public static Unheard Notebook { get; } = new();

    public void Moved()
    {
    }

    public Task StoppedAsync() => Task.CompletedTask;
}
