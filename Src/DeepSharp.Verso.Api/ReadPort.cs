// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso;
using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

/// <summary>
/// The notebook as a look hands it — a button asked whether it can be pressed, a panel asked to draw its section: a look
/// does nothing to the notebook, so every verb is refused, whoever wrote the part, and only which layout and theme the
/// notebook is shown in is answered.
/// </summary>
/// <param name="scaffold">The notebook.</param>
internal sealed class ReadPort(Scaffold scaffold) : NotebookPort(scaffold)
{
    public override Task ExecuteCellAsync(Guid cellId) => throw Refused();

    public override Task ExecuteAllAsync() => throw Refused();

    public override Task ExecuteFromAsync(Guid cellId) => throw Refused();

    public override Task ExecuteCodeAsync(string code, string? language = null, CancellationToken ct = default) => throw Refused();

    public override Task<IReadOnlyList<CellOutput>> ExecuteCodeCaptureOutputsAsync(string code, string? language = null, CancellationToken ct = default) =>
        throw Refused();

    protected override void Admit() => throw Refused();

    private static InvalidOperationException Refused() =>
        new("A part asked only to look at the notebook, to say whether it can be pressed or to draw its section, does nothing to it.");
}
