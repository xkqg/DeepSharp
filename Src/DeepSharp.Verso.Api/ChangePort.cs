// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

/// <summary>
/// The notebook as a change hands it — a click on a control a cell drew, a field of a cell's properties changed, the
/// notebook told its cells changed. A change acts on the notebook as the notebook's own operations let it, and runs
/// DeepSharp's blocks as its own; anything else it asks to run — another cell, all of them, all from one on, or code in
/// no cell — is a run of its own, which every view is told from its ask, which takes the C# turn when it runs C#, and
/// which a stop ends as it ends any run. Once such a run is stopped, nothing else the change asks is done.
/// </summary>
/// <param name="host">The notebook's host, which builds and runs every run.</param>
/// <remarks>
/// The change's runs go one at a time, in the order it asked for them, and the change is not over until every run it asked
/// for has ended, or was stopped, whether or not it waited for them.
/// </remarks>
internal sealed class ChangePort(NotebookHost host) : NotebookPort(host.Scaffold)
{
    // The change's runs, one after another.
    private readonly Lane _runs = new();

    // The change's last run: its token is the port's, and once it was stopped the port refuses.
    private Run? _run;

    public override CancellationToken Token => Volatile.Read(ref _run)?.Token ?? CancellationToken.None;

    public override Task ExecuteCellAsync(Guid cellId)
    {
        Admit();

        return Scaffold.GetCell(cellId) is { Type: StepCellType.StepType }
            ? Engine.ExecuteCellAsync(cellId)
            : RunAsync(cellId, host.RunsCSharp(cellId), run => run.ExecuteCellAsync(cellId));
    }

    public override Task ExecuteAllAsync()
    {
        Admit();

        return RunAsync(null, runsCSharp: true, run => run.ExecuteAllAsync());
    }

    public override Task ExecuteFromAsync(Guid cellId)
    {
        Admit();

        return RunAsync(null, runsCSharp: true, run => run.ExecuteFromAsync(cellId));
    }

    public override Task ExecuteCodeAsync(string code, string? language = null, CancellationToken ct = default)
    {
        Admit();

        return RunAsync(null, host.RunsCSharp(language), run => run.ExecuteCodeAsync(code, language, ct));
    }

    public override async Task<IReadOnlyList<CellOutput>> ExecuteCodeCaptureOutputsAsync(string code, string? language = null, CancellationToken ct = default)
    {
        Admit();

        IReadOnlyList<CellOutput> outputs = [];

        await RunAsync(null, host.RunsCSharp(language), async run => outputs = await run.ExecuteCodeCaptureOutputsAsync(code, language, ct));

        return outputs;
    }

    /// <summary>Waits until every run the change asked for has ended, or was stopped.</summary>
    /// <returns>A task that ends then.</returns>
    public Task SettledAsync() => _runs.TakeTurnAsync(() => Task.FromResult(true));

    // Once the change's run was stopped, nothing else it asks is done.
    protected override void Admit() => Token.ThrowIfCancellationRequested();

    // A run of the change's own, after every run it asked for before; the part is told when it was stopped.
    private Task RunAsync(Guid? cell, bool runsCSharp, Func<RunPort, Task> work) => _runs.TakeTurnAsync(async () =>
    {
        Admit();

        var run = host.RunFor(cell, runsCSharp);

        Volatile.Write(ref _run, run);
        await host.RunForChangeAsync(run, work);
        run.Token.ThrowIfCancellationRequested();

        return true;
    });
}
