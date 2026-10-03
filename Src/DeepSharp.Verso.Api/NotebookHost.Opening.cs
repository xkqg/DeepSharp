// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using DeepSharp.Verso.Notebooks;
using Verso;
using Verso.Abstractions;
using Verso.Extensions;

namespace DeepSharp.Verso.Api;

// Opening the notebook a file holds, saving it, and closing it: the parts DeepSharp registers with an engine, the open
// that reads the file the way Verso's editors read one, the save that writes it whole, and the close that ends every
// view and stops what runs.
public sealed partial class NotebookHost
{
    /// <summary>
    /// Saves the notebook to its file the way Verso's own editors save it: through the serializer for its format, which
    /// leaves out what a block shows, past the guards that run before writing, and written whole under a name of its own
    /// before it takes the file's place, so nothing ever meets half a notebook.
    /// </summary>
    /// <returns>When it is saved.</returns>
    public Task SaveAsync() => TurnAsync(async () =>
    {
        await _file.SaveAsync();

        return true;
    });

    /// <summary>Saves the notebook under another name, which is its file from then on.</summary>
    /// <param name="path">The new file, as a full path.</param>
    /// <returns>When it is saved there.</returns>
    /// <remarks>Only its holder names it, since the holder knows which files are open; <see cref="OpenNotebooks.SaveAsAsync"/>.</remarks>
    internal Task SaveAsAsync(string path) => TurnAsync(async () =>
    {
        await _file.SaveAsAsync(path);

        return true;
    });

    /// <summary>Registers DeepSharp's parts with an engine, before it looks for any beside the application.</summary>
    /// <param name="extensions">The engine.</param>
    /// <returns>When they are registered.</returns>
    /// <remarks>
    /// First, because Verso's own look beside the application passes over a part already registered, while one
    /// registered after it would be refused as a second.
    /// </remarks>
    internal static async Task RegisterAsync(ExtensionHost extensions)
    {
        foreach (var part in Parts())
        {
            await extensions.LoadExtensionAsync(part);
        }
    }

    /// <summary>Every part DeepSharp's notebook has, made anew for one engine.</summary>
    /// <returns>The parts.</returns>
    internal static IExtension[] Parts() =>
    [
        new StepCellType(),
        new StepKernel(),
        new StepRenderer(),
        new StepForm(),
        new RunPipelineAction(),
        new ExportPipelineAction(),
        new TakeOverAction(),
        new FormatGuard(),
        new RestoreBlocksAction(),
    ];

    /// <summary>Opens the notebook a file holds on an engine, and closes the engine when it cannot be opened.</summary>
    /// <param name="filePath">The notebook's file, as a full path.</param>
    /// <param name="extensions">The engine to open it on; closed when the notebook cannot be opened.</param>
    /// <param name="cancellationToken">Stops the open.</param>
    /// <returns>The host.</returns>
    /// <exception cref="NotSupportedException">No format Verso knows reads the file.</exception>
    internal static async Task<NotebookHost> OpenAsync(string filePath, ExtensionHost extensions, CancellationToken cancellationToken)
    {
        Scaffold? scaffold = null;

        try
        {
            var content = await File.ReadAllTextAsync(filePath, cancellationToken);

            extensions.ConsentHandler = static (_, _) => Task.FromResult(false);
            await RegisterAsync(extensions);
            await extensions.LoadBuiltInExtensionsAsync();

            var serializer = extensions.GetSerializers().FirstOrDefault(each => each.CanImport(filePath))
                ?? throw new NotSupportedException($"No format Verso knows reads '{Path.GetFileName(filePath)}'.");
            var notebook = await serializer.ReadAsync(extensions, content, filePath);
            var saved = await serializer.ReadAsync(extensions, content, filePath);

            // A file can repeat a cell's id — cells copied by hand, or by a tool — and nothing tells such cells apart by it,
            // so each repeat is given an id of its own, as Jupyter's own reader repairs repeated cell ids.
            var ids = new HashSet<Guid>();

            foreach (var cell in notebook.Cells)
            {
                if (!ids.Add(cell.Id))
                {
                    cell.Id = Guid.NewGuid();
                }
            }

            scaffold = new Scaffold(notebook, extensions, filePath);
            scaffold.InitializeSubsystems();
            extensions.EnsureDefaults(scaffold);
            await scaffold.RestoreAsync();
            await scaffold.RenderTransientCellsAsync(cancellationToken);

            // Drawn first, since drawing places what the layout had not placed yet; what differs from the file as it opens —
            // a cell's id repaired, a tile placed afresh — is unsaved from the first version.
            var arrangement = await scaffold.ArrangementAsync();

            await scaffold.FlushAsync();

            return new NotebookHost(
                filePath,
                extensions,
                scaffold,
                new Opening(saved, await extensions.ButtonsAsync(scaffold), saved.DiffersFrom(scaffold, filePath, extensions), arrangement));
        }
        catch
        {
            if (scaffold is not null)
            {
                await scaffold.DisposeAsync();
            }

            await extensions.DisposeAsync();

            throw;
        }
    }

    /// <summary>
    /// Closes the notebook, when no view has shown it for the grace, nothing runs or waits — nor runs on, left behind by a
    /// stop — and nothing in it differs from the file it was last saved to; otherwise leaves it open.
    /// </summary>
    /// <param name="grace">How long no view must have shown it.</param>
    /// <param name="forget">Lets its holder forget it, before its engine closes.</param>
    /// <returns>Whether it is still open.</returns>
    /// <remarks>
    /// It takes its turn, so nothing runs while it looks, and a cell still showing more keeps it open because the version
    /// that cell is in has not been published yet. What differs from the file is what Verso's own comparison of two
    /// notebooks finds, told which cells' outputs are never saved: what a block shows never counts.
    /// </remarks>
    internal Task<bool> StaysOpenAsync(TimeSpan grace, Action forget) => _turns.TakeTurnAsync(async () =>
    {
        if (Volatile.Read(ref _audience).Closed)
        {
            return false;
        }

        if (Volatile.Read(ref _pending) > 0 || Executing.Count > 0 || await _file.UnsavedAsync(takeBack: true))
        {
            return true;
        }

        // One swap decides it, so a view that begins meanwhile keeps the notebook open.
        if (!ImmutableInterlocked.Update(ref _audience, audience => audience.Views.IsEmpty && audience.AloneFor >= grace ? Audience.Gone : audience))
        {
            return true;
        }

        forget();
        await CloseEngineAsync();

        return false;
    });

    /// <summary>
    /// Shuts the notebook, the first half of closing it, at once: every view ends, anything asked from now on is refused,
    /// and the run under way is stopped as <see cref="Stop"/> stops it — one that waits never runs, one that runs is left
    /// behind — so no close waits for a run.
    /// </summary>
    internal void Shut()
    {
        foreach (var view in Interlocked.Exchange(ref _audience, Audience.Gone).Views)
        {
            view.End();
        }

        Volatile.Read(ref _running)?.Stop();
    }

    /// <summary>
    /// Closes the notebook: it is shut, so every view ends, anything asked from now on is refused and the run under way is
    /// stopped; a change under way finishes, and then the engine closes.
    /// </summary>
    /// <returns>When it is closed.</returns>
    internal async ValueTask CloseAsync()
    {
        Shut();

        await _turns.TakeTurnAsync(async () =>
        {
            await CloseEngineAsync();

            return true;
        });
    }

    private async Task CloseEngineAsync()
    {
        Scaffold.OnCellExecuting -= Began;
        Scaffold.OnCellExecuted -= Ended;
        Scaffold.OnCellOutputUpdated -= Showed;
        Scaffold.OnKernelRestarting -= Restarting;
        Scaffold.OnKernelRestarted -= Restarted;
        Scaffold.OnKernelRestartFailed -= RestartFailed;
        await Scaffold.DisposeAsync();
        await Extensions.DisposeAsync();
    }

    // How the notebook stands as it opens: what its file holds, the toolbar's buttons, whether it differs from the file
    // already, and how its layout arranges it.
    private readonly record struct Opening(NotebookModel Saved, IReadOnlyList<HostedToolbarAction> Buttons, bool Unsaved, HostedArrangement Arrangement);
}
