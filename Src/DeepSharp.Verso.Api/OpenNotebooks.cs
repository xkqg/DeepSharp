// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using Verso.Extensions;

namespace DeepSharp.Verso.Api;

/// <summary>The notebooks an application has open: one host for each file, however many views show it.</summary>
/// <remarks>
/// A file is known by its full path, compared the way the platform compares file names, so two names of one file are
/// one notebook, and it is opened once however many callers ask for it at the same moment. An open that fails is
/// forgotten, so the next one tries again. The application keeps one of these for as long as it hosts notebooks. Given a
/// grace, a notebook closes by itself once no view has shown it for that long, nothing runs or waits, and nothing in it
/// differs from the file it was last saved to; without one, a notebook stays open until it is closed.
/// </remarks>
public sealed class OpenNotebooks : IAsyncDisposable
{
    // Windows and macOS compare file names without regard to case; every other system with it. Both are asked.
    private readonly ConcurrentDictionary<string, Lazy<Task<NotebookHost>>> _open =
        new(FileNames(ignoringCase: OperatingSystem.IsWindows() | OperatingSystem.IsMacOS()));

    private readonly TimeSpan? _grace;

    // Ends every watch when these notebooks close.
    private readonly CancellationTokenSource _closing = new();

    private int _closed;

    /// <summary>Notebooks that stay open until they are closed.</summary>
    public OpenNotebooks()
    {
    }

    /// <summary>Notebooks that each close by themselves once no view has shown them for a while.</summary>
    /// <param name="grace">
    /// How long no view must have shown a notebook before it closes — for a browser, longer than a page takes to load again.
    /// A notebook with a run under way, something waiting, or changes not yet saved stays open.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">The grace is nothing, or less.</exception>
    public OpenNotebooks(TimeSpan grace)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(grace, TimeSpan.Zero);

        _grace = grace;
    }

    /// <summary>How file names are compared on a file system that does, or does not, ignore case.</summary>
    /// <param name="ignoringCase">Whether the file system takes two names that differ only in case for one file.</param>
    /// <returns>The comparer.</returns>
    internal static StringComparer FileNames(bool ignoringCase) => ignoringCase ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;

    /// <summary>Opens the notebook saved at a path, or hands back the host already holding it.</summary>
    /// <param name="path">The notebook's file.</param>
    /// <param name="cancellationToken">Stops waiting for the notebook; an open other callers share goes on.</param>
    /// <returns>The notebook's host.</returns>
    /// <exception cref="ObjectDisposedException">These notebooks were closed.</exception>
    public Task<NotebookHost> OpenAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _closed) != 0, this);

        var file = Path.GetFullPath(path);
        Lazy<Task<NotebookHost>>? made = null;

        made = new Lazy<Task<NotebookHost>>(() => OpenOrForgetAsync(file, made!));

        return _open.GetOrAdd(file, made).Value.WaitAsync(cancellationToken);
    }

    /// <summary>Saves a notebook these hold under another name; from then on it is that file, and the old one only a file.</summary>
    /// <param name="notebook">The notebook.</param>
    /// <param name="path">Where to save it.</param>
    /// <returns>When it is saved there.</returns>
    /// <exception cref="InvalidOperationException">
    /// These notebooks do not hold it, or another open notebook is that file already.
    /// </exception>
    /// <remarks>Its own name saves it where it is.</remarks>
    public async Task SaveAsAsync(NotebookHost notebook, string path)
    {
        ArgumentNullException.ThrowIfNull(notebook);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var from = notebook.FilePath;
        var holding = await HoldingAsync(notebook);
        var file = Path.GetFullPath(path);

        if (_open.Comparer.Equals(file, from))
        {
            await notebook.SaveAsync();

            return;
        }

        // The new name is taken before anything is written, so no one opens it as a notebook of its own meanwhile; it is
        // taken by the same entry, so whoever watches the notebook finds it under either name.
        if (!_open.TryAdd(file, holding))
        {
            throw new InvalidOperationException($"'{file}' is open already, as a notebook of its own.");
        }

        try
        {
            await notebook.SaveAsAsync(file);
        }
        catch
        {
            _open.TryRemove(KeyValuePair.Create(file, holding));

            throw;
        }

        _open.TryRemove(KeyValuePair.Create(from, holding));
    }

    /// <summary>Closes a notebook these hold at once, whatever it holds that is not saved.</summary>
    /// <param name="notebook">The notebook.</param>
    /// <returns>When it is closed: its views ended, what was under way finished, its engine closed.</returns>
    /// <exception cref="InvalidOperationException">These notebooks do not hold it.</exception>
    public async Task CloseAsync(NotebookHost notebook)
    {
        ArgumentNullException.ThrowIfNull(notebook);

        _open.TryRemove(KeyValuePair.Create(notebook.FilePath, await HoldingAsync(notebook)));
        await notebook.CloseAsync();
    }

    /// <summary>The files open now, as their full paths.</summary>
    internal IReadOnlyCollection<string> Paths => [.. _open.Keys];

    /// <summary>Closes every notebook this holds, once.</summary>
    /// <returns>When they are closed.</returns>
    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _closed, 1) != 0)
        {
            return;
        }

        await _closing.CancelAsync();

        // An open still under way sees the close as it finishes and closes what it built itself, as an open that failed
        // already has; every other one is closed here.
        foreach (var opening in _open.Values)
        {
            var open = opening.Value;

            await Task.WhenAny(open);

            if (open.IsCompletedSuccessfully)
            {
                await open.Result.CloseAsync();
            }
        }

        _open.Clear();
    }

    // The entry holding a notebook, which must be this one and not another of the same file.
    private async Task<Lazy<Task<NotebookHost>>> HoldingAsync(NotebookHost notebook)
    {
        var from = notebook.FilePath;

        return _open.TryGetValue(from, out var holding) && ReferenceEquals(await holding.Value, notebook)
            ? holding
            : throw new InvalidOperationException($"These notebooks do not hold '{from}'.");
    }

    // Opens the file on an engine of its own; an open that fails, or finishes after these notebooks were closed, is
    // forgotten, and closes what it built. Given a grace, the notebook is watched from then on.
    private async Task<NotebookHost> OpenOrForgetAsync(string file, Lazy<Task<NotebookHost>> opening)
    {
        try
        {
            var host = await NotebookHost.OpenAsync(file, new ExtensionHost(), CancellationToken.None);

            if (Volatile.Read(ref _closed) == 0)
            {
                if (_grace is { } grace)
                {
                    _ = WatchAsync(host, opening, grace);
                }

                return host;
            }

            await host.CloseAsync();

            throw new ObjectDisposedException(GetType().FullName);
        }
        catch
        {
            _open.TryRemove(KeyValuePair.Create(file, opening));

            throw;
        }
    }

    // Looks at a notebook once every grace, until it closes or these notebooks do; closing, it is forgotten under the name
    // it has then.
    private async Task WatchAsync(NotebookHost host, Lazy<Task<NotebookHost>> holding, TimeSpan grace)
    {
        do
        {
            await Task.Delay(grace, _closing.Token).ConfigureAwait(ConfigureAwaitOptions.SuppressThrowing);
        }
        while (!_closing.IsCancellationRequested && await host.StaysOpenAsync(grace, () => _open.TryRemove(KeyValuePair.Create(host.FilePath, holding))));
    }
}
