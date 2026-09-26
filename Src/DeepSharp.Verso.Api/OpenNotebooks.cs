// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Extensions;
using Verso.Serializers;

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

    // The close of every notebook these hold, made once: whoever asks again waits for the same close.
    private readonly Lazy<Task> _disposed;

    private int _closed;

    /// <summary>Notebooks that stay open until they are closed.</summary>
    public OpenNotebooks() => _disposed = new(CloseEveryNotebookAsync);

    /// <summary>Notebooks that each close by themselves once no view has shown them for a while.</summary>
    /// <param name="grace">
    /// How long no view must have shown a notebook before it closes — for a browser, longer than a page takes to load again.
    /// A notebook with a run under way, something waiting, or changes not yet saved stays open.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">The grace is nothing, or less.</exception>
    public OpenNotebooks(TimeSpan grace)
        : this()
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

    /// <summary>
    /// Makes a new notebook at a path and opens it: one block, the step that reads a CSV file, written whole under a name
    /// of its own before it takes its place, so nothing ever meets half a notebook — and never over a file already there.
    /// </summary>
    /// <param name="path">Where it is saved: a .verso file, the one format that keeps a block a block.</param>
    /// <param name="cancellationToken">Stops waiting for the new notebook to open; the file is made all the same.</param>
    /// <returns>The new notebook's host.</returns>
    /// <exception cref="ArgumentException">The path names a file of another format.</exception>
    /// <exception cref="IOException">A file is there already; it is left as it was.</exception>
    /// <exception cref="InvalidOperationException">An open notebook is that file already, whether or not the file is still there.</exception>
    /// <exception cref="ObjectDisposedException">These notebooks were closed.</exception>
    public async Task<NotebookHost> CreateAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _closed) != 0, this);

        var file = Path.GetFullPath(path);

        if (!file.EndsWith(".verso", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("A new notebook is a .verso file: the one format that keeps a block a block.", nameof(path));
        }

        if (_open.ContainsKey(file))
        {
            throw new InvalidOperationException($"'{file}' is open already, as a notebook of its own.");
        }

        var notebook = new NotebookModel { Title = Path.GetFileNameWithoutExtension(file), DefaultKernelId = "csharp", ActiveLayout = LayoutDefaults.Reference };

        notebook.Cells.Add(new CellModel { Type = StepCellType.StepType, Language = StepKernel.Language, Source = new StepCellType().GetDefaultContent() });

        // Written whole under a name of its own, then moved into place by the file system in one step that never replaces
        // a file: whoever looks finds no notebook, or the whole of it.
        var whole = Path.Join(Path.GetDirectoryName(file), $".{Path.GetFileName(file)}.{Guid.NewGuid():N}.tmp");

        await File.WriteAllTextAsync(whole, await new VersoSerializer().SerializeAsync(notebook), CancellationToken.None);

        try
        {
            File.Move(whole, file, overwrite: false);
        }
        catch
        {
            File.Delete(whole);

            throw;
        }

        return await OpenAsync(file, cancellationToken);
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

    /// <summary>
    /// Closes a notebook these hold at once, whatever it holds that is not saved: its run under way is stopped, never waited
    /// for, and a change under way finishes.
    /// </summary>
    /// <param name="notebook">The notebook.</param>
    /// <returns>When it is closed: its views ended, its run stopped, a change under way finished, its engine closed.</returns>
    /// <exception cref="InvalidOperationException">These notebooks do not hold it.</exception>
    public async Task CloseAsync(NotebookHost notebook)
    {
        ArgumentNullException.ThrowIfNull(notebook);

        _open.TryRemove(KeyValuePair.Create(notebook.FilePath, await HoldingAsync(notebook)));
        await notebook.CloseAsync();
    }

    /// <summary>The files open now, as their full paths.</summary>
    internal IReadOnlyCollection<string> Paths => [.. _open.Keys];

    /// <summary>
    /// Closes every notebook this holds, once: the run under way in every one is stopped before any of them is closed, so a
    /// run that waited for another notebook's C# run never starts; a change under way finishes. Asked again, it waits for
    /// the same close.
    /// </summary>
    /// <returns>When they are closed.</returns>
    public ValueTask DisposeAsync() => new(_disposed.Value);

    // Shuts every notebook these hold, with nothing awaited in between, and then closes each: stopping one notebook's run
    // hands the C# turn on, and a run another notebook still let wait for it would start.
    private async Task CloseEveryNotebookAsync()
    {
        Interlocked.Exchange(ref _closed, 1);

        foreach (var opening in _open.Values)
        {
            if (opening.Value is { IsCompletedSuccessfully: true } open)
            {
                open.Result.Shut();
            }
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
