// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using Verso.Extensions;

namespace DeepSharp.Verso.Api;

/// <summary>The notebooks an application has open: one host for each file, however many views show it.</summary>
/// <remarks>
/// A file is known by its full path, compared the way the platform compares file names, so two names of one file are
/// one notebook, and it is opened once however many callers ask for it at the same moment. An open that fails is
/// forgotten, so the next one tries again. The application keeps one of these for as long as it hosts notebooks.
/// </remarks>
public sealed class OpenNotebooks : IAsyncDisposable
{
    // Windows and macOS compare file names without regard to case; every other system with it. Both are asked.
    private readonly ConcurrentDictionary<string, Lazy<Task<NotebookHost>>> _open =
        new(FileNames(ignoringCase: OperatingSystem.IsWindows() | OperatingSystem.IsMacOS()));

    private int _closed;

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

        if (!_open.TryGetValue(from, out var holding) || !ReferenceEquals(await holding.Value, notebook))
        {
            throw new InvalidOperationException($"These notebooks do not hold '{from}'.");
        }

        var file = Path.GetFullPath(path);

        if (_open.Comparer.Equals(file, from))
        {
            await notebook.SaveAsync();

            return;
        }

        // The new name is taken before anything is written, so no one opens it as a notebook of its own meanwhile.
        var kept = new Lazy<Task<NotebookHost>>(Task.FromResult(notebook));

        if (!_open.TryAdd(file, kept))
        {
            throw new InvalidOperationException($"'{file}' is open already, as a notebook of its own.");
        }

        try
        {
            await notebook.SaveAsAsync(file);
        }
        catch
        {
            _open.TryRemove(KeyValuePair.Create(file, kept));

            throw;
        }

        _open.TryRemove(KeyValuePair.Create(from, holding));
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

    // Opens the file on an engine of its own; an open that fails, or finishes after these notebooks were closed, is
    // forgotten, and closes what it built.
    private async Task<NotebookHost> OpenOrForgetAsync(string file, Lazy<Task<NotebookHost>> opening)
    {
        try
        {
            var host = await NotebookHost.OpenAsync(file, new ExtensionHost(), CancellationToken.None);

            if (Volatile.Read(ref _closed) == 0)
            {
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
}
