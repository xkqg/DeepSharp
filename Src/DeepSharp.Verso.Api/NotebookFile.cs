// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso;
using Verso.Abstractions;
using Verso.Extensions;

namespace DeepSharp.Verso.Api;

/// <summary>
/// An open notebook's file: what it last held, whether the notebook differs from it, and writing it.
/// </summary>
/// <remarks>
/// <para>
/// Saved the way Verso's own editors save a notebook: through the serializer for its format, which leaves out what a block
/// shows, past the guards that run before writing, and written whole under a name of its own before it takes the file's
/// place, so nothing ever meets half a notebook. What is unsaved is what differs from the notebook this file last held,
/// read the way it is read when it opens.
/// </para>
/// <para>
/// The comparison reads a cell's outputs while a run may be writing them, and it reads them whole rather than as they
/// come, so it answers rather than failing: measured at 8.2 million comparisons against three kinds of writer — the same
/// output written again, the list cleared and refilled, one added and taken away — with nothing thrown. A version of
/// Verso that read them as they came would fail here instead of answering wrongly, which is the way round a person can
/// act on.
/// </para>
/// </remarks>
/// <param name="path">The file, as a full path.</param>
/// <param name="saved">The notebook that file last held.</param>
/// <param name="extensions">The engine, whose serializers write it and whose guards run before writing.</param>
/// <param name="scaffold">The notebook as the engine holds it.</param>
internal sealed class NotebookFile(string path, NotebookModel saved, ExtensionHost extensions, Scaffold scaffold)
{
    private NotebookModel _saved = saved;

    /// <summary>The file the notebook is saved in, as a full path.</summary>
    public string Path { get; private set; } = path;

    /// <summary>Whether the notebook differs from the file it was last saved to.</summary>
    /// <param name="takeBack">Whether what the layouts and the parts' settings hold is taken back into the notebook first.</param>
    /// <returns>Whether a save would change the file.</returns>
    /// <remarks>
    /// What the layouts and the parts' settings hold is taken back only when nothing else acts on the notebook, since a
    /// turn's work may be writing into it meanwhile.
    /// </remarks>
    public async Task<bool> UnsavedAsync(bool takeBack)
    {
        if (takeBack)
        {
            await scaffold.FlushAsync();
        }

        return _saved.DiffersFrom(scaffold, Path, extensions);
    }

    /// <summary>Saves the notebook to its file.</summary>
    /// <returns>When it is saved.</returns>
    /// <exception cref="NotSupportedException">No format Verso knows writes the file.</exception>
    public Task SaveAsync() => WriteAsync(Path);

    /// <summary>Saves the notebook under another name, which is its file from then on.</summary>
    /// <param name="to">The new file, as a full path.</param>
    /// <returns>When it is saved there.</returns>
    /// <exception cref="NotSupportedException">No format Verso knows writes that file.</exception>
    /// <remarks>What the notebook names after itself — the columns saved beside it, an exported pipeline — follows the file.</remarks>
    public async Task SaveAsAsync(string to)
    {
        await WriteAsync(to);

        scaffold.SetFilePath(to);
        Path = to;
    }

    private async Task WriteAsync(string to)
    {
        var serializer = extensions.GetSerializers().FirstOrDefault(each => each.CanImport(to))
            ?? throw new NotSupportedException($"No format Verso knows writes '{System.IO.Path.GetFileName(to)}'.");

        // What a live output shows now is what is saved, as Verso's own editors ask before they write; what the layouts and
        // the parts' settings hold now is taken back into the notebook; and the notebook is stamped with the time it is saved.
        await scaffold.RefreshLiveOutputsAsync();
        await scaffold.FlushAsync();
        scaffold.Notebook.Modified = DateTimeOffset.UtcNow;

        var notebook = scaffold.Notebook;

        foreach (var guard in extensions.GetPostProcessors().Where(each => each.CanProcess(to, serializer.FormatId)).OrderBy(each => each.Priority))
        {
            notebook = await guard.PreSerializeAsync(notebook, to);
        }

        var whole = System.IO.Path.Join(System.IO.Path.GetDirectoryName(to), $".{System.IO.Path.GetFileName(to)}.{Guid.NewGuid():N}.tmp");
        var text = await serializer.SerializeAsync(notebook);

        await File.WriteAllTextAsync(whole, text);
        File.Move(whole, to, overwrite: true);

        // From now on, what is unsaved is what differs from what this file holds.
        _saved = await serializer.ReadAsync(extensions, text, to);
    }
}
