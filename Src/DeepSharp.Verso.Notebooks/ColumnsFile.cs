// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>What the file beside a notebook holds now.</summary>
/// <param name="Text">Its text as read; nothing when there is no file, or it cannot be read.</param>
/// <param name="Preset">What it decided; nothing when there is no file, or it cannot be read.</param>
/// <param name="Unreadable">Why it cannot be read, as the block says it; nothing when it can, or there is no file.</param>
internal readonly record struct StoredColumns(string? Text, PipelinePreset? Preset, CellOutput? Unreadable);

/// <summary>What saving beside a notebook would write, worked out before anything is written.</summary>
/// <param name="Text">The text to write; nothing when the file holds it already, or nothing is to be saved.</param>
/// <param name="Unreadable">Why the file is not written over, as the block says it; nothing when it can be.</param>
internal readonly record struct ColumnsSave(string? Text, CellOutput? Unreadable)
{
    /// <summary>What saving the source's columns a list showed would write, and nothing else: the decisions stay.</summary>
    /// <param name="stored">What the file held when the list was drawn.</param>
    /// <param name="shown">The source's columns the list showed.</param>
    /// <returns>The save; nothing to write when the file holds them now, or there is no file.</returns>
    public static ColumnsSave OfSource(StoredColumns stored, IReadOnlyList<string> shown)
    {
        if (stored.Preset is not { } preset)
        {
            return default;
        }

        var text = new PipelinePreset(preset.Declare, preset.Drop, preset.Output, shown).ToJson();

        return text == stored.Text ? default : new ColumnsSave(text, null);
    }
}

/// <summary>
/// The file beside a notebook that holds what its blocks decided about their columns: the schema with the columns it
/// excludes, the columns dropped, the output, and the source's columns the list last showed.
/// </summary>
/// <param name="Path">Where the file is.</param>
/// <remarks>
/// Written whole after every change the blocks accept and every run of the whole pipeline, and read back through the
/// verbs a notebook knows, as a pipeline file is. The source's columns it holds are the list's to write: a list drawn
/// writes them and nothing else, a change made from the list writes the header it showed, and every other write keeps
/// the ones the file already has. What a save writes is worked out first and written after, so a block lets the writing
/// through whole once a stop can no longer reach it. A file is written under a name of its own for each write first and
/// then moved into place, so nothing ever meets half of one and no two writes share a name; and a file that cannot be
/// read is never written over, since it may hold what this notebook cannot see.
/// </remarks>
internal readonly record struct ColumnsFile(string Path)
{
    /// <summary>What the file holds now.</summary>
    /// <returns>Its text and what it decided; or why it cannot be read; or nothing, when there is no file.</returns>
    public StoredColumns Stored()
    {
        if (!File.Exists(Path))
        {
            return default;
        }

        try
        {
            var text = File.ReadAllText(Path);

            return new StoredColumns(text, PipelinePreset.FromJson(text, NotebookVerbs.Catalog()), null);
        }
        catch (PipelineFileException unreadable)
        {
            return new StoredColumns(null, null, StepCard.ColumnsUnreadable([.. unreadable.Faults.Select(fault => fault.ToString())]));
        }
        catch (Exception unreadable) when (unreadable is IOException or UnauthorizedAccessException)
        {
            return new StoredColumns(null, null, StepCard.ColumnsUnreadable([unreadable.Message]));
        }
    }

    /// <summary>What saving what a declaration decided about its columns would write.</summary>
    /// <param name="declaration">The declaration the whole notebook makes.</param>
    /// <param name="shown">
    /// The source's columns a list showed, for a change made from it; nothing to keep the ones the file holds.
    /// </param>
    /// <returns>The save: the text, or why the file is not written over; nothing to write when it holds the decisions now.</returns>
    public ColumnsSave Saving(PipelineDeclaration declaration, IReadOnlyList<string>? shown = null)
    {
        // A pipeline without a schema has decided nothing about its columns.
        if (!declaration.Steps.OfType<DeclareStep>().Any())
        {
            return default;
        }

        var stored = Stored();

        if (stored.Unreadable is { } unreadable)
        {
            return new ColumnsSave(null, unreadable);
        }

        var text = PipelinePreset.Of(declaration, shown ?? stored.Preset?.Source).ToJson();

        return text == stored.Text ? default : new ColumnsSave(text, null);
    }

    /// <summary>Writes what a save worked out.</summary>
    /// <param name="save">The save.</param>
    /// <returns>Why nothing was saved, as the block says it; nothing when the file holds what the save meant it to.</returns>
    public CellOutput? Write(ColumnsSave save) => save.Unreadable ?? (save.Text is { } text ? Written(text) : null);

    // Written under a name of its own for this write and moved into place; a write or a move refused leaves nothing of its
    // own behind — the name is this write's alone, so nothing of this program holds it.
    private CellOutput? Written(string text)
    {
        var own = $"{Path}.{Guid.NewGuid():N}.tmp";

        try
        {
            File.WriteAllText(own, text);
            File.Move(own, Path, overwrite: true);

            return null;
        }
        catch (Exception refused) when (refused is IOException or UnauthorizedAccessException)
        {
            File.Delete(own);

            return StepCard.ColumnsNotWritten(refused.Message);
        }
    }
}
