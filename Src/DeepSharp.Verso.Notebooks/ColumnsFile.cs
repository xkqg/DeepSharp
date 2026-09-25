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

/// <summary>
/// The file beside a notebook that holds what its blocks decided about their columns: the schema with the columns it
/// excludes, the columns dropped, the output, and the source's columns the list last showed.
/// </summary>
/// <param name="Path">Where the file is.</param>
/// <remarks>
/// Written whole after every change the blocks accept and every run of the whole pipeline, and read back through the
/// verbs a notebook knows, as a pipeline file is. The source's columns it holds are the list's to write: a list drawn
/// writes them and nothing else, a change made from the list writes the header it showed, and every other write keeps
/// the ones the file already has. A file is written under a name of its own first and then moved into place, so
/// nothing ever meets half of one; and a file that cannot be read is never written over, since it may hold what this
/// notebook cannot see.
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

    /// <summary>Saves what a declaration decided about its columns.</summary>
    /// <param name="declaration">The declaration the whole notebook makes.</param>
    /// <param name="shown">
    /// The source's columns a list showed, for a change made from it; nothing to keep the ones the file holds.
    /// </param>
    /// <returns>Why nothing was saved, as the block says it; nothing when the file holds the decisions now.</returns>
    public CellOutput? Save(PipelineDeclaration declaration, IReadOnlyList<string>? shown = null)
    {
        // A pipeline without a schema has decided nothing about its columns.
        if (!declaration.Steps.OfType<DeclareStep>().Any())
        {
            return null;
        }

        var stored = Stored();

        if (stored.Unreadable is { } unreadable)
        {
            return unreadable;
        }

        var text = PipelinePreset.Of(declaration, shown ?? stored.Preset?.Source).ToJson();

        return text == stored.Text ? null : Written(text);
    }

    /// <summary>Saves the source's columns a list showed, and nothing else: the decisions stay as the file holds them.</summary>
    /// <param name="stored">What the file held when the list was drawn.</param>
    /// <param name="shown">The source's columns the list showed.</param>
    /// <returns>Why nothing was saved, as the block says it; nothing when the file holds them now, or there is no file.</returns>
    public CellOutput? SaveSource(StoredColumns stored, IReadOnlyList<string> shown)
    {
        if (stored.Preset is not { } preset)
        {
            return null;
        }

        var text = new PipelinePreset(preset.Declare, preset.Drop, preset.Output, shown).ToJson();

        return text == stored.Text ? null : Written(text);
    }

    // Written under a name of its own and moved into place; a move refused leaves nothing of its own behind.
    private CellOutput? Written(string text)
    {
        var own = $"{Path}.{Environment.ProcessId}.tmp";

        try
        {
            File.WriteAllText(own, text);
            File.Move(own, Path, overwrite: true);

            return null;
        }
        catch (Exception refused) when (refused is IOException or UnauthorizedAccessException)
        {
            try
            {
                File.Delete(own);
            }
            catch (Exception left) when (left is IOException or UnauthorizedAccessException)
            {
                // Left behind: this process writes under the same name the next time, over it.
            }

            return StepCard.ColumnsNotWritten(refused.Message);
        }
    }
}
