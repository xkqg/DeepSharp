// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// The file beside a notebook that holds what its blocks decided about their columns: the schema with the columns it
/// excludes, the columns dropped, the output, and the source's columns the list last showed.
/// </summary>
/// <param name="Path">Where the file is.</param>
/// <remarks>
/// Written whole after every change the blocks accept and every run of the whole pipeline, and read back through the
/// verbs a notebook knows, as a pipeline file is. The source's columns it holds are the list's to write, so a write
/// keeps the ones the file already has. A file is written under a name of its own first and then moved into place, so
/// nothing ever meets half of one; and a file that cannot be read is never written over, since it may hold what this
/// notebook cannot see.
/// </remarks>
internal readonly record struct ColumnsFile(string Path)
{
    /// <summary>Saves what a declaration decided about its columns.</summary>
    /// <param name="declaration">The declaration the whole notebook makes.</param>
    /// <returns>Why nothing was saved, as the block says it; nothing when the file holds the decisions now.</returns>
    public CellOutput? Save(PipelineDeclaration declaration)
    {
        // A pipeline without a schema has decided nothing about its columns.
        if (!declaration.Steps.OfType<DeclareStep>().Any())
        {
            return null;
        }

        string? stored = null;
        IReadOnlyList<string>? source = null;

        if (File.Exists(Path))
        {
            try
            {
                stored = File.ReadAllText(Path);
                source = PipelinePreset.FromJson(stored, NotebookVerbs.Catalog()).Source;
            }
            catch (PipelineFileException unreadable)
            {
                return StepCard.ColumnsUnreadable([.. unreadable.Faults.Select(fault => fault.ToString())]);
            }
            catch (Exception unreadable) when (unreadable is IOException or UnauthorizedAccessException)
            {
                return StepCard.ColumnsUnreadable([unreadable.Message]);
            }
        }

        var text = PipelinePreset.Of(declaration, source).ToJson();

        return text == stored ? null : Written(text);
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
