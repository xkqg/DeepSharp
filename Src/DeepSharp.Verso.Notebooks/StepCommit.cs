// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>A gesture, with the notebook it was made in.</summary>
/// <param name="Session">The notebook's session, which the block's kernel reads.</param>
/// <param name="Notebook">The notebook, every cell of it.</param>
/// <param name="Operations">What can be done to the notebook: running a cell, clearing one, adding and removing one.</param>
/// <param name="Variables">The values the notebook's cells share.</param>
/// <param name="Cell">The block the gesture was made on.</param>
/// <param name="MayAddAndRemove">Whether the layout the notebook is shown in lets a part add and remove blocks.</param>
internal readonly record struct Gesture(
    NotebookSession Session, NotebookModel Notebook, INotebookOperations Operations, IVariableStore Variables, Guid Cell,
    bool MayAddAndRemove);

/// <summary>What a commit does to one block.</summary>
internal enum BlockChangeKind
{
    /// <summary>Left as it is: its step did not change.</summary>
    Kept,

    /// <summary>Written again in its own place, as a block no front end has seen, keeping what the notebook holds about it.</summary>
    Rewritten,

    /// <summary>A new block, after the block before it.</summary>
    Inserted,

    /// <summary>Taken out of the notebook.</summary>
    Removed,
}

/// <summary>What a commit does to one block, in the order the steps stand after it.</summary>
/// <param name="Kind">What is done.</param>
/// <param name="From">Where the block's step stood before, counting from nought, for a block kept, rewritten or removed.</param>
/// <param name="To">The step the block holds after, for a block rewritten or inserted.</param>
internal readonly record struct BlockChange(BlockChangeKind Kind, int? From, IPipelineStep? To);

/// <summary>
/// New steps written into a notebook's blocks, whatever made them: a gesture on a grid, a list of the columns, a take-over
/// of saved decisions.
/// </summary>
/// <remarks>
/// Only the blocks whose steps changed are written, each in its own place: the steps both lists start and end with are
/// left alone, and between them a step is matched with the one of the same verb in the same order, so it is written
/// again where it stood and keeps what the notebook holds about it — or left alone when it did not change at all, view
/// and all. A step added goes after the block before it and one taken away is removed. Steps the rules refuse are not
/// written, and the block says which rule; the same steps again write nothing and run nothing. Verso tells a front end
/// nothing of a block whose text a part changed, and the next keystroke there would put the old text back, so a block
/// written is a new block in the old one's place, and every block written is run.
/// </remarks>
internal static class StepCommit
{
    /// <summary>What a commit does to each block, to go from one list of steps to another.</summary>
    /// <param name="before">The steps the blocks hold.</param>
    /// <param name="after">The steps they are to hold.</param>
    /// <returns>A change for every block, in the order the steps stand after it, a block removed where it stood.</returns>
    internal static IReadOnlyList<BlockChange> Changes(IReadOnlyList<IPipelineStep> before, IReadOnlyList<IPipelineStep> after)
    {
        var start = 0;

        while (start < before.Count && start < after.Count && before[start].Equals(after[start]))
        {
            start++;
        }

        var end = 0;

        while (end < before.Count - start && end < after.Count - start && before[^(end + 1)].Equals(after[^(end + 1)]))
        {
            end++;
        }

        List<BlockChange> changes = [.. Enumerable.Range(0, start).Select(at => new BlockChange(BlockChangeKind.Kept, at, null))];

        changes.AddRange(Between(before, after, start, before.Count - end, after.Count - end));
        changes.AddRange(Enumerable.Range(before.Count - end, end).Select(at => new BlockChange(BlockChangeKind.Kept, at, null)));

        return changes;
    }

    /// <summary>Writes new steps into the blocks, unless they are the same steps or the rules refuse them.</summary>
    /// <param name="gesture">The gesture that asked for them.</param>
    /// <param name="assembled">The pipeline the blocks made when the gesture was made.</param>
    /// <param name="steps">The steps the blocks are to hold, from the first block on.</param>
    /// <returns><see langword="true"/> when blocks were written; <see langword="false"/> when nothing was.</returns>
    internal static async Task<bool> CommitAsync(Gesture gesture, NotebookPipeline assembled, IReadOnlyList<IPipelineStep> steps)
    {
        var changes = Changes(assembled.Readable.Steps, steps);

        if (changes.All(change => change.Kind == BlockChangeKind.Kept))
        {
            return false;
        }

        IReadOnlyList<string> notMade = !gesture.MayAddAndRemove
            ? ["the layout this notebook is shown in cannot add or remove a block; show it in the notebook layout to change the pipeline from the grid."]
            : [.. PipelineDeclaration.FaultsIn(steps).Select(fault => fault.ToString())];

        if (notMade.Count > 0)
        {
            gesture.Session.Request(gesture.Cell, assembled.RequestFor(gesture.Cell, ViewTrigger.Commit, page: 0) with { NotMade = notMade });
            await gesture.Operations.ExecuteCellAsync(gesture.Cell);

            return false;
        }

        var written = await WriteAsync(gesture, assembled, changes);
        var now = NotebookPipeline.Of(gesture.Notebook.Cells);
        var shown = gesture with { Cell = written.Shown };

        foreach (var cell in gesture.Session.ForgetStale(now, except: shown.Cell))
        {
            // A block deleted since it was shown has nothing left to clear.
            if (gesture.Notebook.Cells.Any(each => each.Id == cell))
            {
                await gesture.Operations.ClearOutputAsync(cell);
            }
        }

        // A block written is one a front end has never seen, and its run is what makes a front end read the notebook
        // again; the block shown is run by showing it.
        foreach (var block in written.Blocks.Where(block => block != shown.Cell))
        {
            await gesture.Operations.ExecuteCellAsync(block);
        }

        await ShowAsync(shown, now, ViewTrigger.Commit, page: 0);

        return true;
    }

    /// <summary>Shows the data at the block a gesture was made on, by leaving its kernel a request and running it.</summary>
    /// <param name="gesture">The gesture.</param>
    /// <param name="assembled">The pipeline the blocks make.</param>
    /// <param name="trigger">What asked for it.</param>
    /// <param name="page">Which page of rows, counting from nought.</param>
    /// <returns>Nothing; or why a cell that is not a block shows nothing.</returns>
    internal static async Task<string?> ShowAsync(Gesture gesture, NotebookPipeline assembled, ViewTrigger trigger, int page)
    {
        if (assembled.PositionOf(gesture.Cell) < 0)
        {
            return "This cell is not a block of the pipeline.";
        }

        gesture.Session.Publish(assembled);

        gesture.Session.HandOver(gesture.Variables, assembled);
        gesture.Session.Request(gesture.Cell, assembled.RequestFor(gesture.Cell, trigger, page));

        await gesture.Operations.ExecuteCellAsync(gesture.Cell);

        return null;
    }

    // Between the steps both lists start and end with: the old and the new steps matched by verb, in order, the longest
    // run of them — a pair that is the same step is kept, any other pair rewritten in its place; a new step matched with
    // nothing is inserted, an old one matched with nothing removed.
    private static IEnumerable<BlockChange> Between(IReadOnlyList<IPipelineStep> before, IReadOnlyList<IPipelineStep> after, int start, int oldEnd, int newEnd)
    {
        var old = oldEnd - start;
        var made = newEnd - start;
        var matched = new int[old + 1, made + 1];

        for (var i = old - 1; i >= 0; i--)
        {
            for (var j = made - 1; j >= 0; j--)
            {
                matched[i, j] = before[start + i].Verb == after[start + j].Verb
                    ? matched[i + 1, j + 1] + 1
                    : Math.Max(matched[i + 1, j], matched[i, j + 1]);
            }
        }

        int o = 0, n = 0;

        while (o < old || n < made)
        {
            if (o < old && n < made && before[start + o].Verb == after[start + n].Verb)
            {
                yield return before[start + o].Equals(after[start + n])
                    ? new BlockChange(BlockChangeKind.Kept, start + o, null)
                    : new BlockChange(BlockChangeKind.Rewritten, start + o, after[start + n]);
                o++;
                n++;
            }
            else if (o < old && (n == made || matched[o + 1, n] >= matched[o, n + 1]))
            {
                yield return new BlockChange(BlockChangeKind.Removed, start + o, null);
                o++;
            }
            else
            {
                yield return new BlockChange(BlockChangeKind.Inserted, null, after[start + n]);
                n++;
            }
        }
    }

    // Writes the changes in the order the steps stand after them, each new block after the block before it.
    private static async Task<Written> WriteAsync(Gesture gesture, NotebookPipeline assembled, IReadOnlyList<BlockChange> changes)
    {
        var blocks = assembled.Blocks;
        var top = gesture.Notebook.Cells.FindIndex(cell => cell.Id == blocks[0].Cell);
        var written = new List<Guid>();
        var standing = new List<Guid>();
        var shown = gesture.Cell;
        int? goneAfter = null;

        foreach (var change in changes)
        {
            switch (change.Kind)
            {
                case BlockChangeKind.Kept:
                    standing.Add(blocks[change.From!.Value].Cell);
                    break;

                case BlockChangeKind.Rewritten:
                {
                    var old = gesture.Notebook.Cells.Single(cell => cell.Id == blocks[change.From!.Value].Cell);
                    var replacing = await InsertAsync(gesture, gesture.Notebook.Cells.IndexOf(old), change.To!);

                    foreach (var (key, value) in old.Metadata)
                    {
                        replacing.Metadata[key] = value;
                    }

                    await RemoveAsync(gesture, old.Id);
                    shown = old.Id == shown ? replacing.Id : shown;
                    standing.Add(replacing.Id);
                    written.Add(replacing.Id);
                    break;
                }

                case BlockChangeKind.Inserted:
                {
                    // After the block before it; the first step goes where the first block stood.
                    var at = standing.Count > 0 ? gesture.Notebook.Cells.FindIndex(cell => cell.Id == standing[^1]) + 1 : top;
                    var inserted = await InsertAsync(gesture, at, change.To!);

                    standing.Add(inserted.Id);
                    written.Add(inserted.Id);
                    break;
                }

                default:
                {
                    var gone = blocks[change.From!.Value].Cell;

                    await RemoveAsync(gesture, gone);
                    goneAfter = gone == shown ? standing.Count : goneAfter;
                    break;
                }
            }
        }

        // The block a gesture was made on is gone: the block standing before it shows what came of the gesture, or the
        // first block when none stands before it.
        return new Written(written, goneAfter is { } place ? standing[Math.Max(place - 1, 0)] : shown);
    }

    private static async Task<CellModel> InsertAsync(Gesture gesture, int at, IPipelineStep step)
    {
        var id = Guid.Parse(await gesture.Operations.InsertCellAsync(at, StepCellType.StepType, StepKernel.Language));
        var cell = gesture.Notebook.Cells.Single(each => each.Id == id);

        cell.Source = step.AsBlockText();

        return cell;
    }

    private static async Task RemoveAsync(Gesture gesture, Guid cell)
    {
        await gesture.Operations.RemoveCellAsync(cell);
        gesture.Session.Hidden(cell);
        gesture.Session.Accepted(cell);
    }

    /// <summary>What a commit wrote: every block written, and the block that shows what came of the gesture.</summary>
    /// <param name="Blocks">The blocks written, rewritten or new, in the order they stand.</param>
    /// <param name="Shown">The block the gesture was made on, or the block that stands for it now.</param>
    private readonly record struct Written(IReadOnlyList<Guid> Blocks, Guid Shown);
}
