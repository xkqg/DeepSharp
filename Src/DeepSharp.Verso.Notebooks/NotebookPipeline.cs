// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>One block of a notebook, as the pipeline reads it.</summary>
/// <param name="Cell">The cell the block is.</param>
/// <param name="Step">The step its text reads as, or nothing when it does not read as one.</param>
/// <param name="Faults">What is wrong with the block: its text, or a rule it breaks where it stands.</param>
internal readonly record struct Block(Guid Cell, IPipelineStep? Step, IReadOnlyList<string> Faults);

/// <summary>What asked a block's kernel to show something.</summary>
internal enum ViewTrigger
{
    /// <summary>"Show the data here", on a block.</summary>
    Show,

    /// <summary>Another page of the data a block shows.</summary>
    Page,

    /// <summary>A change a gesture made to the blocks, or asked for and did not get.</summary>
    Commit,

    /// <summary>The toolbar's run of the whole pipeline.</summary>
    Run,

    /// <summary>The toolbar's take-over of the columns saved beside the notebook: listed, and nothing changed yet.</summary>
    TakeOver,
}

/// <summary>What a list of the source's columns is drawn with: the picks a person made on it, which commit nothing.</summary>
/// <param name="Type">The kind of output the list's boxes make, when one was picked; nothing to draw the output's own.</param>
internal sealed record ListPicks(string? Type)
{
    /// <summary>A list drawn with no picks.</summary>
    public static ListPicks None { get; } = new(Type: null);
}

/// <summary>What a block's kernel is asked to show.</summary>
/// <param name="Declaration">The declaration to run, or nothing when the blocks down to this one do not make one.</param>
/// <param name="Position">The block's place in it, counting from nought.</param>
/// <param name="Page">Which page of rows to show, counting from nought.</param>
/// <param name="Faults">Why nothing can be shown, when nothing can.</param>
/// <param name="Trigger">What asked for it.</param>
/// <param name="Whole">Whether every block is in the declaration: the whole notebook makes one pipeline.</param>
/// <param name="NotMade">
/// A change a gesture asked for and that is not made, each rule it would break: said above the data, which is as it was.
/// </param>
/// <param name="Card">
/// What the block shows in place of its data, drawn by what asked for it: the list of a take-over, which reads no rows
/// and changes nothing. Nothing for every other request.
/// </param>
/// <param name="List">
/// The list of the source's columns, drawn with these picks, in place of the grid; nothing for the grid.
/// </param>
/// <remarks>
/// The whole pipeline is fitted only when the toolbar's run asks it of a notebook that makes one pipeline; every other
/// request shows the data at its block, the list of the source's columns, or the card it carries.
/// </remarks>
internal readonly record struct ViewRequest(
    PipelineDeclaration? Declaration, int Position, int Page, IReadOnlyList<string> Faults, ViewTrigger Trigger, bool Whole,
    IReadOnlyList<string> NotMade, CellOutput? Card = null, ListPicks? List = null)
{
    /// <summary>Whether to run the whole pipeline, fitting every step, and hand what it learned over.</summary>
    public bool RunsTheWholePipeline => Trigger == ViewTrigger.Run && Whole;

    /// <summary>
    /// Whether to save what the blocks decided about their columns beside the notebook: after a change the blocks
    /// accepted, or a run of the whole pipeline, when every block is in the pipeline.
    /// </summary>
    public bool SavesTheColumns => Trigger is ViewTrigger.Commit or ViewTrigger.Run && Whole && NotMade.Count == 0;
}

/// <summary>
/// The pipeline a notebook's blocks declare, read in the order the blocks stand, the way a file is read.
/// </summary>
/// <remarks>
/// Assembled afresh at every gesture from the cells as they are, through the declaration's own constructor, so a
/// block inserted, moved, deleted or edited is always seen: nothing about the order is remembered in between. The
/// part of it that is a declaration is the longest run of blocks from the top whose steps read and keep every rule;
/// any block can show its data if it stands in that run, and any block below says which block stops it.
/// </remarks>
internal sealed class NotebookPipeline
{
    private readonly Block[] _blocks;
    private IReadOnlyList<KnownColumn>? _known;

    private NotebookPipeline(Block[] blocks, IReadOnlyList<CellModel> cells, PipelineDeclaration readable)
    {
        _blocks = blocks;
        Cells = cells;
        Readable = readable;
    }

    /// <summary>The blocks, in the order they stand.</summary>
    public IReadOnlyList<Block> Blocks => _blocks;

    /// <summary>The cells the blocks were read from, in the order they stood.</summary>
    /// <remarks>
    /// Each cell's text, read again, is its text now; a block inserted, moved or deleted since is not in this list,
    /// and only assembling the notebook again sees it.
    /// </remarks>
    public IReadOnlyList<CellModel> Cells { get; }

    /// <summary>The longest run of blocks from the top that makes a declaration.</summary>
    public PipelineDeclaration Readable { get; }

    /// <summary>Whether every block is in the declaration: the whole notebook makes one pipeline.</summary>
    public bool Whole => Readable.Steps.Count == _blocks.Length;

    /// <summary>The block that declares the columns, when the blocks that make the pipeline hold one.</summary>
    public Guid? SchemaBlock
    {
        get
        {
            var at = Readable.Steps.TakeWhile(step => step is not DeclareStep).Count();

            return at < Readable.Steps.Count ? _blocks[at].Cell : null;
        }
    }

    /// <summary>Why the blocks do not make one pipeline: every fault of the first block at fault, each naming it.</summary>
    /// <remarks>None when they do.</remarks>
    public IReadOnlyList<string> Stopping
    {
        get
        {
            var stopping = Array.FindIndex(_blocks, block => block.Faults.Count > 0);

            if (stopping < 0)
            {
                return [];
            }

            var number = (stopping + 1).ToString(CultureInfo.InvariantCulture);

            return [.. _blocks[stopping].Faults.Select(fault => $"block {number}: {fault}")];
        }
    }


    /// <summary>Every column the declaration knows after any of its steps, with each kind it holds somewhere, in the order they come.</summary>
    /// <remarks>Followed from the schema down, as the reader follows them; worked out once, since the blocks it was read from do not change.</remarks>
    public IReadOnlyList<KnownColumn> ColumnsKnown => _known ??=
        [.. Enumerable.Range(1, Readable.Steps.Count)
            .SelectMany(steps => Readable.ColumnsBefore(steps).Columns)
            .Select(column => new KnownColumn(column.Name, column.Kind, Surely: true))
            .Distinct()];

    /// <summary>Reads the pipeline a notebook's cells declare, through the verbs a notebook knows.</summary>
    /// <param name="cells">The notebook's cells, in order; those that are not blocks are passed over.</param>
    /// <returns>The pipeline.</returns>
    public static NotebookPipeline Of(IEnumerable<CellModel> cells)
    {
        var catalog = NotebookVerbs.Catalog();
        var blocks = new List<Block>();
        var steps = new List<IPipelineStep>();
        CellModel[] read = [.. cells.Where(cell => cell.Type == StepCellType.StepType)];

        foreach (var cell in read)
        {
            try
            {
                var step = catalog.ReadStep(cell.Source);

                blocks.Add(new Block(cell.Id, step, []));

                // The rules are asked of the blocks above the first that does not read; one block that is not a
                // step would move every block below it and make faults of blocks that have none.
                if (steps.Count == blocks.Count - 1)
                {
                    steps.Add(step);
                }
            }
            catch (PipelineFileException refused)
            {
                blocks.Add(new Block(cell.Id, null, [.. refused.Faults.Select(fault => fault.ToString())]));
            }
        }

        var faults = PipelineDeclaration.FaultsIn(steps);

        foreach (var fault in faults)
        {
            blocks[fault.At] = blocks[fault.At] with { Faults = [.. blocks[fault.At].Faults, fault.ToString()] };
        }

        // Every rule holds for every beginning of a declaration that keeps it, so the blocks above the first one
        // at fault make a declaration.
        var count = faults.Count == 0 ? steps.Count : faults.Min(fault => fault.At);

        return new NotebookPipeline([.. blocks], read, new PipelineDeclaration(steps.Take(count)));
    }

    /// <summary>Where a block stands among the blocks.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <returns>Its place, counting from nought, or minus one when the cell is not a block.</returns>
    public int PositionOf(Guid cell) => Array.FindIndex(_blocks, block => block.Cell == cell);

    /// <summary>The key of every step the rows at a block are worked out from, when the block stands in the declaration.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <returns>The key, or nothing for a cell that is not a block of the declaration.</returns>
    public string? ViewKeyOf(Guid cell)
    {
        var position = PositionOf(cell);

        return position >= 0 && position < Readable.Steps.Count ? Readable.ViewKeyAt(position) : null;
    }

    /// <summary>What a block's kernel is to show: the data there, or why there is none.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <param name="trigger">What asks for it.</param>
    /// <param name="page">Which page of rows, counting from nought.</param>
    /// <returns>The request, saying what asked for it and whether the blocks make one pipeline.</returns>
    public ViewRequest RequestFor(Guid cell, ViewTrigger trigger, int page)
    {
        var position = PositionOf(cell);

        // The first block at or above one below the declaration that is at fault is what stops it.
        return position >= 0 && position < Readable.Steps.Count
            ? new ViewRequest(Readable, position, page, [], trigger, Whole, [])
            : new ViewRequest(null, position, page, Stopping, trigger, Whole, []);
    }
}
