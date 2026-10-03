// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso;
using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

// What a person does to the notebook's cells: typing in one, adding, removing and moving them, turning one into
// another kind, and what a kernel offers while they are typed in. Every one of them takes the notebook's turn, and
// the next version says what changed.
public sealed partial class NotebookHost
{
    /// <summary>
    /// Sets a cell's text, as typing it does; nothing runs, and the notebook is told at once, so what was worked out from
    /// a block as it was — a grid, the pipeline handed to C# cells — is taken back before anything else is asked of it.
    /// </summary>
    /// <param name="cell">The cell.</param>
    /// <param name="source">Its new text.</param>
    /// <returns>The cell as it stands after.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <exception cref="LayoutCapabilityException">The layout the notebook is shown in does not let a cell's text be changed.</exception>
    public Task<HostedCell> EditAsync(Guid cell, string source)
    {
        ArgumentNullException.ThrowIfNull(source);

        return TurnAsync(async () =>
        {
            Standing(cell);
            Scaffold.ThrowIfNoEdit();
            Scaffold.UpdateCellSource(cell, source);

            // The notebook is told at once: what was worked out from the block as it was is taken back.
            await TellAsync();

            return (await PublishedAsync()).Cells.First(each => each.Id == cell);
        });
    }

    /// <summary>Adds a cell of a kind right after another, as the add button between two cells does; it starts empty.</summary>
    /// <param name="after">The cell it follows.</param>
    /// <param name="kind">
    /// One of <see cref="Kinds"/>; one named with no language is given the language Verso's editors give it — code in the
    /// notebook's default kernel, else C#.
    /// </param>
    /// <returns>The new cell.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell it follows or took it away.</exception>
    /// <exception cref="InvalidOperationException">
    /// The notebook lists no such kind, or the layout it is shown in does not let a cell be added
    /// (<see cref="LayoutCapabilityException"/>).
    /// </exception>
    public Task<HostedCell> InsertAsync(Guid after, HostedKind kind) => TurnAsync(() =>
        AddedAsync(Scaffold.Notebook.Cells.IndexOf(Standing(after)) + 1, kind));

    /// <summary>Adds a cell of a kind at the end, as the add button under the last cell does; it starts empty.</summary>
    /// <param name="kind">
    /// One of <see cref="Kinds"/>; one named with no language is given the language Verso's editors give it — code in the
    /// notebook's default kernel, else C#.
    /// </param>
    /// <returns>The new cell.</returns>
    /// <exception cref="InvalidOperationException">
    /// The notebook lists no such kind, or the layout it is shown in does not let a cell be added
    /// (<see cref="LayoutCapabilityException"/>).
    /// </exception>
    public Task<HostedCell> AddAsync(HostedKind kind) => TurnAsync(() => AddedAsync(Scaffold.Cells.Count, kind));

    /// <summary>Takes a cell away.</summary>
    /// <param name="cell">The cell.</param>
    /// <returns>When it is gone.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <exception cref="LayoutCapabilityException">The layout the notebook is shown in does not let a cell be taken away.</exception>
    public Task RemoveAsync(Guid cell) => TurnAsync(async () =>
    {
        Standing(cell);
        await Scaffold.NotebookOps.RemoveCellAsync(cell);
        await TellAsync();

        return true;
    });

    /// <summary>Moves a cell up past its neighbour, to stand right before it.</summary>
    /// <param name="cell">The cell.</param>
    /// <param name="neighbour">The cell it passes: the one above it, as the page showed them.</param>
    /// <returns>When it is moved.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote either cell or took it away.</exception>
    /// <exception cref="LayoutCapabilityException">The layout the notebook is shown in does not let a cell be moved.</exception>
    /// <remarks>
    /// Named by the neighbour it passes rather than by a place, since another view may have moved cells meanwhile, and a
    /// place counted there is somewhere else here. A cell named as its own neighbour passes nothing.
    /// </remarks>
    public Task MoveBeforeAsync(Guid cell, Guid neighbour) => MoveAsync(cell, neighbour, after: false);

    /// <summary>Moves a cell down past its neighbour, to stand right after it.</summary>
    /// <param name="cell">The cell.</param>
    /// <param name="neighbour">The cell it passes: the one below it, as the page showed them.</param>
    /// <returns>When it is moved.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote either cell or took it away.</exception>
    /// <exception cref="LayoutCapabilityException">The layout the notebook is shown in does not let a cell be moved.</exception>
    /// <remarks>
    /// Named by the neighbour it passes rather than by a place, since another view may have moved cells meanwhile, and a
    /// place counted there is somewhere else here. A cell named as its own neighbour passes nothing.
    /// </remarks>
    public Task MoveAfterAsync(Guid cell, Guid neighbour) => MoveAsync(cell, neighbour, after: true);

    /// <summary>
    /// Turns a cell into another kind in one step, as Verso's editors do: its type and language change, its text stays,
    /// and what it showed is cleared. The same kind again changes nothing.
    /// </summary>
    /// <param name="cell">The cell.</param>
    /// <param name="kind">
    /// One of <see cref="Kinds"/>; one named with no language is given the language Verso's editors give it — code in the
    /// notebook's default kernel, else C#.
    /// </param>
    /// <returns>The cell as it stands after.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <exception cref="InvalidOperationException">
    /// The notebook lists no such kind, or the layout it is shown in does not let a cell be changed
    /// (<see cref="LayoutCapabilityException"/>).
    /// </exception>
    public Task<HostedCell> ChangeKindAsync(Guid cell, HostedKind kind) => TurnAsync(async () =>
    {
        var listed = Listed(kind);
        var changed = Standing(cell);

        Scaffold.ThrowIfNoEdit();

        if (!changed.Type.IsNamed(listed.Type) || !changed.Language.IsNamed(listed.Language))
        {
            changed.Type = listed.Type;
            changed.Language = listed.Language;
            changed.Outputs.Clear();
            Scaffold.OutputChannels.CloseForCell(cell, "the cell's outputs were cleared");
            await TellAsync();
        }

        return (await PublishedAsync()).Cells.First(each => each.Id == cell);
    });

    /// <summary>
    /// What a cell's kernel offers to write next where the cursor stands, as Verso's editors ask it while a person types.
    /// </summary>
    /// <param name="cell">The cell.</param>
    /// <param name="code">Its text as the person has it, which may be ahead of what was sent.</param>
    /// <param name="position">Where the cursor stands in it, counted in characters.</param>
    /// <returns>
    /// What is offered; nothing for a cell whose text no kernel reads, and nothing when the kernel was started afresh while it
    /// was asked, whether it answered or failed as it was put away.
    /// </returns>
    /// <exception cref="CellGoneException">The notebook's last version holds no such cell: a change rewrote it or took it away.</exception>
    /// <exception cref="ObjectDisposedException">The notebook was closed.</exception>
    /// <remarks>
    /// Answered beside whatever holds the notebook's turn, as Verso's editors ask while a cell runs. The kernel is started
    /// first when it has not been, as Verso's editors start it, rather than in the background, so the first thing offered
    /// may take the time a kernel takes to start.
    /// </remarks>
    public Task<IReadOnlyList<HostedCompletion>> CompletionsAsync(Guid cell, string code, int position)
    {
        ArgumentNullException.ThrowIfNull(code);

        return ReadAsync(cell, standing => KernelReadAsync(async () =>
        {
            if (await Scaffold.KernelForAsync(standing) is not { } kernel)
            {
                return (IReadOnlyList<HostedCompletion>)[];
            }

            return [.. (await kernel.GetCompletionsAsync(code, position))
                .Select(offered => new HostedCompletion(offered.DisplayText, offered.InsertText, offered.Kind, offered.Description, offered.SortText))];
        }, nothing: []));
    }

    /// <summary>What a word in a cell's text means, as Verso's editors ask it when the cursor rests on the word.</summary>
    /// <param name="cell">The cell.</param>
    /// <param name="code">Its text as the person has it.</param>
    /// <param name="position">Where the cursor rests in it, counted in characters.</param>
    /// <returns>
    /// What the word means; nothing where the kernel says nothing, for a cell whose text no kernel reads, or when the kernel
    /// was started afresh while it was asked, whether it answered or failed as it was put away.
    /// </returns>
    /// <exception cref="CellGoneException">The notebook's last version holds no such cell: a change rewrote it or took it away.</exception>
    /// <exception cref="ObjectDisposedException">The notebook was closed.</exception>
    /// <remarks>Answered beside whatever holds the notebook's turn, as Verso's editors ask while a cell runs.</remarks>
    public Task<HostedHover?> HoverAsync(Guid cell, string code, int position)
    {
        ArgumentNullException.ThrowIfNull(code);

        return ReadAsync(cell, standing => KernelReadAsync(async () =>
        {
            if (await Scaffold.KernelForAsync(standing) is not { } kernel || await kernel.GetHoverInfoAsync(code, position) is not { } said)
            {
                return (HostedHover?)null;
            }

            return new HostedHover(
                said.Content,
                said.MimeType,
                said.Range is { } range ? new HostedRange(range.StartLine, range.StartColumn, range.EndLine, range.EndColumn) : null);
        }, nothing: null));
    }

    /// <summary>
    /// Titles the notebook, as Verso's Metadata panel does: the title names what it is exported as. The same title again
    /// changes nothing.
    /// </summary>
    /// <param name="title">The title; nothing for none.</param>
    /// <returns>When it is titled.</returns>
    public Task RetitleAsync(string? title) => TurnAsync(() =>
    {
        Scaffold.Title = title;

        return Task.FromResult(true);
    });

    private CellModel Standing(Guid cell) => Scaffold.GetCell(cell) ?? throw new CellGoneException(cell, Current.Version);

    // A kind the notebook lists, as it lists it. A kind named with no language is given the one Verso's editors give it: its
    // cell type's kernel's; none for a type a renderer draws; else the notebook's default kernel, else C#.
    private HostedKind Listed(HostedKind kind) => Kinds.Listed(kind, kind.Language ?? Extensions.LanguageOf(Scaffold, kind.Type));

    // A cell of a listed kind added at a place, empty, through the port the notebook's layout guards; the notebook is told.
    private async Task<HostedCell> AddedAsync(int at, HostedKind kind)
    {
        var listed = Listed(kind);
        var added = Guid.Parse(await Scaffold.NotebookOps.InsertCellAsync(at, listed.Type, listed.Language));

        await TellAsync();

        return (await PublishedAsync()).Cells.First(each => each.Id == added);
    }

    // A cell moved to stand right before or right after its neighbour, wherever either stands now.
    private Task MoveAsync(Guid cell, Guid neighbour, bool after) => TurnAsync(async () =>
    {
        var cells = Scaffold.Notebook.Cells;
        var from = cells.IndexOf(Standing(cell));
        var to = cells.IndexOf(Standing(neighbour));

        if (from != to)
        {
            await Scaffold.NotebookOps.MoveCellAsync(cell, to - (from < to ? 1 : 0) + (after ? 1 : 0));
            await TellAsync();
        }

        return true;
    });

    // Tells the notebook its cells changed, so what was worked out from the blocks as they were is taken back.
    private Task TellAsync() => _blocks.BlocksChangedAsync(Scaffold.Notebook, Scaffold.Variables, new ChangePort(this));
}
