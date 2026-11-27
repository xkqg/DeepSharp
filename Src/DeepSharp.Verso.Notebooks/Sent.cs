// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>The row a list's control is about.</summary>
/// <param name="List">The gesture, at the list's block.</param>
/// <param name="Column">The row's column.</param>
/// <param name="Source">The rows this session read the source as, and their fingerprint.</param>
internal readonly record struct ListRow(Gesture List, string Column, SourceRows Source);

/// <summary>The list a control is on.</summary>
/// <param name="On">The gesture, at the list's block.</param>
/// <param name="Source">The rows this session read the source as, and their fingerprint.</param>
internal readonly record struct ListOn(Gesture On, SourceRows Source);

/// <summary>What one control sent: the gesture it made, the blocks as they stood, what Verso delivered it in, and what it carries.</summary>
/// <remarks>
/// What a control's gesture does is written beside it, as <see cref="SentExtensions"/>; what every one of them asks first —
/// whether the blocks make a pipeline, whether the list is still the one drawn — is asked here, once, and said at the block
/// when the answer is no.
/// </remarks>
/// <param name="Gesture">The gesture, on the block it was made on.</param>
/// <param name="Assembled">The pipeline the blocks made when it was sent.</param>
/// <param name="Context">What Verso delivered it in: what it sent, and where a change to the notebook is said.</param>
/// <param name="Action">The control's gesture and what it carries.</param>
internal readonly record struct Sent(Gesture Gesture, NotebookPipeline Assembled, CellInteractionContext Context, ControlAction Action)
{
    // Said at the list when a row is sent in a session that has not read the source the list was drawn from.
    private const string ListSourceNotRead = "Choose the columns again — the source is not read in this session.";

    /// <summary>What the list the control is on was drawn with: the kind of output its boxes make, and the picks.</summary>
    public ListPicks Picks => ListPicks.Of(Action);

    /// <summary>Whether the control was drawn over the blocks and the source's bytes as they are now.</summary>
    /// <param name="source">The rows this session read the source as, and their fingerprint.</param>
    /// <returns><see langword="true"/> while the list it is on still says what holds.</returns>
    public bool DrawnOver(SourceRows source) =>
        Action.Text(StepRenderer.DrawnKey) == NotebookSession.KeyOf(Assembled.Readable) && Action.Text(StepRenderer.SourceKey) == source.Fingerprint;

    /// <summary>Whether the blocks make one pipeline; when they do not, says at a block why, and the change is not made.</summary>
    /// <param name="at">The gesture, at the block that says it.</param>
    /// <param name="picks">The list to show under the refusal; nothing for the grid.</param>
    /// <returns><see langword="true"/> when the blocks make one pipeline.</returns>
    public async Task<bool> WholeOrRefuseAsync(Gesture at, ListPicks? picks)
    {
        if (Assembled.Whole)
        {
            return true;
        }

        await StepCommit.NotMadeAsync(at, Assembled, [$"the blocks do not make a pipeline yet: {string.Join("; ", Assembled.Stopping)}"], picks);

        return false;
    }

    /// <summary>
    /// Whether the control was drawn over the blocks and the source's bytes as they are now; when it was not, the list is
    /// drawn again and the change is not made.
    /// </summary>
    /// <param name="at">The gesture, at the list's block.</param>
    /// <param name="source">The rows this session read the source as, and their fingerprint.</param>
    /// <returns><see langword="true"/> while the list it is on still says what holds.</returns>
    public async Task<bool> ListOrRefreshAsync(Gesture at, SourceRows source)
    {
        if (DrawnOver(source))
        {
            return true;
        }

        await RefreshAsync(at);

        return false;
    }

    /// <summary>Draws the list again with the picks it was drawn with, forgetting where a range started.</summary>
    /// <param name="at">The gesture, at the list's block.</param>
    /// <returns>A task that ends when the block has run.</returns>
    public async Task RefreshAsync(Gesture at) => await StepCommit.ShowAsync(at, Assembled, ViewTrigger.Show, Picks.Afresh());

    /// <summary>The row the control is about: nothing when it names no column, or is on no list.</summary>
    /// <returns>The row.</returns>
    public async Task<ListRow?> RowAsync() =>
        Action.Text(StepRenderer.ColumnKey) is { } column && await ListAsync() is { } list ? new ListRow(list.On, column, list.Source) : null;

    /// <summary>
    /// The list the control is on: the schema's block the blocks hold now — found by what the control carries, whichever
    /// block the send names, since a change writes that block anew — with the rows this session read the source as.
    /// </summary>
    /// <returns>
    /// The list; nothing when the blocks hold no schema, and nothing — the list drawn again, saying why — in a session that has
    /// not read the source.
    /// </returns>
    public async Task<ListOn?> ListAsync()
    {
        if (Assembled.SchemaBlock is not { } schema)
        {
            return null;
        }

        var list = Gesture with { Cell = schema };

        if (Gesture.Session.Sources.KeptFor(Assembled.Readable) is not { } source)
        {
            await StepCommit.NotMadeAsync(list, Assembled, [ListSourceNotRead], ListPicks.None);

            return null;
        }

        return new ListOn(list, source);
    }
}
