// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// What every button this package puts on the notebook's toolbar says in the same words, said once.
/// </summary>
/// <remarks>
/// A button is a part of the notebook like any other, so it is told who it is the way every part is, and what is left to
/// it is what makes it this button: its label, its place, its icon, when it can be pressed, and what pressing it does. The
/// rest is the same for all of them — the id the toolbar knows it by is the part's own, it shows an icon beside its label, it
/// is not the primary button, and it asks nothing before it acts — and so is the way a press begins: it is made on the
/// notebook's session, one gesture at a time. <see cref="RunOnSessionAsync"/> is that beginning.
/// </remarks>
public abstract class ToolbarButton : NotebookExtension, IToolbarAction
{
    /// <summary>Only this package's own buttons are toolbar buttons.</summary>
    private protected ToolbarButton()
    {
    }

    /// <inheritdoc />
    public string ActionId => ExtensionId;

    /// <inheritdoc />
    public abstract string DisplayName { get; }

    /// <inheritdoc />
    public string Icon => Drawing.Svg;

    /// <inheritdoc />
    public bool IconOnly => false;

    /// <inheritdoc />
    public bool IsPrimary => false;

    /// <inheritdoc />
    public string? ConfirmationPrompt => null;

    /// <inheritdoc />
    /// <remarks>The main toolbar, unless a button says it belongs in another place.</remarks>
    public virtual ToolbarPlacement Placement => ToolbarPlacement.MainToolbar;

    /// <inheritdoc />
    public abstract int Order { get; }

    /// <inheritdoc />
    public abstract Task<bool> IsEnabledAsync(IToolbarActionContext context);

    /// <inheritdoc />
    public abstract Task ExecuteAsync(IToolbarActionContext context);

    /// <summary>What the button's icon is drawn as.</summary>
    private protected abstract ToolbarIcon Drawing { get; }

    /// <summary>Makes a press on the notebook's session: after the gesture before it, and before the one after.</summary>
    /// <param name="context">What the part is told about the notebook.</param>
    /// <param name="gesture">What pressing does, given the session and its turn; whether it did anything.</param>
    /// <returns>A task that ends when the gesture has.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="context"/> is <see langword="null"/>.</exception>
    /// <exception cref="InvalidOperationException">
    /// Verso did not load this button beside the block type, so there is no notebook to press it on.
    /// </exception>
    private protected Task RunOnSessionAsync(IToolbarActionContext context, Func<NotebookSession, NotebookTurn, Task<bool>> gesture)
    {
        ArgumentNullException.ThrowIfNull(context);

        var session = RequiredSession;

        return session.OneAtATimeAsync(context.CancellationToken, turn => gesture(session, turn));
    }
}

/// <summary>The drawing on a toolbar button: one path, in the frame every button's icon is drawn in.</summary>
/// <param name="Path">The path's data, as the <c>d</c> of an SVG path in a square of twenty-four.</param>
internal readonly record struct ToolbarIcon(string Path)
{
    /// <summary>The icon as the toolbar shows it: sixteen pixels square, in the colour of the label beside it.</summary>
    public string Svg =>
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" width=\"16\" height=\"16\" fill=\"currentColor\">"
        + $"<path d=\"{Path}\"/></svg>";
}
