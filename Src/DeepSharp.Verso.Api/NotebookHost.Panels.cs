// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso;
using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

// What the notebook's parts and its layout do: a click on a control a cell drew, a toolbar button pressed, the fields of
// a cell's properties and one of them set, what a layout hands back when a person acts in it, and the layout or theme
// the notebook is shown in. Each is a change, and a change runs the blocks as its own.
public sealed partial class NotebookHost
{
    /// <summary>Hands a click on a control a cell drew to the part the control names.</summary>
    /// <param name="gesture">The click.</param>
    /// <returns>Whether it changed the cells, and what the part answered, which the cell then shows.</returns>
    /// <exception cref="InvalidOperationException">No part of that name answers a click.</exception>
    /// <remarks>
    /// A page sends a click from the card it shows until it draws again, and a change may have rewritten the card's block
    /// meanwhile; the click is handed on all the same, as Verso's own editors hand every click on, because the part knows
    /// the block it became. What the part answers is shown only by a cell that still stands. A click is a change: it runs
    /// DeepSharp's blocks as its own, and anything else it asks to run is a run of its own, told and stopped as any run
    /// is; a click whose run was stopped ends as a stopped run ends, with nothing to answer.
    /// </remarks>
    public Task<GestureResult> GestureAsync(HostedGesture gesture) => TurnAsync(async () =>
    {
        var part = Extensions.GetInteractionHandler(gesture.ExtensionId)
            ?? throw new InvalidOperationException($"No part named '{gesture.ExtensionId}' answers a click.");
        var change = new ChangePort(this);
        var context = new CellInteractionContext
        {
            Region = CellRegion.Output,
            InteractionType = gesture.Action,
            Payload = gesture.Payload,
            CellId = gesture.Cell,
            ExtensionId = gesture.ExtensionId,
            CancellationToken = CancellationToken.None,
            Variables = Scaffold.Variables,
            Notebook = change,
            NotebookModel = Scaffold.Notebook,
        };
        var answer = await change.WholeAsync(() => part.OnCellInteractionAsync(context));

        if (answer is not null)
        {
            // As Verso's browser editor shows one: the answer is what the cell shows now, while the cell still stands.
            foreach (var shown in Scaffold.Cells.Where(each => each.Id == gesture.Cell))
            {
                shown.Outputs.Clear();
                shown.Outputs.Add(new CellOutput("text/html", answer));
            }
        }

        return new GestureResult(context.StateChanged, answer);
    });

    /// <summary>Presses a toolbar button.</summary>
    /// <param name="id">The button.</param>
    /// <param name="cells">The cells it is pressed for, for a button on a cell's toolbar.</param>
    /// <returns>The file it handed over, for whoever pressed it; nothing when it handed none.</returns>
    /// <exception cref="InvalidOperationException">The engine has no button of that name.</exception>
    /// <remarks>
    /// A button may run cells, C# among them, so a press takes its turn among the process's C# runs as a C# run does. A
    /// press is one run, and the button acts on the notebook through it, so once the run is stopped the notebook refuses
    /// what the button asks. A file is never written beside the notebook: where it is saved is for whoever pressed the
    /// button to say.
    /// </remarks>
    public Task<HostedFile?> RunToolbarAsync(string id, params Guid[] cells) => TurnAsync(async () =>
    {
        var action = Extensions.GetToolbarActions().FirstOrDefault(each => each.ActionId == id)
            ?? throw new InvalidOperationException($"The notebook has no toolbar button '{id}'.");
        var run = RunFor(null, runsCSharp: true);

        // The button acts on the notebook through its run, so a stop reaches every cell it would still run; one that runs a
        // cell, or code, that never ends is stopped as a cell's run is, and a stop starts afresh only what runs.
        var context = new ToolbarContext(Scaffold, cells, new RunPort(Scaffold, run));

        await RunUntilStoppedAsync(run, () => action.ExecuteAsync(context));

        // A stopped press hands nothing over: what it wrote after the stop is no one's.
        return run.Claimed ? null : context.Handed;
    });

    /// <summary>A cell's properties panel: a section from every part that has one for the cell, in their order.</summary>
    /// <param name="cell">The cell.</param>
    /// <returns>The sections.</returns>
    /// <exception cref="CellGoneException">The notebook's last version holds no such cell: a change rewrote it or took it away.</exception>
    /// <exception cref="ObjectDisposedException">The notebook was closed.</exception>
    /// <exception cref="InvalidOperationException">The layout the notebook is shown in has no properties panel.</exception>
    /// <remarks>Drawn beside whatever holds the notebook's turn, as Verso's editors draw the panel while a cell runs.</remarks>
    public Task<IReadOnlyList<HostedSection>> PropertiesAsync(Guid cell) => ReadAsync(cell, async shown =>
    {
        Scaffold.ThrowIfNoPanel();

        // Drawing a section is a look, and a look does nothing to the notebook.
        var context = new RenderContext(Scaffold, shown, new ReadPort(Scaffold));
        var sections = new List<HostedSection>();

        foreach (var part in Extensions.GetPropertyProviders().Where(each => each.AppliesTo(shown, context)).OrderBy(each => each.Order))
        {
            var section = await part.GetPropertiesSectionAsync(shown, context);

            sections.Add(new HostedSection(
                part.ExtensionId,
                section.Title,
                section.Description,
                [.. section.Fields.Select(field => new HostedField(
                    field.Name,
                    field.DisplayName,
                    Enum.Parse<FieldKind>(field.FieldType.ToString()),
                    field.CurrentValue,
                    field.Description,
                    [.. (field.Options ?? []).Select(option => new HostedOption(option.Value, option.DisplayName))],
                    field.IsReadOnly))]));
        }

        return (IReadOnlyList<HostedSection>)sections;
    });

    /// <summary>Changes a field of a cell's properties panel, through the part its section came from.</summary>
    /// <param name="cell">The cell.</param>
    /// <param name="part">The part the field's section came from.</param>
    /// <param name="field">The field.</param>
    /// <param name="value">What it is set to.</param>
    /// <returns>When the part has made the change.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <exception cref="InvalidOperationException">
    /// No part of that name has a properties section, or the layout the notebook is shown in has no properties panel.
    /// </exception>
    public Task SetPropertyAsync(Guid cell, string part, string field, object? value) => TurnAsync(async () =>
    {
        var changed = Standing(cell);

        Scaffold.ThrowIfNoPanel();
        var provider = Extensions.GetPropertyProviders().FirstOrDefault(each => each.ExtensionId == part)
            ?? throw new InvalidOperationException($"No part named '{part}' has a properties section.");

        var change = new ChangePort(this);

        return await change.WholeAsync(async () =>
        {
            await provider.OnPropertyChangedAsync(changed, field, value, new RenderContext(Scaffold, changed, change));

            return true;
        });
    });

    /// <summary>
    /// Hands what a person did to the arrangement the notebook's layout drew — a tile moved, resized or run — to the layout's
    /// own part, as Verso's editors hand it on. An act named in the host's own space is the host's: every version carries the
    /// arrangement as it is drawn, so asking for it again leaves nothing to do. An act the layout has no part for changes
    /// nothing.
    /// </summary>
    /// <param name="interaction">The act.</param>
    /// <returns>The file the part handed over, for whoever acted; nothing when it handed none.</returns>
    /// <exception cref="InvalidOperationException">
    /// The notebook is no longer shown in the layout that drew what was acted on; nothing is done.
    /// </exception>
    /// <remarks>
    /// An act is a change: it runs DeepSharp's blocks as its own, and anything else it asks to run — a tile's C# cell — is a
    /// run of its own, told and stopped as any run is, so a tile moved or resized runs nothing and waits for no C# run. A
    /// layout may add, take away or move cells through the notebook's operations, as Verso's notebook layout does, so the
    /// notebook is told its cells changed after every act.
    /// </remarks>
    public Task<HostedFile?> InteractAsync(HostedLayoutInteraction interaction) => TurnAsync(async () =>
    {
        var layout = Scaffold.LayoutManager?.ActiveLayout;

        if (layout is null || !layout.LayoutId.IsNamed(interaction.Layout))
        {
            throw new InvalidOperationException($"The notebook is no longer shown in the layout '{interaction.Layout}'.");
        }

        var extension = (layout as IExtension)?.ExtensionId ?? string.Empty;

        if (interaction.Action.StartsWith(HostsOwn, StringComparison.Ordinal) || !Extensions.TryGetLayoutInteractionHandler(extension, layout.LayoutId, out var part))
        {
            return null;
        }

        var change = new ChangePort(this);
        var context = new ToolbarContext(Scaffold, [], change);

        await change.WholeAsync(async () =>
        {
            // What the part asks to draw again is drawn with the next version, whatever it asks for.
            await part.OnLayoutInteractionAsync(new LayoutInteractionContext
            {
                ExtensionId = extension,
                LayoutId = layout.LayoutId,
                InteractionType = interaction.Action,
                Payload = interaction.Payload,
                TargetId = interaction.Target,
                Verso = context,
            });

            return true;
        });

        await TellAsync();

        // A stopped act hands nothing over, as a stopped press hands nothing over.
        return change.Stopped ? null : context.Handed;
    });

    /// <summary>
    /// Shows the notebook in another layout, as Verso's View panel switches it: through the engine's own operations, which
    /// name the layout in the notebook, so it is saved as the notebook's choice.
    /// </summary>
    /// <param name="id">One of <see cref="Layouts"/>.</param>
    /// <returns>When it is shown in it.</returns>
    /// <exception cref="InvalidOperationException">The engine has no layout of that id; nothing changes.</exception>
    public Task SwitchLayoutAsync(string id)
    {
        ArgumentNullException.ThrowIfNull(id);

        return TurnAsync(() =>
        {
            var layout = Layouts.FirstOrDefault(each => each.Id.IsNamed(id));

            if (layout.Id is null)
            {
                throw new InvalidOperationException($"The notebook has no layout '{id}' to be shown in.");
            }

            Scaffold.NotebookOps.SetActiveLayout(layout.Id);

            return Task.FromResult(true);
        });
    }

    /// <summary>
    /// Draws the notebook in a theme, as Verso's View panel switches it: an explicit choice, which the notebook names from
    /// then on and saves as its own. Only such a choice names one; until then the notebook is drawn in a view's own look.
    /// </summary>
    /// <param name="id">One of <see cref="Themes"/>.</param>
    /// <returns>When it is drawn in it.</returns>
    /// <exception cref="InvalidOperationException">The engine has no theme of that id; nothing changes.</exception>
    public Task SwitchThemeAsync(string id)
    {
        ArgumentNullException.ThrowIfNull(id);

        return TurnAsync(() =>
        {
            var theme = Themes.FirstOrDefault(each => each.Id.IsNamed(id));

            if (theme.Id is null)
            {
                throw new InvalidOperationException($"The notebook has no theme '{id}' to be drawn in.");
            }

            Scaffold.NotebookOps.SetActiveTheme(theme.Id);

            return Task.FromResult(true);
        });
    }
}
