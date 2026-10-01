// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// The form face of a block: Verso's properties panel, one field for every value the step takes.
/// </summary>
/// <remarks>
/// The fields are read from the step as it writes itself, and a change is written into the step's JSON and read
/// back through the catalog a pipeline file is read with, so the step's own rules hold and a value it refuses is
/// never written: the text is left as it was, and the form says why until the block changes. The block's text is
/// the one place a step lives, so the form writes it and never the cell's metadata, where a number would come back
/// as something else. It never throws, because Verso shows an empty panel for a part that does.
/// <para>
/// A column is picked from the columns in scope at the block as the notebook was last read: this part is handed one
/// cell, and reaches the others only through the notebook the last gesture read. A change makes stale what was worked
/// out from the block — its own card, and every view shown below it — and those are cleared; and since no gesture saw
/// the change, it goes through the rule every such change does, which takes back the pipeline handed to C# cells.
/// </para>
/// </remarks>
[VersoExtension]
public sealed class StepForm : NotebookExtension, ICellPropertyProvider
{
    /// <summary>The form's id.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.form";

    private const string Title = "Pipeline step";

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp pipeline step form";

    /// <inheritdoc />
    public override string Description => "Edits the step a block holds in the properties panel, one field for each value it takes.";

    /// <inheritdoc />
    /// <remarks>First: the step is what the block is.</remarks>
    public int Order => 0;


    /// <inheritdoc />
    public bool AppliesTo(CellModel cell, ICellRenderContext context)
    {
        ArgumentNullException.ThrowIfNull(cell);

        return cell.Type == StepCellType.StepType;
    }

    /// <inheritdoc />
    /// <remarks>A block whose text is not one step yet shows why, rather than an empty panel.</remarks>
    public Task<PropertySection> GetPropertiesSectionAsync(CellModel cell, ICellRenderContext context)
    {
        ArgumentNullException.ThrowIfNull(cell);

        var catalog = NotebookVerbs.Catalog();
        IPipelineStep step;

        try
        {
            step = catalog.ReadStep(cell.Source);
        }
        catch (PipelineFileException refused)
        {
            return Task.FromResult(new PropertySection(
                Title, $"This block is not one step a pipeline can read yet: {string.Join("; ", refused.Faults)}", []));
        }

        // A form Verso never loaded reads the step as well as any other; it knows no columns and refused nothing.
        using var written = JsonDocument.Parse(step.AsBlockText());
        var session = LoadedSession;
        var description = catalog.Describe(step.Verb);
        var visitor = new FormFields(written.RootElement, ScopeOf(session, cell.Id));
        List<PropertyField> fields = [Verbs(step, catalog), .. description.Parameters.SelectMany(parameter => parameter.Accept(visitor))];
        var notMade = session?.RefusalFor(cell.Id, cell.Source);

        return Task.FromResult(new PropertySection(
            Title, notMade is null ? description.Purpose : $"{description.Purpose} The last change was not made: {notMade}", fields));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Written into the step's text only when the catalog reads the step back; the same value twice writes once. A
    /// change is one change to the notebook, so it waits for a gesture already on it, and is made only on a block that
    /// still stands then: one the gesture rewrote or took away is left alone, as a host that looks the block up by its id
    /// and finds none leaves it.
    /// </remarks>
    public Task OnPropertyChangedAsync(CellModel cell, string propertyName, object? value, ICellRenderContext context)
    {
        ArgumentNullException.ThrowIfNull(cell);
        ArgumentNullException.ThrowIfNull(propertyName);
        ArgumentNullException.ThrowIfNull(context);

        var session = RequiredSession;

        return session.OneAtATimeAsync(
            context.CancellationToken, turn => Task.FromResult(new PropertyChange(session, turn, cell, context.Variables).Made(propertyName, value)));
    }

    // The step as the field changes it, read back through the catalog; nothing when the field is none of the step's.
    internal static IPipelineStep? Edited(FormEdit edit, StepCatalog catalog)
    {
        var step = edit.Read;

        if (edit.Field == StepCatalog.StepKey)
        {
            return Switched(step, edit.Value.Text, catalog);
        }

        return catalog.Describe(step.Verb).Parameters.Any(parameter => parameter.Accept(edit))
            ? catalog.ReadStep(edit.Json.ToJsonString())
            : null;
    }

    // Another verb that acts as this one does — the capability D9 names, never the stage's name, which two
    // capabilities can share — made by the catalog, keeping every value the other verb takes under the same key.
    private static IPipelineStep Switched(IPipelineStep step, string? verb, StepCatalog catalog)
    {
        if (verb == step.Verb)
        {
            return step;
        }

        if (verb is null || !VerbsActingAsIt(step, catalog).Contains(verb, StringComparer.Ordinal))
        {
            throw new FormatException($"'{verb}' is not a step that does what '{step.Verb}' does.");
        }

        return catalog.Make(verb, [], step);
    }

    // What the form knows around a block: the columns before it, when the last gesture assembled it into the
    // pipeline, and the columns the source has, when its rows were read.
    private static FormScope ScopeOf(NotebookSession? session, Guid cell)
    {
        if (session?.Assembled is not { } assembled)
        {
            return FormScope.Unknown;
        }

        var position = assembled.PositionOf(cell);
        var declaration = assembled.Readable;
        var columns = position >= 0 && position < declaration.Steps.Count ? declaration.ColumnsBefore(position).Columns : null;
        var source = session.Sources.KeptFor(declaration)?.Rows.ColumnNames;

        return new FormScope(columns, source);
    }

    private static PropertyField Verbs(IPipelineStep step, StepCatalog catalog) =>
        new(StepCatalog.StepKey, StepCatalog.StepKey, PropertyFieldType.Select, step.Verb, catalog.Describe(step.Verb).Purpose,
            [.. VerbsActingAsIt(step, catalog).Select(verb => new PropertyFieldOption(verb, verb))]);

    private static IEnumerable<string> VerbsActingAsIt(IPipelineStep step, StepCatalog catalog) =>
        catalog.Descriptions
            .Where(description => catalog.ReadStep(description.Template).ActingCapability() == step.ActingCapability())
            .Select(description => description.Verb);

    /// <summary>A change made in one block's panel, in its turn on the notebook.</summary>
    /// <param name="Session">The notebook's session.</param>
    /// <param name="Turn">The change's turn: what it writes beyond the block is let through for it.</param>
    /// <param name="Cell">The block.</param>
    /// <param name="Variables">The values the notebook's cells share, from which what was handed to C# cells is taken back.</param>
    private readonly record struct PropertyChange(NotebookSession Session, NotebookTurn Turn, CellModel Cell, IVariableStore Variables)
    {
        /// <summary>Makes the change on a block that stands.</summary>
        /// <param name="field">The field that changed.</param>
        /// <param name="value">What it was set to.</param>
        /// <returns>Whether the field was one of the step's.</returns>
        public bool Made(string field, object? value)
        {
            var catalog = NotebookVerbs.Catalog();

            if (!Session.Stands(Cell.Id) || catalog.TryReadStep(Cell.Source) is not { } step)
            {
                return false;
            }

            try
            {
                if (Edited(new FormEdit(field, FieldValue.Of(value), ScopeOf(Session, Cell.Id), step), catalog) is not { } edited)
                {
                    return false;
                }

                Session.Accepted(Cell.Id);

                if (!edited.Equals(step))
                {
                    Write(edited);
                }
            }
            catch (FormatException refused)
            {
                Session.Refused(Cell.Id, Cell.Source, refused.Message);
            }

            return true;
        }

        // Writes the step, and clears what was worked out from the block as it was: its own card, and every view shown
        // that the change made stale. No gesture saw the change, so it goes through the rule every such change does, which
        // takes back what was handed to C# cells. The edit is the person's and stands whatever else happens; what it made
        // stale elsewhere is caught up with as one write for the change's turn, each view cleared before it is forgotten.
        private void Write(IPipelineStep edited)
        {
            var session = Session;
            var variables = Variables;

            Cell.Source = edited.AsBlockText();
            Cell.Outputs.Clear();
            session.Hidden(Cell.Id);

            session.LetThrough(Turn, () =>
            {
                if (session.Assembled is { } before)
                {
                    // A view of a block deleted since, or of a cell turned into another kind, is forgotten with the rest, and
                    // nothing of such a cell's is cleared.
                    var now = NotebookPipeline.Of(before.Cells);

                    foreach (var stale in session.StaleIn(now, except: null))
                    {
                        before.Cells.First(each => each.Id == stale).Outputs.Clear();
                    }

                    session.CaughtUp(now, variables, except: null);
                }
                else
                {
                    NotebookSession.Withdraw(variables);
                }
            });
        }
    }
}
