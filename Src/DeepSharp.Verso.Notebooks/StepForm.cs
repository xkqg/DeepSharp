// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using System.Text.Json.Nodes;
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
/// A column is picked from the columns in scope at the block as the last gesture assembled the notebook: this part
/// is handed one cell and never assembles the notebook itself. A change makes stale what was worked out from the
/// block — its own card, and every view shown below it — and those are cleared; and since no gesture saw the change,
/// the pipeline handed to C# cells is withdrawn until one does.
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
    /// Written into the step's text only when the catalog reads the step back; the same value twice writes once.
    /// </remarks>
    public Task OnPropertyChangedAsync(CellModel cell, string propertyName, object? value, ICellRenderContext context)
    {
        ArgumentNullException.ThrowIfNull(cell);
        ArgumentNullException.ThrowIfNull(propertyName);
        ArgumentNullException.ThrowIfNull(context);

        var session = RequiredSession;
        var catalog = NotebookVerbs.Catalog();

        if (catalog.TryReadStep(cell.Source) is not { } step)
        {
            return Task.CompletedTask;
        }

        try
        {
            if (Edited(step, propertyName, FieldValue.Of(value), ScopeOf(session, cell.Id), catalog) is { } edited)
            {
                session.Accepted(cell.Id);

                if (!edited.Equals(step))
                {
                    Write(session, context.Variables, cell, edited, catalog);
                }
            }
        }
        catch (FormatException refused)
        {
            session.Refused(cell.Id, cell.Source, refused.Message);
        }

        return Task.CompletedTask;
    }

    // The step as the field changes it, read back through the catalog; nothing when the field is none of the step's.
    private static IPipelineStep? Edited(IPipelineStep step, string field, FieldValue value, FormScope scope, StepCatalog catalog)
    {
        var json = JsonNode.Parse(step.AsBlockText())!.AsObject();

        if (field == StepCatalog.StepKey)
        {
            return Switched(step, json, value.Text, catalog);
        }

        var edit = new FormEdit(field, value, json, scope, step);

        return catalog.Describe(step.Verb).Parameters.Any(parameter => parameter.Accept(edit))
            ? catalog.ReadStep(json.ToJsonString())
            : null;
    }

    // Another verb that acts as this one does — the capability D9 names, never the stage's name, which two
    // capabilities can share — keeping every value the other verb takes under the same key.
    private static IPipelineStep Switched(IPipelineStep step, JsonObject json, string? verb, StepCatalog catalog)
    {
        if (verb == step.Verb)
        {
            return step;
        }

        if (verb is null || !VerbsActingAsIt(step, catalog).Contains(verb, StringComparer.Ordinal))
        {
            throw new FormatException($"'{verb}' is not a step that does what '{step.Verb}' does.");
        }

        var other = JsonNode.Parse(catalog.Describe(verb).Template)!.AsObject();

        foreach (var key in catalog.Describe(verb).Keys.Where(json.ContainsKey))
        {
            other[key] = json[key]!.DeepClone();
        }

        return catalog.ReadStep(other.ToJsonString());
    }

    // Writes the step, and clears what was worked out from the block as it was: its own card, and every view shown
    // that the change made stale. No gesture saw the change, so what was handed to C# cells is withdrawn.
    private static void Write(NotebookSession session, IVariableStore variables, CellModel cell, IPipelineStep edited, StepCatalog catalog)
    {
        cell.Source = edited.AsBlockText();
        cell.Outputs.Clear();
        session.Hidden(cell.Id);

        if (session.Assembled is { } before)
        {
            // A view of a block deleted since is forgotten with the rest: there is nothing left to clear.
            foreach (var stale in session.ForgetStale(NotebookPipeline.Of(before.Cells), except: null))
            {
                before.Cells.FirstOrDefault(each => each.Id == stale)?.Outputs.Clear();
            }
        }

        NotebookSession.Withdraw(variables);
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
        var source = declaration.Steps.Count > 0 && declaration.Steps[0] is ReadCsvStep read ? session.Sources.ColumnNamesFor(read) : null;

        return new FormScope(columns, source);
    }

    private static PropertyField Verbs(IPipelineStep step, StepCatalog catalog) =>
        new(StepCatalog.StepKey, StepCatalog.StepKey, PropertyFieldType.Select, step.Verb, catalog.Describe(step.Verb).Purpose,
            [.. VerbsActingAsIt(step, catalog).Select(verb => new PropertyFieldOption(verb, verb))]);

    private static IEnumerable<string> VerbsActingAsIt(IPipelineStep step, StepCatalog catalog) =>
        catalog.Descriptions
            .Where(description => catalog.ReadStep(description.Template).ActingCapability() == step.ActingCapability())
            .Select(description => description.Verb);
}
