// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>The kind, and for moments the format, a column the schema does not name is taken in with.</summary>
/// <param name="Kind">The kind.</param>
/// <param name="Format">How its moments are written, for a timestamp not written as ISO 8601; nothing otherwise.</param>
/// <param name="Proposed">What the column's cells propose, when that is what decided; nothing when the saved file did, or nothing did.</param>
/// <remarks>
/// One rule for every box that takes a column in — the list's and the grid's: as the file saved beside the notebook
/// declares it, else as its cells propose, else as the text the source holds. It is worked out where the box is drawn,
/// from the saved file and the rows that view read, and carried in the box, so a tick takes the column in as the box
/// said it would. A column the schema names is brought back as it was, whatever a box carries.
/// </remarks>
internal readonly record struct TakenIn(ColumnKind Kind, string? Format = null, ColumnProposal? Proposed = null)
{
    /// <summary>The kind a column is taken in with, by the one rule.</summary>
    /// <param name="column">The column.</param>
    /// <param name="stored">What the file beside the notebook holds, when it holds anything.</param>
    /// <param name="proposal">What the source's cells say each column holds, when its rows were read.</param>
    /// <returns>As the saved file declares it, else as its cells propose, else text.</returns>
    public static TakenIn Of(string column, PipelinePreset? stored, KindProposal? proposal)
    {
        if (stored?.Declare.Columns.FirstOrDefault(declared => declared.Name == column) is { } saved)
        {
            return new(saved.Kind, saved.Format);
        }

        return (proposal?.Columns ?? []).Where(proposed => proposed.Name == column).Select(proposed => new TakenIn(proposed.Kind, proposed.Format, proposed))
            .DefaultIfEmpty(new TakenIn(ColumnKind.Text))
            .First();
    }

    /// <summary>What a box carries to take a column in so.</summary>
    /// <param name="action">The box's action.</param>
    /// <returns>The kind and format it carries; text, for a box that carries none.</returns>
    public static TakenIn CarriedBy(ControlAction action) =>
        new(action.Text(StepRenderer.KindKey).AsKind() ?? ColumnKind.Text, action.Text(StepRenderer.FormatKey));

    /// <summary>Writes what a box carries to take a column in so.</summary>
    /// <param name="carried">The box's own values.</param>
    /// <returns>The same values, with the kind and, when there is one, the format.</returns>
    public JsonObject Into(JsonObject carried)
    {
        carried[StepRenderer.KindKey] = Kind.Word();

        if (Format is { } format)
        {
            carried[StepRenderer.FormatKey] = format;
        }

        return carried;
    }

    /// <summary>The steps with a column taken in as this says.</summary>
    /// <param name="declaration">The pipeline.</param>
    /// <param name="column">The column.</param>
    /// <param name="header">The source's columns, in order: a new column stands where the source has it.</param>
    /// <returns>
    /// The steps with the column taken in: a column the schema names brought back as it was, any other taken in with this
    /// kind, and read by this format when it holds moments written another way than ISO 8601.
    /// </returns>
    public IReadOnlyList<IPipelineStep> Into(PipelineDeclaration declaration, string column, IReadOnlyList<string> header)
    {
        var named = declaration.ChoicesFor([column]).Rows[0].Standing != ColumnStanding.NotDeclared;
        var steps = declaration.Including(column, Kind, header);

        return Format is { } format && !named && Kind == ColumnKind.Timestamp && !steps.SequenceEqual(declaration.Steps)
            ? new PipelineDeclaration(steps).WithFormat(column, format)
            : steps;
    }

    /// <summary>What the cells propose, in the words a list shows beside the column; nothing when they did not decide.</summary>
    /// <returns>The kind, with its format; that it is proposed; how many different values; what else is offered.</returns>
    public string? Said()
    {
        if (Proposed is not { } proposed)
        {
            return null;
        }

        var written = proposed.Format is { } format ? $" written as {format}" : string.Empty;
        var values = proposed.Distinct switch
        {
            null => "more than 64 values",
            1 => "1 value",
            { } count => string.Create(CultureInfo.InvariantCulture, $"{count} values"),
        };
        var offered = proposed.Offered switch
        {
            ColumnKind.Category => "; category offered",
            ColumnKind.Timestamp => $"; timestamp offered, written as {string.Join(" or ", proposed.Formats)}",
            _ => string.Empty,
        };

        return $"{proposed.Kind.Word()}{written}, proposed, {values}{offered}";
    }
}
