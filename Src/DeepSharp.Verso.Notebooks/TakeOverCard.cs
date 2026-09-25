// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// What taking over the columns saved beside a notebook would change, listed at a block before anything is changed.
/// </summary>
/// <remarks>
/// Every column whose decision would change, from how it stands to how it would stand; the output and the schema's order
/// when they would change; and the source's columns the saved file never showed. Under the list, one box that applies
/// it, carrying what it listed: the saved file as it was read, and the key of the blocks it was listed for — so what is
/// applied is what was shown, and only to the blocks it was shown for. A take-over whose blocks would break a rule is
/// listed with every rule and offers nothing to apply.
/// </remarks>
internal static class TakeOverCard
{
    private const string Style =
        "<style>.deepsharp-block{font-family:inherit;line-height:1.5}"
        + ".deepsharp-faults,.deepsharp-changes{margin:.3em 0;padding-left:1.2em}"
        + ".deepsharp-actions{margin-top:.4em}</style>";

    /// <summary>The list of a take-over.</summary>
    /// <param name="blocks">The declaration the blocks make, which the take-over was worked out against.</param>
    /// <param name="takenOver">What taking the saved columns over makes of it, and changes.</param>
    /// <param name="saved">The saved file's text, as it was read.</param>
    /// <returns>The list; marked as an error when the take-over is refused.</returns>
    public static CellOutput Of(PipelineDeclaration blocks, PresetTakeOver takenOver, string saved)
    {
        if (takenOver.Faults.Count > 0)
        {
            return StepCard.Listed(
                "The columns saved beside the notebook cannot be taken over: the blocks would then break these rules:",
                takenOver.Faults.Select(fault => fault.ToString()));
        }

        var html = new StringBuilder(Style).Append("<div class=\"deepsharp-block deepsharp-takeover\">");

        if (takenOver.Steps.SequenceEqual(blocks.Steps))
        {
            html.Append("<div class=\"deepsharp-head\">The blocks already hold every column decision saved beside the notebook.</div>");
            NewColumns(html, takenOver.NewColumns);

            return CellOutput.Html(html.Append("</div>").ToString());
        }

        var after = new PipelineDeclaration(takenOver.Steps);

        html.Append("<div class=\"deepsharp-head\">Taking over the columns saved beside the notebook changes:</div><ul class=\"deepsharp-changes\">");

        foreach (var change in takenOver.Changes)
        {
            html.Append("<li><code>").Append(Encoded(change.Column)).Append("</code>: ")
                .Append(Encoded(Stands(change.Before, blocks))).Append(" → ").Append(Encoded(Stands(change.After, after)))
                .Append(change.InSource == false ? " — not in the source" : string.Empty).Append("</li>");
        }

        if (takenOver.Output is { } output)
        {
            html.Append("<li>the output: ").Append(Quoted(output.Before)).Append(" → ").Append(Quoted(output.After)).Append("</li>");
        }

        if (takenOver.DeclareOrder is { } order)
        {
            html.Append("<li>the schema's order: ").Append(Encoded(string.Join(", ", order.Before))).Append(" → ")
                .Append(Encoded(string.Join(", ", order.After))).Append("</li>");
        }

        html.Append("</ul>");
        NewColumns(html, takenOver.NewColumns);

        // Drawn unticked; ticking it applies what is listed above, and the key says which blocks it was listed for.
        var apply = ControlAction.Of(StepRenderer.Apply, new JsonObject
        {
            [StepRenderer.PresetKey] = saved,
            [StepRenderer.DrawnKey] = NotebookSession.KeyOf(blocks),
        });

        html.Append("<div class=\"deepsharp-actions\"><label><input type=\"checkbox\" data-action=\"").Append(Encoded(apply))
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append("\"> Apply</label></div>");

        return CellOutput.Html(html.Append("</div>").ToString());
    }

    /// <summary>Why the columns saved beside a notebook cannot be taken over: the file cannot be read.</summary>
    /// <param name="faults">What is wrong with it, each at its line and column.</param>
    /// <returns>The explanation, marked as an error.</returns>
    public static CellOutput Unreadable(IEnumerable<string> faults) =>
        StepCard.Listed("The columns saved beside the notebook cannot be read, so there is nothing to take over:", faults);

    // How a column stands, in the words a person reads: a column a step makes is named by that step.
    private static string Stands(ColumnChoice column, PipelineDeclaration declaration) => column.Standing switch
    {
        ColumnStanding.Taking => $"taken, {Kind(column)}",
        ColumnStanding.Excluded => $"excluded, {Kind(column)}",
        ColumnStanding.Dropped => "dropped",
        ColumnStanding.Kept => "kept with the rest of the file",
        ColumnStanding.Made => $"made by step {column.MadeBy + 1}, '{declaration.Steps[column.MadeBy!.Value].Verb}'",
        _ => "not in the schema",
    };

    private static string Kind(ColumnChoice column) =>
        column is { Kind: ColumnKind.Category, Was: { } was } ? $"category (was {Word(was)})" : Word(column.Kind!.Value);

    private static string Word(ColumnKind kind) => kind.ToString().ToLowerInvariant();

    private static string Quoted(INamesTheAnswer? output) => output is null ? "none" : $"<code>{Encoded(output.AsLine())}</code>";

    private static void NewColumns(StringBuilder html, IReadOnlyList<string>? columns)
    {
        if (columns is { Count: > 0 })
        {
            html.Append("<div class=\"deepsharp-new\">New in the source: ").Append(Encoded(string.Join(", ", columns))).Append("</div>");
        }
    }

    private static string Encoded(string text) => WebUtility.HtmlEncode(text);
}
