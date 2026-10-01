// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Net;
using System.Text;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// What a block shows about its own step: the stage it belongs to, the verb, what the verb does, and the gesture
/// that shows the data there — and, on the schema's block, the one that lists the source's columns — or, when the block
/// is not one step, every fault at its line and column.
/// </summary>
/// <remarks>
/// Plain HTML with the theme's own colours inherited, so it sits in whatever theme the notebook has. Every word
/// that comes from a person or a file is encoded before it reaches the page.
/// </remarks>
internal static class StepCard
{
    private const string Style =
        "<style>.deepsharp-block{font-family:inherit;line-height:1.5}"
        + ".deepsharp-stage{display:inline-block;border:1px solid currentColor;border-radius:3px;padding:0 5px;margin-right:6px;font-size:.85em;opacity:.8}"
        + ".deepsharp-purpose{opacity:.8}"
        + ".deepsharp-faults{margin:.3em 0;padding-left:1.2em}"
        + ".deepsharp-actions{margin-top:.4em}</style>";

    /// <summary>The card of a block whose text is one step.</summary>
    /// <param name="step">The step.</param>
    /// <param name="purpose">What its verb does, in the words the catalog has for it.</param>
    /// <returns>The card.</returns>
    public static CellOutput Of(IPipelineStep step, string purpose)
    {
        var html = new StringBuilder(Style);

        html.Append("<div class=\"deepsharp-block\">")
            .Append("<div class=\"deepsharp-head\"><span class=\"deepsharp-stage\">").Append(Encoded(step.Stage())).Append("</span>")
            .Append("<code class=\"deepsharp-verb\">").Append(Encoded(step.Verb)).Append("</code></div>")
            .Append("<div class=\"deepsharp-purpose\">").Append(Encoded(purpose)).Append("</div>")
            .Append("<div class=\"deepsharp-actions\">");

        Button(html, StepRenderer.Show, "Show the data here");

        // The schema lists every column of the source, and takes each in or leaves it out.
        if (step is DeclareStep)
        {
            html.Append(' ');
            Button(html, StepRenderer.Columns, "Choose the columns");
        }

        html.Append("</div></div>");

        return CellOutput.Html(html.ToString());
    }

    /// <summary>The card of a block whose text is not one step a pipeline can read.</summary>
    /// <param name="faults">Every fault, each at its line and column in the block.</param>
    /// <returns>The card, marked as an error.</returns>
    public static CellOutput Refused(IEnumerable<PipelineFileFault> faults)
    {
        var html = new StringBuilder(Style);

        html.Append("<div class=\"deepsharp-block deepsharp-refused\">")
            .Append("<div class=\"deepsharp-head\">This block is not one step a pipeline can read.</div>")
            .Append("<ul class=\"deepsharp-faults\">");

        foreach (var fault in faults)
        {
            html.Append("<li><code>").Append(Encoded(string.Create(CultureInfo.InvariantCulture, $"({fault.Line},{fault.Column})"))).Append("</code> ")
                .Append(Encoded(fault.Message)).Append("</li>");
        }

        html.Append("</ul></div>");

        return new CellOutput("text/html", html.ToString(), IsError: true);
    }

    /// <summary>Why a block cannot show its data: the block that stops the pipeline above it, and what is wrong there.</summary>
    /// <param name="faults">What is wrong, each naming its block.</param>
    /// <returns>The explanation, marked as an error.</returns>
    public static CellOutput NoData(IEnumerable<string> faults) =>
        Listed("There is no data to show here: the blocks down to this one do not make a pipeline yet.", faults);

    /// <summary>Why a block cannot show its data though the blocks make a pipeline: what the rows met on the way.</summary>
    /// <param name="why">What stopped them, in the words of the file or the step that did.</param>
    /// <returns>The explanation, marked as an error.</returns>
    public static CellOutput RowsRefused(string why) =>
        Listed("There is no data to show here: the rows cannot be taken through the steps down to this one.", [why]);

    /// <summary>Why a change a gesture asked for is not made: every rule it would break, or what it needs that is not there.</summary>
    /// <param name="faults">Each reason: a rule at the step that would break it, or what is missing.</param>
    /// <returns>The explanation, marked as an error; the data below it is as it was.</returns>
    public static CellOutput NotMade(IEnumerable<string> faults) => Listed("This change is not made:", faults);

    /// <summary>
    /// What a report's block shows under its grid while nothing is handed back to it: the measures the report names, on which
    /// parts and shown how, and where they come from — a C# cell that trains a model, since the notebook trains none.
    /// </summary>
    /// <param name="report">The report.</param>
    /// <returns>The card, which is no error: nothing is wrong, and nothing is measured yet.</returns>
    public static CellOutput NotMeasured(INamesTheMeasures report)
    {
        var html = new StringBuilder(Style);

        html.Append("<div class=\"deepsharp-block deepsharp-report\"><div class=\"deepsharp-head\">This report measures ")
            .Append(Encoded(Phrase(report.Metrics.Select(metric => metric.Word())))).Append(" on ")
            .Append(Encoded(Phrase(report.Parts.Select(part => part.Word())))).Append(", shown as ")
            .Append(Encoded(Phrase(report.Shown.Select(shown => shown.Word())))).Append(".</div>")
            .Append("<div class=\"deepsharp-purpose\">It measures a trained model, and the notebook trains none. A C# cell that trains one ")
            .Append("hands its predictions back with <code>")
            .Append(Encoded($"Variables.Set(\"{StepKernel.HandedBack}\", trained.Measures!.PredictionsToJson())"))
            .Append("</code>; showing the data here then measures them on this notebook's own rows and draws them below. A cell that ")
            .Append("ends with <code>").Append(Encoded("trained.Measures!.Report()")).Append("</code> shows the same report where it stands.</div></div>");

        return CellOutput.Html(html.ToString());
    }

    /// <summary>Why the predictions handed back to a report's block are not measured: what refused them, and what to do.</summary>
    /// <param name="why">What refused them, in the words of the measuring, or of the text they were handed back as.</param>
    /// <returns>The explanation, marked as an error; the grid above it is as it was.</returns>
    public static CellOutput PredictionsRefused(string why) =>
        Listed(
            $"The predictions handed back under {StepKernel.HandedBack} are not measured here:",
            [why, "Run the C# cell that trains again: what it hands back is then made behind the blocks as they stand, and measured here."]);

    /// <summary>Why what the blocks decided was not saved: the file beside the notebook cannot be read.</summary>
    /// <param name="faults">What is wrong with the file.</param>
    /// <returns>The explanation, marked as an error; the file is as it was.</returns>
    public static CellOutput ColumnsUnreadable(IEnumerable<string> faults) =>
        Listed("The saved columns beside the notebook cannot be read, so they are not written over:", faults);

    /// <summary>Why what the blocks decided was not saved: the file beside the notebook cannot be written.</summary>
    /// <param name="why">What refused it.</param>
    /// <returns>The explanation, marked as an error; the blocks hold the decisions.</returns>
    public static CellOutput ColumnsNotWritten(string why) =>
        Listed("The saved columns could not be written beside the notebook:", [why]);

    /// <summary>A head and what it lists, marked as an error: the one way a block says what stopped something.</summary>
    /// <param name="head">What stopped.</param>
    /// <param name="items">Each reason.</param>
    /// <returns>The explanation.</returns>
    internal static CellOutput Listed(string head, IEnumerable<string> items)
    {
        var html = new StringBuilder(Style);

        html.Append("<div class=\"deepsharp-block deepsharp-refused\">")
            .Append("<div class=\"deepsharp-head\">").Append(Encoded(head)).Append("</div>")
            .Append("<ul class=\"deepsharp-faults\">");

        foreach (var item in items)
        {
            html.Append("<li>").Append(Encoded(item)).Append("</li>");
        }

        html.Append("</ul></div>");

        return new CellOutput("text/html", html.ToString(), IsError: true);
    }

    // A button carries its gesture and nothing else: the block it stands on is what it concerns.
    private static void Button(StringBuilder html, string gesture, string label) =>
        html.Append("<button type=\"button\" data-action=\"").Append(gesture)
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append("\">").Append(label).Append("</button>");

    private static string Encoded(string text) => WebUtility.HtmlEncode(text);

    // Words as a sentence lists them: "rmse", "rmse and mae", "rmse, mae and r2".
    private static string Phrase(IEnumerable<string> words)
    {
        string[] listed = [.. words];

        return listed.Length == 1 ? listed[0] : $"{string.Join(", ", listed[..^1])} and {listed[^1]}";
    }
}
