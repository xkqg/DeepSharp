// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Net;
using System.Text;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Notebooks.Verso;

/// <summary>
/// What a block shows about its own step: the stage it belongs to, the verb, what the verb does, and the gesture
/// that shows the data there — or, when the block is not one step, every fault at its line and column.
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
            .Append("<div class=\"deepsharp-actions\"><button type=\"button\" data-action=\"").Append(StepRenderer.Show)
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append("\">Show the data here</button></div>")
            .Append("</div>");

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

    /// <summary>Why a change a gesture asked for is not made: every rule it would break.</summary>
    /// <param name="faults">Each rule, at the step that would break it.</param>
    /// <returns>The explanation, marked as an error; the data below it is as it was.</returns>
    public static CellOutput NotMade(IEnumerable<string> faults) =>
        Listed("This change is not made: the pipeline would break.", faults);

    private static CellOutput Listed(string head, IEnumerable<string> items)
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

    private static string Encoded(string text) => WebUtility.HtmlEncode(text);
}
