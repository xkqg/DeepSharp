// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// Every column of the source as one row — its name, its first values, whether it is in, and its kind — so nothing a
/// person leaves out ever disappears from sight.
/// </summary>
/// <remarks>
/// The rows are the source's columns in its order, then every column the schema names that the source lacks. A row's
/// box is the grid's box, asked of the same column rules; it carries its column, the kind the row shows — the one the
/// schema declares, else the one a tick takes it in with by the one rule every box keeps, which a column's cells decide
/// only where the saved file does not, and says so beside the kind with what the cells hold — and what the list was
/// drawn from: the key of the blocks and the fingerprint of the source's bytes. Everything that comes from a person or a
/// file is encoded before it reaches the page.
/// <para>
/// What the list lists is worked out once, from the declaration, the source's rows and the file beside the notebook; the
/// picks a person made and whether every block is in the declaration only say how it is drawn.
/// </para>
/// </remarks>
internal sealed class ColumnList
{
    /// <summary>How many of a column's first values a row shows.</summary>
    public const int ValuesShown = 10;

    private const string Style =
        "<style>.deepsharp-list{margin-top:.4em}"
        + ".deepsharp-list table{border-collapse:collapse}"
        + ".deepsharp-list th,.deepsharp-list td{padding:1px 6px;border:1px solid rgba(128,128,128,.25);white-space:nowrap}"
        + ".deepsharp-list td.deepsharp-values{opacity:.8;max-width:32em;overflow:hidden;text-overflow:ellipsis}</style>";

    // Written for a value the source left blank, so a gap reads as one.
    private const string Blank = "∅";

    private readonly StepCatalog _catalog;
    private readonly PipelineDeclaration _declaration;
    private readonly SourceRows _source;
    private readonly IReadOnlyList<string> _header;
    private readonly IReadOnlyList<ColumnDeclaration> _declared;
    private readonly IReadOnlyList<string> _fresh;
    private readonly IReadOnlyList<IReadOnlyList<string?>> _first;
    private readonly IReadOnlyList<ColumnChoice> _rows;
    private readonly Dictionary<string, TakenIn> _taking;
    private readonly Dictionary<string, ColumnKind> _kinds;
    private readonly IReadOnlyList<string> _verbs;
    private readonly Dictionary<string, string?> _stopped;

    /// <summary>The list of a source's columns under a declaration.</summary>
    /// <param name="catalog">The verbs the notebook knows.</param>
    /// <param name="declaration">The declaration the blocks make.</param>
    /// <param name="source">The rows as the source reads them, and the fingerprint of their bytes.</param>
    /// <param name="stored">
    /// What the file beside the notebook holds, when it holds anything: how a column it declares is taken in, and which of the
    /// source's columns it never showed, which the list marks new.
    /// </param>
    public ColumnList(StepCatalog catalog, PipelineDeclaration declaration, SourceRows source, PipelinePreset? stored)
    {
        _catalog = catalog;
        _declaration = declaration;
        _source = source;
        _header = source.Rows.ColumnNames;
        _declared = declaration.Steps.OfType<DeclareStep>().FirstOrDefault()?.Columns ?? [];
        _fresh = stored is { } preset ? preset.NewColumns(_header) : [];

        var header = _header;
        IReadOnlyList<string> columns = [.. header.Concat(_declared.Select(column => column.Name).Where(name => !header.Contains(name, StringComparer.Ordinal)))];

        _first = [.. source.Rows.Rows.Take(ValuesShown)];
        _rows = declaration.ChoicesFor(columns).Rows;
        _taking = _rows.ToDictionary(choice => choice.Name, choice => TakenIn.Of(choice.Name, stored, source.Proposal));
        _kinds = _rows.ToDictionary(choice => choice.Name, choice => choice.Kind ?? _taking[choice.Name].Kind);
        _verbs = OutputBox.Verbs(catalog);
        _stopped = _verbs.ToDictionary(each => each, Stopped);
    }

    /// <summary>The list, drawn with the picks a person made on it.</summary>
    /// <param name="picks">
    /// The picks the list is drawn with: the kind of output its boxes make, ticking one column or a range, where a range
    /// starts, and the kind a range takes its columns in with.
    /// </param>
    /// <param name="whole">Whether every block is in the declaration: an output is changed only then.</param>
    /// <returns>The list, every control carrying the key of the declaration it is drawn for.</returns>
    public CellOutput Drawn(ListPicks picks, bool whole)
    {
        // The kind of output the boxes make: the one picked, else the output's own, else the first a row can take.
        var verb = picks.Type is { } picked && _verbs.Contains(picked, StringComparer.Ordinal) ? picked
            : _declaration.Output?.Verb ?? _verbs.FirstOrDefault(each => _stopped[each] is null) ?? _verbs[0];

        // A range of an output of many takes the kinds its answer reads; an output of one column is ticked one at a time.
        var reads = picks.Range ? OutputBox.RangeKinds(_catalog, verb) : null;
        var list = new ListDrawing(verb, NotebookSession.KeyOf(_declaration), _source.Fingerprint, picks, whole);
        var html = new StringBuilder(Style).Append("<div class=\"deepsharp-list\">")
            .Append("<div class=\"deepsharp-head\">The source's columns, each as the pipeline takes it:</div>");

        Picks(html, list);
        Output(html, list, reads);

        // The output's own values, while the list makes the kind of output that stands.
        if (_declaration.Output is { } output && output.Verb == verb)
        {
            Parameters(html, new OutputParameters(_catalog, _declaration, _header).Of(output), list);
        }

        html.Append("<table><thead><tr><th>column</th><th>first values</th><th>in</th><th>kind</th><th>output</th><th></th></tr></thead><tbody>");

        foreach (var choice in _rows)
        {
            var at = _header.PlaceOf(choice.Name);
            var mark = string.Join(
                "; ",
                new[]
                {
                    at < 0 ? "not in the source" : _fresh.Contains(choice.Name, StringComparer.Ordinal) ? "new" : null,
                    choice.Name == picks.IncludeFrom || choice.Name == picks.OutputFrom ? "range from here" : null,
                }.OfType<string>());

            html.Append("<tr data-column=\"").Append(Encoded(choice.Name)).Append("\">")
                .Append("<td class=\"deepsharp-name\">").Append(Encoded(choice.Name)).Append("</td>")
                .Append("<td class=\"deepsharp-values\">").Append(Encoded(at < 0 ? string.Empty : Values(_first, at))).Append("</td>")
                .Append("<td class=\"deepsharp-in\">");
            Box(html, choice, list);
            html.Append("</td><td class=\"deepsharp-kind\">");
            Select(html, choice, list);

            if (choice.Kind is null && _taking[choice.Name].Said() is { } proposed)
            {
                html.Append(" <span class=\"deepsharp-proposed\">").Append(Encoded(proposed)).Append("</span>");
            }

            html.Append("</td><td class=\"deepsharp-answer\">");
            Answer(html, choice, list, reads is not null);
            html.Append("</td><td class=\"deepsharp-mark\">").Append(Encoded(mark)).Append("</td></tr>");
        }

        return CellOutput.Html(html.Append("</tbody></table></div>").ToString());
    }

    // Above the rows: ticking one column or a range, and in a range the one kind it takes its columns in with — every
    // kind a schema declares, text first, as the source holds it.
    private static void Picks(StringBuilder html, ListDrawing list)
    {
        html.Append("<div class=\"deepsharp-picks\">tick <select data-action=\"")
            .Append(Encoded(ControlAction.Of(StepRenderer.ListRange, list.Carried([])))).Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append("\">")
            .Append("<option value=\"one\"").Append(list.Picks.Range ? string.Empty : " selected").Append(">one column</option>")
            .Append("<option value=\"range\"").Append(list.Picks.Range ? " selected" : string.Empty).Append(">a range</option></select>");

        if (list.Picks.Range)
        {
            html.Append(" taking them in as ");
            Kinds(html, RangeKind(list, "include"), [ColumnKind.Text, .. Enum.GetValues<ColumnKind>().Where(kind => kind != ColumnKind.Text)], list.Picks.IncludeKind ?? ColumnKind.Text);
        }

        html.Append("</div>");
    }

    // What a range's kind select sends: the pick, and what the range is of.
    private static string RangeKind(ListDrawing list, string of) =>
        ControlAction.Of(StepRenderer.ListRangeKind, list.Carried(new JsonObject { [StepRenderer.ForKey] = of }));

    // A range's kind: a pick, which carries what the range is of in what it sends.
    private static void Kinds(StringBuilder html, string action, IReadOnlyList<ColumnKind> kinds, ColumnKind picked)
    {
        html.Append("<select data-action=\"")
            .Append(Encoded(action))
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append("\">");

        foreach (var kind in kinds)
        {
            html.Append("<option value=\"").Append(kind.Word()).Append('"').Append(kind == picked ? " selected" : string.Empty)
                .Append('>').Append(kind.Word()).Append("</option>");
        }

        html.Append("</select>");
    }

    // The output's section: the select that picks the kind of output the boxes make, each kind no row can take drawn
    // disabled with the rule that stops it; the box that takes the output away; and in a range of an output of many,
    // the kind the range takes its columns in with — one the output reads, never text.
    private void Output(StringBuilder html, ListDrawing list, IReadOnlyList<ColumnKind>? reads)
    {
        html.Append("<div class=\"deepsharp-output\">the output: <select data-action=\"")
            .Append(Encoded(ControlAction.Of(StepRenderer.ListType, list.Carried([])))).Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append("\">");

        foreach (var each in _verbs)
        {
            html.Append("<option value=\"").Append(Encoded(each)).Append('"')
                .Append(each == list.Verb ? " selected" : string.Empty)
                .Append(_stopped[each] is null ? string.Empty : " disabled").Append('>').Append(Encoded(each))
                .Append(_stopped[each] is { } why ? Encoded($" — {why}") : string.Empty).Append("</option>");
        }

        html.Append("</select> <label><input type=\"checkbox\" data-action=\"")
            .Append(Encoded(ControlAction.Of(StepRenderer.ListRemoveOutput, list.Carried([]))))
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append('"')
            .Append(_declaration.Output is null || !list.Whole ? " disabled" : string.Empty)
            .Append("> remove the output</label>");

        if (reads is not null)
        {
            html.Append(" answer columns as ");
            Kinds(html, RangeKind(list, "output"), reads, list.Picks.OutputKind ?? reads[0]);
        }

        html.Append("</div>");
    }

    // A select for each of the output's own values, offering what the rules keep, not said written as such; a value the
    // list cannot offer is set in the output block's form. A select carries its key and what the list was drawn from and
    // with, and the router sends the value it is at.
    private static void Parameters(StringBuilder html, IReadOnlyList<OutputParameter> parameters, ListDrawing list)
    {
        html.Append("<div class=\"deepsharp-parameters\">");

        foreach (var parameter in parameters)
        {
            if (parameter.Options is not { } options)
            {
                html.Append("<span class=\"deepsharp-parameter\">")
                    .Append(Encoded($"{parameter.Key}: {parameter.Current} — set in the output block's form")).Append("</span> ");

                continue;
            }

            var action = ControlAction.Of(StepRenderer.ListParameter, list.Carried(new JsonObject { [StepRenderer.ParameterKey] = parameter.Key }));

            html.Append("<label>").Append(Encoded(parameter.Key)).Append(" <select data-action=\"").Append(Encoded(action))
                .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append('"').Append(list.Whole ? string.Empty : " disabled").Append('>');

            foreach (var option in options)
            {
                html.Append("<option value=\"").Append(Encoded(option)).Append('"').Append(option == parameter.Current ? " selected" : string.Empty)
                    .Append('>').Append(option.Length == 0 ? "not said" : Encoded(option)).Append("</option>");
            }

            html.Append("</select></label> ");
        }

        html.Append("</div>");
    }

    // Why no row can make a kind of output: the first row's reason; nothing when some row can. A kind whose answer is
    // many columns is made by a range, which no single tick starts, so it is never stopped here: a range the rules refuse
    // says so itself.
    private string? Stopped(string verb)
    {
        if (OutputBox.AnswerOf(_catalog, verb) is ColumnsParameter)
        {
            return null;
        }

        var boxes = new OutputBox(_catalog, _declaration, verb, _header);
        string? first = null;

        foreach (var choice in _rows)
        {
            var why = boxes.NotOffered(choice.Name, _kinds[choice.Name], choice.Role == ColumnRole.Answer);

            if (why is null)
            {
                return null;
            }

            first ??= why;
        }

        return first;
    }

    // A row's output box: ticked when its column is an answer, clickable when the rules allow what the click asks for,
    // carrying the kind of output it makes; and what the column is to the output. In a range of an output of many a box
    // not ticked starts or ends the range, and the range speaks for itself.
    private void Answer(StringBuilder html, ColumnChoice choice, ListDrawing list, bool ranged)
    {
        var kind = _kinds[choice.Name];
        var ticked = choice.Role == ColumnRole.Answer;
        var enabled = list.Whole && ((ranged && !ticked) || new OutputBox(_catalog, _declaration, list.Verb, _header).NotOffered(choice.Name, kind, ticked) is null);
        var action = ControlAction.Of(StepRenderer.ListOutput, list.Carried(new JsonObject
        {
            [StepRenderer.ColumnKey] = choice.Name,
            [StepRenderer.KindKey] = kind.Word(),
        }));

        html.Append("<label><input type=\"checkbox\" data-action=\"").Append(Encoded(action))
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append('"')
            .Append(ticked ? " checked" : string.Empty)
            .Append(enabled ? string.Empty : " disabled")
            .Append("> answer</label> <span class=\"deepsharp-role\">")
            .Append(choice.Role switch { ColumnRole.Answer => "answer", ColumnRole.Scale => "its way back reads it", _ => string.Empty })
            .Append("</span>");
    }

    // A row's kind select: the kind the schema declares, or none for a column it does not name, then every kind the rules
    // let the column take. It carries its column and what the list was drawn from, and the router sends its value.
    private void Select(StringBuilder html, ColumnChoice choice, ListDrawing list)
    {
        var declared = _declared.FirstOrDefault(column => column.Name == choice.Name)?.Kind;
        var kinds = _declaration.KindsFor(choice.Name, _header);
        var action = ControlAction.Of(StepRenderer.ListKind, list.Carried(new JsonObject { [StepRenderer.ColumnKey] = choice.Name }));

        html.Append("<select data-action=\"").Append(Encoded(action))
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append('"')
            .Append(kinds.Count == 0 ? " disabled" : string.Empty).Append('>');

        // A column the schema does not name picks no kind yet: sent again, as a click or a key sends it, it changes nothing.
        if (declared is null)
        {
            html.Append("<option value=\"\" selected>")
                .Append(choice.Standing == ColumnStanding.Kept ? "kept as text" : "not taken").Append("</option>");
        }

        foreach (var kind in kinds)
        {
            html.Append("<option value=\"").Append(kind.Word()).Append('"')
                .Append(kind == declared ? " selected" : string.Empty).Append('>').Append(kind.Word()).Append("</option>");
        }

        html.Append("</select>");
    }

    // A row's box: the gesture, the column, the kind a tick takes it in with and the format it reads its moments by — as the
    // schema declares it, else by the one rule every box keeps — and what the list was drawn from and with.
    private void Box(StringBuilder html, ColumnChoice choice, ListDrawing list)
    {
        var taken = choice.Kind is null ? _taking[choice.Name] : new TakenIn(_kinds[choice.Name]);
        var box = choice.IncludedBox();
        var action = ControlAction.Of(StepRenderer.ListInclude, list.Carried(taken.Into(new JsonObject { [StepRenderer.ColumnKey] = choice.Name })));

        html.Append("<label><input type=\"checkbox\" data-action=\"").Append(Encoded(action))
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append('"')
            .Append(box.Ticked ? " checked" : string.Empty)
            .Append(box.Enabled ? string.Empty : " disabled")
            .Append("> in</label>");
    }

    private static string Values(IReadOnlyList<IReadOnlyList<string?>> rows, int at) =>
        string.Join(", ", rows.Select(row => row[at] is { Length: > 0 } value ? value : Blank));

    private static string Encoded(string text) => WebUtility.HtmlEncode(text);

    /// <summary>What a list is drawn with, which each of its controls carries so a list drawn again keeps it.</summary>
    /// <param name="Verb">The kind of output its boxes make.</param>
    /// <param name="Key">The key of the whole declaration it is drawn for.</param>
    /// <param name="Fingerprint">The fingerprint of the source's bytes it is drawn from.</param>
    /// <param name="Picks">The picks it is drawn with.</param>
    /// <param name="Whole">Whether every block is in the declaration: an output is changed only then.</param>
    private readonly record struct ListDrawing(string Verb, string Key, string Fingerprint, ListPicks Picks, bool Whole)
    {
        /// <summary>What a control carries: its own values, then what the list was drawn with and from.</summary>
        /// <param name="own">The control's own values.</param>
        /// <returns>All it carries.</returns>
        public JsonObject Carried(JsonObject own)
        {
            Picks.Into(own, Verb);
            own[StepRenderer.DrawnKey] = Key;
            own[StepRenderer.SourceKey] = Fingerprint;

            return own;
        }
    }
}
