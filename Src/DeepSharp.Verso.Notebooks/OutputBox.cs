// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// What a control of the list's output section asks for — a row's output box, or a select setting one of the output's
/// values: the steps it makes, or why it is not made.
/// </summary>
/// <param name="Steps">
/// The steps, when the change can be made; the steps as they are when it asks for what already holds; nothing when it is
/// not made, or asks for nothing the output holds.
/// </param>
/// <param name="NotMade">Why it is not made; none when it is.</param>
internal readonly record struct ListOutputChange(IReadOnlyList<IPipelineStep>? Steps, IReadOnlyList<string> NotMade);

/// <summary>
/// A row's box for whether its column is an answer of the output: the one rule the list draws the box by and acts on,
/// so a box that can be clicked makes what the click asks for.
/// </summary>
/// <remarks>
/// A kind of output names its answer in its first column, or list of columns, it cannot do without. Ticking puts the
/// row's column there and unticking takes it out, through the one builder every door makes an output with — the rest
/// of what the output holds stays — and the output is placed as the column rules place one. A column of an output of
/// many goes in after the nearest column the output holds before it in the source's order, first when none does, and
/// comes out where it stands, so a tick never moves a column it does not touch. A tick on a column that does not reach
/// the end takes it in first, in the same change. The verb's own rules refuse what they refuse, in its words; and an
/// output of another kind than the one picked is not changed from here. A range of an output of many puts every column
/// of it in the same way, in one change, and takes them in with the range's one kind — a column the schema gives a kind
/// the output does not read is made that kind, since the range asks for its answers as that kind.
/// </remarks>
internal static class OutputBox
{
    /// <summary>What a row's output box asks for.</summary>
    /// <param name="catalog">The verbs the notebook knows.</param>
    /// <param name="declaration">The declaration the blocks make.</param>
    /// <param name="verb">The kind of output the list's boxes make.</param>
    /// <param name="column">The row's column.</param>
    /// <param name="kind">The kind a tick takes the column in with, when the schema does not take it.</param>
    /// <param name="header">The source's columns, in their order.</param>
    /// <param name="ticked">Whether the box is ticked.</param>
    /// <returns>The steps, or why not.</returns>
    public static ListOutputChange Change(
        StepCatalog catalog, PipelineDeclaration declaration, string verb, string column, ColumnKind kind, IReadOnlyList<string> header, bool ticked) =>
        Change(catalog, declaration, verb, [column], kind, header, ticked, reads: null);

    /// <summary>What a range of an output of many asks for: every column of it put in, in one change.</summary>
    /// <param name="catalog">The verbs the notebook knows.</param>
    /// <param name="declaration">The declaration the blocks make.</param>
    /// <param name="verb">The kind of output the list's boxes make; its answer is a list of columns.</param>
    /// <param name="columns">The range's columns, in the source's order.</param>
    /// <param name="kind">The range's kind.</param>
    /// <param name="header">The source's columns, in their order.</param>
    /// <returns>The steps, or why not.</returns>
    public static ListOutputChange Ranged(
        StepCatalog catalog, PipelineDeclaration declaration, string verb, IReadOnlyList<string> columns, ColumnKind kind, IReadOnlyList<string> header) =>
        Change(catalog, declaration, verb, columns, kind, header, ticked: true, reads: ((ColumnsParameter)AnswerOf(catalog, verb)!).Accepts);

    // Puts columns into the output, or takes them out; for a range, a column the schema gives a kind the output does not
    // read is made the range's kind.
    private static ListOutputChange Change(
        StepCatalog catalog, PipelineDeclaration declaration, string verb, IReadOnlyList<string> columns, ColumnKind kind, IReadOnlyList<string> header,
        bool ticked, IReadOnlyList<ColumnKind>? reads)
    {
        var output = declaration.Output;

        if (output is not null && output.Verb != verb)
        {
            return new ListOutputChange(null, [$"the output is '{output.Verb}': pick it above to change its columns, or take it away first."]);
        }

        if (AnswerOf(catalog, verb) is not { } answer)
        {
            return new ListOutputChange(null, [$"'{verb}' is not a kind of output the list can make."]);
        }

        var held = output is null ? [] : Held(output, answer);
        string[] asked = [.. columns.Where(column => held.Contains(column, StringComparer.Ordinal) != ticked)];

        if (asked.Length == 0)
        {
            return new ListOutputChange(declaration.Steps, []);
        }

        JsonNode stated = answer is ColumnsParameter
            ? new JsonArray([.. (ticked ? asked.Aggregate(held, (each, column) => Inserted(each, column, header)) : [.. held.Except(asked)]).Select(each => (JsonNode)each)])
            : ticked ? asked[0] : string.Empty;

        INamesTheAnswer made;

        try
        {
            made = (INamesTheAnswer)catalog.Make(verb, new JsonObject { [answer.Key] = stated }, (IPipelineStep?)output);
        }
        catch (PipelineFileException refused)
        {
            return new ListOutputChange(null, [.. refused.Faults.Select(fault => fault.Message)]);
        }

        // A column the schema does not take is taken in first, in the same change.
        IReadOnlyList<IPipelineStep> taken;

        try
        {
            taken = ticked ? TakenIn(declaration, asked, kind, header, reads) : declaration.Steps;
        }
        catch (DeclarationException refused)
        {
            return new ListOutputChange(null, [.. refused.Faults.Select(fault => fault.ToString())]);
        }

        var faults = PipelineDeclaration.FaultsIn(taken);

        return faults.Count > 0
            ? new ListOutputChange(null, [.. faults.Select(fault => fault.ToString())])
            : new ListOutputChange(new PipelineDeclaration(taken).WithOutput(made), []);
    }

    // The columns taken in with the kind given; for a range, one the schema then gives a kind the output does not read is
    // made that kind. A column whose change breaks a rule stops the rest, and the rule is said.
    private static IReadOnlyList<IPipelineStep> TakenIn(
        PipelineDeclaration declaration, IReadOnlyList<string> columns, ColumnKind kind, IReadOnlyList<string> header, IReadOnlyList<ColumnKind>? reads) =>
        reads is null
            ? declaration.Including(columns, kind, header)
            : columns.Aggregate(declaration.Including(columns, kind, header), (steps, column) =>
                steps.OfType<DeclareStep>().First().Taking.FirstOrDefault(each => each.Name == column) is { } declared && !reads.Contains(declared.Kind)
                    ? new PipelineDeclaration(steps).WithKind(column, kind)
                    : steps);

    /// <summary>The kinds a range of an output's columns can take them in with: those its answer reads, never text.</summary>
    /// <param name="catalog">The verbs the notebook knows.</param>
    /// <param name="verb">The kind of output.</param>
    /// <returns>The kinds, in the order the answer names them; nothing for a kind of output whose answer is one column.</returns>
    public static IReadOnlyList<ColumnKind>? RangeKinds(StepCatalog catalog, string verb) =>
        AnswerOf(catalog, verb) is ColumnsParameter many ? [.. many.Accepts.Where(kind => kind != ColumnKind.Text)] : null;

    /// <summary>
    /// Whether a row's output box can be clicked, as the include box's offers are: what the click asks for changes the
    /// steps and keeps the rules.
    /// </summary>
    /// <param name="catalog">The verbs the notebook knows.</param>
    /// <param name="declaration">The declaration the blocks make.</param>
    /// <param name="verb">The kind of output the list's boxes make.</param>
    /// <param name="column">The row's column.</param>
    /// <param name="kind">The kind a tick takes the column in with.</param>
    /// <param name="header">The source's columns.</param>
    /// <param name="ticked">Whether the box is drawn ticked.</param>
    /// <returns>Why the click is not offered, the first reason; nothing when it is.</returns>
    /// <remarks>
    /// A column an answer is made from is drawn ticked, since the answer comes back to it, while the output names the
    /// column made: unticking it takes out nothing the output names, so it is not offered.
    /// </remarks>
    public static string? NotOffered(
        StepCatalog catalog, PipelineDeclaration declaration, string verb, string column, ColumnKind kind, IReadOnlyList<string> header, bool ticked)
    {
        var change = Change(catalog, declaration, verb, column, kind, header, !ticked);

        return change.Steps is not { } steps ? change.NotMade[0]
            : steps.SequenceEqual(declaration.Steps) ? "nothing changes."
            : PipelineDeclaration.FaultsIn(steps) is [var first, ..] ? first.Message
            : null;
    }

    /// <summary>The kinds of output a list can make: the catalog's outputs that name their answer in a column.</summary>
    /// <param name="catalog">The verbs the notebook knows.</param>
    /// <returns>Their verbs, in the catalog's order.</returns>
    public static IReadOnlyList<string> Verbs(StepCatalog catalog) =>
        [.. catalog.Descriptions.Where(description => AnswerOf(catalog, description.Verb) is not null).Select(description => description.Verb)];

    /// <summary>
    /// Where a kind of output the list can make names its answer: its first column, or list of columns, it cannot do
    /// without. The one rule the list's type select is drawn by, its boxes act by, and its value selects leave alone.
    /// </summary>
    /// <param name="catalog">The verbs the notebook knows.</param>
    /// <param name="verb">The kind of output.</param>
    /// <returns>The parameter; nothing for a verb the catalog does not know, or one that makes no output.</returns>
    internal static StepParameter? AnswerOf(StepCatalog catalog, string verb) =>
        catalog.Knows(verb) && catalog.Describe(verb) is var description && catalog.ReadStep(description.Template) is INamesTheAnswer
            ? description.Parameters.FirstOrDefault(parameter => parameter is ColumnParameter { Optional: false } or ColumnsParameter { Optional: false })
            : null;

    // The columns an output names as its answer, read as the answer is written: a list for a parameter of many columns,
    // one column else.
    private static IReadOnlyList<string> Held(INamesTheAnswer output, StepParameter answer)
    {
        var stated = JsonNode.Parse(output.AsBlockText())![answer.Key]!;

        return answer is ColumnsParameter ? [.. stated.AsArray().Select(each => each!.GetValue<string>())] : [stated.GetValue<string>()];
    }

    // A column put into an output of many: after the nearest column the output holds before it in the source's order,
    // or first when none does.
    private static IReadOnlyList<string> Inserted(IReadOnlyList<string> held, string column, IReadOnlyList<string> header)
    {
        var place = header.PlaceOf(column);
        var before = held.Where(each => header.PlaceOf(each) < place).MaxBy(header.PlaceOf);
        var at = before is null ? 0 : held.ToList().IndexOf(before) + 1;

        return [.. held.Take(at), column, .. held.Skip(at)];
    }
}
