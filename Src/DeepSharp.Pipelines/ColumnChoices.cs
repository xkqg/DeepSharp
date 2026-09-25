// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>How a column stands in a pipeline.</summary>
public enum ColumnStanding
{
    /// <summary>The schema declares it and it takes part.</summary>
    Taking,

    /// <summary>The schema does not name it, and keeps it with the rest of the file.</summary>
    Kept,

    /// <summary>A step makes it.</summary>
    Made,

    /// <summary>The schema names it with its kind and excludes it.</summary>
    Excluded,

    /// <summary>A step leaves it out, after the steps that read it.</summary>
    Dropped,

    /// <summary>The schema does not name it, and the rest of the file is not kept.</summary>
    NotDeclared,
}

/// <summary>What can be done to a column, as the rules allow it.</summary>
[Flags]
public enum ColumnOffers
{
    /// <summary>Nothing.</summary>
    None = 0,

    /// <summary>It can be taken in.</summary>
    Include = 1,

    /// <summary>It can be left out.</summary>
    Exclude = 2,

    /// <summary>It can be made a category.</summary>
    MakeCategory = 4,

    /// <summary>A category can go back to the kind it was.</summary>
    BackToWas = 8,
}

/// <summary>One column, as it stands and what can be done to it.</summary>
/// <param name="Name">The column's name.</param>
/// <param name="Standing">How it stands.</param>
/// <param name="Kind">What it holds, when that is known: the kind the schema declares, or the kind a step made it with.</param>
/// <param name="Was">The kind a category was before it became one, when it says.</param>
/// <param name="Offers">What can be done to it without breaking a rule.</param>
public readonly record struct ColumnChoice(string Name, ColumnStanding Standing, ColumnKind? Kind, ColumnKind? Was, ColumnOffers Offers);

/// <summary>Every column asked about, as each stands.</summary>
/// <param name="Rows">One row per column asked, in the order they were asked.</param>
public sealed record ColumnChoices(IReadOnlyList<ColumnChoice> Rows);

/// <summary>
/// What can be done to a pipeline's columns, and how each stands: the one set of rules every door that changes the
/// columns goes through — a list of the source's columns, a grid of the data, a take-over of saved decisions.
/// </summary>
/// <remarks>
/// Each operation hands back the steps it would make and never judges them: the rules every declaration keeps do
/// that, through <see cref="PipelineDeclaration.FaultsIn"/>, so the rules and what is offered cannot disagree. Asked
/// for what already is, an operation hands back the steps it was given, so the same gesture twice does the same
/// thing once. The schema's own operations — <see cref="DeclareStep.WithColumn"/>,
/// <see cref="DeclareStep.WithColumnExcluded"/>, <see cref="DeclareStep.WithColumnKind"/> — change one block; these
/// change the pipeline, and leave a column out where the steps that read it allow.
/// </remarks>
public static class ColumnChoiceExtensions
{
    /// <summary>The steps with a column taken in.</summary>
    /// <param name="declaration">The pipeline.</param>
    /// <param name="column">The column.</param>
    /// <param name="kind">The kind it takes when the schema does not name it yet; a declared column keeps its own.</param>
    /// <param name="header">The source's columns, in order: a new column stands where the source has it.</param>
    /// <returns>
    /// The steps with the column taking part: an excluded one brought back as it was, a dropped one's name taken out
    /// of its drop — a drop left empty goes — and one the schema does not name taken in with the kind given. The steps
    /// as they are when the column takes part already, or when nothing here can bring it back.
    /// </returns>
    public static IReadOnlyList<IPipelineStep> Including(
        this PipelineDeclaration declaration, string column, ColumnKind kind, IReadOnlyList<string> header)
    {
        ArgumentNullException.ThrowIfNull(declaration);
        ArgumentNullException.ThrowIfNull(header);

        var steps = declaration.Steps;

        if (Schema(declaration) is not { } declare || Reaches(declaration, column))
        {
            return steps;
        }

        var declared = declare.Columns.FirstOrDefault(each => each.Name == column);

        if (declared is { Excluded: true })
        {
            return Replaced(steps, declaration.ColumnsAt, declare.WithColumn(column, kind, header));
        }

        if (DropOf(steps, column) is { } at)
        {
            string[] left = [.. ((DropColumnsStep)steps[at]).Columns.Where(each => each != column)];

            return left.Length == 0 ? [.. steps.Take(at), .. steps.Skip(at + 1)] : Replaced(steps, at, new DropColumnsStep(left));
        }

        return declared is null && !Made(declaration, column)
            ? Replaced(steps, declaration.ColumnsAt, declare.WithColumn(column, kind, header))
            : steps;
    }

    /// <summary>The steps with a column left out.</summary>
    /// <param name="declaration">The pipeline.</param>
    /// <param name="column">The column.</param>
    /// <returns>
    /// The steps without it: a declared column no step reads is excluded in the schema, keeping its kind; any other is
    /// dropped after the last step that reads it, or after the step that makes it, as another name in a drop standing
    /// there. The steps as they are when the column does not reach the end.
    /// </returns>
    public static IReadOnlyList<IPipelineStep> Excluding(this PipelineDeclaration declaration, string column)
    {
        ArgumentNullException.ThrowIfNull(declaration);

        var steps = declaration.Steps;

        if (Schema(declaration) is not { } declare || !Reaches(declaration, column))
        {
            return steps;
        }

        var readers = Enumerable.Range(0, steps.Count).Where(at => steps[at].ColumnsRead.Any(read => read.Column == column)).ToArray();

        if (readers.Length == 0 && declare.Columns.Any(each => each.Name == column))
        {
            return Replaced(steps, declaration.ColumnsAt, declare.WithColumnExcluded(column));
        }

        var after = Math.Max(readers.DefaultIfEmpty(-1).Max(), MadeAt(declaration, column)) + 1;

        return after < steps.Count && steps[after] is DropColumnsStep drop
            ? Replaced(steps, after, new DropColumnsStep([.. drop.Columns, column]))
            : [.. steps.Take(after), new DropColumnsStep([column]), .. steps.Skip(after)];
    }

    /// <summary>The steps with a column the schema names given another kind.</summary>
    /// <param name="declaration">The pipeline.</param>
    /// <param name="column">The column.</param>
    /// <param name="kind">The kind.</param>
    /// <returns>The steps with the schema changed, as <see cref="DeclareStep.WithColumnKind"/> changes it.</returns>
    /// <exception cref="ArgumentException">The schema does not name the column; taking one in is <see cref="Including"/>.</exception>
    public static IReadOnlyList<IPipelineStep> WithKind(this PipelineDeclaration declaration, string column, ColumnKind kind)
    {
        ArgumentNullException.ThrowIfNull(declaration);

        return Schema(declaration) is { } declare
            ? Replaced(declaration.Steps, declaration.ColumnsAt, declare.WithColumnKind(column, kind))
            : declaration.Steps;
    }

    /// <summary>How each column asked about stands, and what can be done to it without breaking a rule.</summary>
    /// <param name="declaration">The pipeline.</param>
    /// <param name="columns">The columns, in the order the rows are wanted: the source's, or those a block shows.</param>
    /// <returns>One row per column asked.</returns>
    /// <remarks>
    /// An operation is offered when it changes the steps and the rules keep what it makes: so leaving out the answer
    /// is not offered, nor making a category of a column a step below scales as a number.
    /// </remarks>
    public static ColumnChoices ChoicesFor(this PipelineDeclaration declaration, IReadOnlyList<string> columns)
    {
        ArgumentNullException.ThrowIfNull(declaration);
        ArgumentNullException.ThrowIfNull(columns);

        return new([.. columns.Select(column => Choice(declaration, column, columns))]);
    }

    private static ColumnChoice Choice(PipelineDeclaration declaration, string column, IReadOnlyList<string> header)
    {
        var declared = Schema(declaration)?.Columns.FirstOrDefault(each => each.Name == column);
        var offers = ColumnOffers.None;

        if (Changes(declaration, declaration.Including(column, ColumnKind.Text, header)))
        {
            offers |= ColumnOffers.Include;
        }

        if (Changes(declaration, declaration.Excluding(column)))
        {
            offers |= ColumnOffers.Exclude;
        }

        if (declared is { Kind: not ColumnKind.Category } && Changes(declaration, declaration.WithKind(column, ColumnKind.Category)))
        {
            offers |= ColumnOffers.MakeCategory;
        }

        if (declared is { Kind: ColumnKind.Category, Was: { } was } && Changes(declaration, declaration.WithKind(column, was)))
        {
            offers |= ColumnOffers.BackToWas;
        }

        return new(column, Standing(declaration, declared, column), declared?.Kind ?? KnownKind(declaration, column), declared?.Was, offers);
    }

    private static ColumnStanding Standing(PipelineDeclaration declaration, ColumnDeclaration? declared, string column)
    {
        if (declared is { Excluded: true })
        {
            return ColumnStanding.Excluded;
        }

        if (DropOf(declaration.Steps, column) is not null)
        {
            return ColumnStanding.Dropped;
        }

        if (declared is not null)
        {
            return ColumnStanding.Taking;
        }

        if (Made(declaration, column))
        {
            return ColumnStanding.Made;
        }

        return Reaches(declaration, column) ? ColumnStanding.Kept : ColumnStanding.NotDeclared;
    }

    // Whether an operation changes the steps, and the rules keep what it makes.
    private static bool Changes(PipelineDeclaration declaration, IReadOnlyList<IPipelineStep> made) =>
        !made.SequenceEqual(declaration.Steps) && PipelineDeclaration.FaultsIn(made).Count == 0;

    private static DeclareStep? Schema(PipelineDeclaration declaration) => declaration.Steps.OfType<DeclareStep>().FirstOrDefault();

    private static bool Reaches(PipelineDeclaration declaration, string column) =>
        declaration.ColumnsBefore(declaration.Steps.Count).Allows(column);

    // Where a column comes to be: the first step after which it may be read.
    private static int MadeAt(PipelineDeclaration declaration, string column) =>
        Enumerable.Range(0, declaration.Steps.Count).First(at => declaration.ColumnsBefore(at + 1).Allows(column));

    // A column no source brought, that a step names as one it leaves behind.
    private static bool Made(PipelineDeclaration declaration, string column) => KnownKind(declaration, column) is not null;

    private static ColumnKind? KnownKind(PipelineDeclaration declaration, string column) =>
        Enumerable.Range(0, declaration.Steps.Count + 1)
            .Select(at => declaration.ColumnsBefore(at).Find(column))
            .FirstOrDefault(known => known is not null)?.Kind;

    // The drop that names a column; in a declaration that keeps its rules there is at most one, since a column is
    // gone after it.
    private static int? DropOf(IReadOnlyList<IPipelineStep> steps, string column)
    {
        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is DropColumnsStep drop && drop.Columns.Contains(column, StringComparer.Ordinal))
            {
                return at;
            }
        }

        return null;
    }

    // The steps with one replaced; the very steps given when it is the same step, so nothing changed reads as nothing changed.
    private static IReadOnlyList<IPipelineStep> Replaced(IReadOnlyList<IPipelineStep> steps, int at, IPipelineStep step) =>
        steps[at].Equals(step) ? steps : [.. steps.Take(at), step, .. steps.Skip(at + 1)];
}
