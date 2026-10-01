// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// The steps a run takes: every declared step, or — in a run for a learner — each as that learner takes it: as it was
/// declared, what stands in its place, or left out.
/// </summary>
/// <remarks>
/// Worked out from the declaration and the stated need alone, never from the rows. A step some learners do without
/// (<see cref="IMeetsANeed"/>) is left out, or takes what it offers in its place, for a learner that does without it — unless
/// leaving it out would change more than that learner does without: an answer's way back runs through it, or a step below
/// reads a column it read, made, changed or took away, or a member of a family it made. Every walk goes through a course,
/// and so do the columns followed down a run, the fits a run has to have, and where a feature lands for a learner on one
/// scale. The way back goes through the declaration, of which no course leaves out a step it runs through.
/// </remarks>
internal sealed class Course
{
    private static readonly Needs[] EveryNeed = Enum.GetValues<Needs>();

    private Course(PipelineDeclaration declaration, IReadOnlyList<int> skipped)
    {
        IPipelineStep[] steps = [.. declaration.Steps];

        foreach (var at in skipped)
        {
            steps[at] = InPlaceOf(at, declaration.Steps[at]);
        }

        Declaration = declaration;
        Steps = steps;
        Skipped = skipped;
    }

    /// <summary>The steps as they were declared.</summary>
    public PipelineDeclaration Declaration { get; }

    /// <summary>The steps as the run takes them, one at each place of the declaration.</summary>
    public IReadOnlyList<IPipelineStep> Steps { get; }

    /// <summary>The places of the steps the run left out, or took in another form, in ascending order.</summary>
    public IReadOnlyList<int> Skipped { get; }

    /// <summary>A run of every step, as it was declared: what <see cref="Pipeline.Run()"/> walks.</summary>
    /// <param name="declaration">The declaration.</param>
    /// <returns>The course.</returns>
    public static Course Whole(PipelineDeclaration declaration)
    {
        ArgumentNullException.ThrowIfNull(declaration);

        return new(declaration, []);
    }

    /// <summary>A run for a learner with this need: every step it does without left out, or taken in the form it offers.</summary>
    /// <param name="declaration">The declaration.</param>
    /// <param name="needs">What the learner needs of its features.</param>
    /// <returns>The course.</returns>
    /// <exception cref="ArgumentOutOfRangeException">No need is named by the value.</exception>
    /// <exception cref="InvalidOperationException">A step offers in its place a step written otherwise than it is.</exception>
    public static Course For(PipelineDeclaration declaration, Needs needs) =>
        new(declaration, LeftOut(declaration, Candidates(declaration), needs.Named()));

    /// <summary>The course of a run that left out these steps, as its file names them.</summary>
    /// <param name="declaration">The declaration.</param>
    /// <param name="skipped">The places of the steps it left out, which <see cref="FaultsIn"/> finds nothing wrong with.</param>
    /// <returns>The course.</returns>
    public static Course Of(PipelineDeclaration declaration, IReadOnlyList<int> skipped) => new(declaration, [.. skipped.Distinct().Order()]);

    /// <summary>Everything wrong with these steps as the steps a run for one learner left out.</summary>
    /// <param name="declaration">The declaration.</param>
    /// <param name="skipped">The places of the steps said to be left out.</param>
    /// <returns>
    /// Each step no run leaves out, whatever its learner needs; or, when each is left out by some run, the first of them when
    /// no single run leaves out exactly these; nothing when they are the steps one run leaves out.
    /// </returns>
    public static IReadOnlyList<DeclarationFault> FaultsIn(PipelineDeclaration declaration, IReadOnlyList<int> skipped)
    {
        int[] asked = [.. skipped.Distinct().Order()];

        if (asked.Length == 0)
        {
            return [];
        }

        var candidates = Candidates(declaration);
        int[][] runs = [.. EveryNeed.Select(needs => LeftOut(declaration, candidates, needs))];

        DeclarationFault[] taken =
        [
            .. asked
                .Where(at => !runs.Any(run => run.Contains(at)))
                .Select(at => new DeclarationFault(at, declaration.Steps[at].Verb, "is taken by every run, whatever its learner needs, so no run leaves it out.")),
        ];

        return taken.Length > 0 || runs.Any(run => run.SequenceEqual(asked))
            ? taken
            :
            [
                new DeclarationFault(
                    asked[0],
                    declaration.Steps[asked[0]].Verb,
                    $"is one of the steps left out here, {string.Join(", ", asked.Select(at => at + 1))}, and no run for one learner leaves out exactly these."),
            ];
    }

    /// <summary>The columns there are before a step, as the run takes the steps above it.</summary>
    /// <param name="position">The step's place; the number of steps for after the last.</param>
    /// <returns>The columns, followed from the schema down through the steps the run took.</returns>
    public ColumnState ColumnsBefore(int position) =>
        ColumnFlow.Follow(Steps, ColumnState.None, ..position, declared: false).State;

    /// <summary>The steps this run left out that a learner with this need cannot do without.</summary>
    /// <param name="needs">What the learner needs.</param>
    /// <returns>Their places, in ascending order; none when the run can be handed to that learner.</returns>
    public IReadOnlyList<int> Missed(Needs needs) =>
        [.. Skipped.Where(at => ((IMeetsANeed)Declaration.Steps[at]).NeededBy(needs))];

    /// <summary>Each column this run hands over as the places of its categories, with the categories the training rows held.</summary>
    /// <param name="fitted">What the run's steps learned, by place.</param>
    /// <returns>The categories of each such column, in the order of their places; none when the run hands none over so.</returns>
    public IReadOnlyDictionary<string, IReadOnlyList<string>> CategoriesHanded(IReadOnlyDictionary<int, FittedStepValues> fitted)
    {
        var handed = new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal);

        foreach (var at in Skipped.Where(at => Steps[at] is CategoriesAsPlaces))
        {
            foreach (var (column, categories) in ((CategoriesAsPlaces)Steps[at]).Declared.CategoriesIn(fitted[at]))
            {
                handed[column] = categories;
            }
        }

        return handed;
    }

    // The steps a learner with this need does without, among those a run may leave out.
    private static int[] LeftOut(PipelineDeclaration declaration, int[] candidates, Needs needs) =>
        [.. candidates.Where(at => !((IMeetsANeed)declaration.Steps[at]).NeededBy(needs))];

    // The steps a run may leave out for some learner: each that some learners do without, on no answer's way back, whose
    // columns no step below reads.
    private static int[] Candidates(PipelineDeclaration declaration)
    {
        var onAWayBack = (declaration.Output?.Answers ?? [])
            .SelectMany(answer => UndoChain.For(declaration, answer).Links)
            .Select(link => link.At)
            .ToHashSet();
        ColumnState[] columns = [.. Enumerable.Range(0, declaration.Steps.Count + 1).Select(declaration.ColumnsBefore)];

        return
        [
            .. Enumerable.Range(0, declaration.Steps.Count)
                .Where(at => declaration.Steps[at] is IMeetsANeed && !onAWayBack.Contains(at) && !ReadBelow(declaration, columns, at)),
        ];
    }

    // Whether a step below reads a column this one read, made, changed or took away, or a member of a family it made. A step
    // that says nothing of the columns it leaves behind may read any of them.
    private static bool ReadBelow(PipelineDeclaration declaration, ColumnState[] columns, int at)
    {
        var touched = Touched(declaration.Steps[at], columns[at], columns[at + 1]);
        string[] families = [.. columns[at + 1].Families.Except(columns[at].Families)];

        return Enumerable.Range(at + 1, declaration.Steps.Count - at - 1).Any(below =>
            declaration.Steps[below] is not IDescribesColumns
            || Touched(declaration.Steps[below], columns[below], columns[below + 1])
                .Any(name => touched.Contains(name) || families.Any(family => name.StartsWith(family, StringComparison.Ordinal))));
    }

    // The columns a step names, and every column whose state it moved: one it made, changed or took away.
    private static HashSet<string> Touched(IPipelineStep step, ColumnState before, ColumnState after) =>
    [
        .. step.ColumnsRead.Select(read => read.Column),
        .. after.Columns.Where(column => before.Find(column.Name) != column).Select(column => column.Name),
        .. before.Columns.Where(column => after.Find(column.Name) is null).Select(column => column.Name),
    ];

    // What a run takes in the place of a step it left out: what the step offers, or nothing done at all. Either is written as
    // the step it stands for, since a run's file names every step as it was declared.
    private static IPipelineStep InPlaceOf(int at, IPipelineStep declared)
    {
        var taken = ((IMeetsANeed)declared).Instead ?? (IPipelineStep)new DoneWithout(declared);

        return taken.Canonical().AsSpan().SequenceEqual(declared.Canonical())
            ? taken
            : throw new InvalidOperationException(
                $"Step {at + 1}, '{declared.Verb}', offers in its place '{taken.Verb}', which is not written as the step it stands for. "
                + "A run's file names every step as it was declared, so what stands in a step's place is written as that step.");
    }
}

/// <summary>A step a run left out whole: in its place nothing is done, nothing is learned, and every column stays as it was.</summary>
/// <param name="Declared">The step as it was declared, which the run's file names.</param>
internal sealed record DoneWithout(IPipelineStep Declared) : IDescribesColumns
{
    /// <inheritdoc />
    public string Verb => Declared.Verb;

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer) => Declared.WriteTo(writer);

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;
}
