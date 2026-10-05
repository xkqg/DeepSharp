// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// How far a notebook has followed a course: which of the course's steps its blocks already say, and which come next.
/// </summary>
/// <remarks>
/// <para>
/// Read from the blocks' text alone, so a block that waits for something, and so does not read as a step, still counts as
/// the step it names. A notebook may skip a step the course names — settling, features and a scale are for the pipelines
/// that need them — and may hold a step the course does not name; the course then goes on from behind the last step it
/// says. A step skipped is not written back, even one a later step needs: writing it would put a block that waits above
/// blocks that work, and the rule that needs it says so at the block that does. What a notebook may not do is take the
/// steps in another order than the course teaches, since a course continued from there would say one thing and the blocks
/// another. Such a notebook has gone its own way, and a course starts one.
/// </para>
/// <para>
/// A block says a step of the course when its verb is the step's, or — for the steps a pipeline holds only one of, which
/// its rules name: a source, the schema, a split, an answer, a learner, a report — when it is another verb of the same
/// kind, so a Parquet file stands where the course reads a comma-separated one. The order is not among them: a series is
/// put in its order and a table shuffled, so a shuffle says nothing of a series'. A block that takes columns away says the
/// course's drop wherever it stands, before what learns or after it, and neither moves the course on nor takes it away.
/// </para>
/// </remarks>
internal sealed class CourseProgress
{
    /// <summary>The kinds of step a declaration holds one of: each is a rule of its own, and a test holds this list to them.</summary>
    internal static readonly IReadOnlyList<Type> OnceOnly =
    [
        typeof(IOpensRows), typeof(IBindsColumns), typeof(IOrdersRows), typeof(ISplitStep), typeof(INamesTheAnswer),
        typeof(INamesTheLearner), typeof(INamesTheMeasures),
    ];

    // Of those, the kinds whose verbs stand for one another in a course. Not the order: its two verbs are opposite choices.
    private static readonly IReadOnlyList<Type> Interchangeable = [.. OnceOnly.Where(kind => kind != typeof(IOrdersRows))];

    private CourseProgress(bool follows, int matched, IReadOnlyList<CourseStep> missing, int insertAt)
    {
        Follows = follows;
        Matched = matched;
        Missing = missing;
        InsertAt = insertAt;
    }

    /// <summary>Whether the blocks take the course's steps in its order, as far as they say any.</summary>
    public bool Follows { get; }

    /// <summary>How many of the course's steps the blocks say.</summary>
    public int Matched { get; }

    /// <summary>The steps of the course still to come, in order: those behind the last step the blocks say, but for a drop they say.</summary>
    public IReadOnlyList<CourseStep> Missing { get; }

    /// <summary>The place in the course of the first step still to come; the course's length when nothing is.</summary>
    public int NextRow { get; private init; }

    /// <summary>Where the next block goes: right after the last block that is a step, or at the end when there is none.</summary>
    public int InsertAt { get; }

    /// <summary>How far a notebook's cells have followed a course.</summary>
    /// <param name="course">The course.</param>
    /// <param name="cells">The notebook's cells, in order; those that are not blocks are passed over.</param>
    /// <param name="catalog">The verbs a notebook knows.</param>
    /// <returns>The progress.</returns>
    public static CourseProgress Of(PipelineCourse course, IReadOnlyList<CellModel> cells, StepCatalog catalog)
    {
        var kinds = new Kinds(catalog);
        var pointer = 0;
        var matched = 0;
        var last = -1;
        HashSet<int> dropped = [];

        for (var at = 0; at < cells.Count; at++)
        {
            if (cells[at].Type != StepCellType.StepType)
            {
                continue;
            }

            last = at;

            var verb = StepText.Of(cells[at].Source).Verb;
            var row = verb is not null && catalog.Knows(verb) ? RowOf(course, verb, kinds) : -1;

            if (row < 0)
            {
                continue;
            }

            if (kinds.Of(verb!).Drops)
            {
                dropped.Add(row);
                matched++;

                continue;
            }

            // In order, or the same step said again; any other is a step taken out of the order the course teaches.
            if (row >= pointer)
            {
                pointer = row + 1;
                matched++;
            }
            else if (row != pointer - 1)
            {
                return new CourseProgress(follows: false, matched, [], at + 1) { NextRow = course.Steps.Count };
            }
        }

        var next = Enumerable.Range(pointer, course.Steps.Count - pointer).Where(row => !dropped.Contains(row)).ToArray();

        return new CourseProgress(follows: true, matched, [.. next.Select(row => course.Steps[row])], last >= 0 ? last + 1 : cells.Count)
        {
            NextRow = next.Length > 0 ? next[0] : course.Steps.Count,
        };
    }

    /// <summary>The course a notebook is following: the one its blocks say most of, and the table's when they say as much of both.</summary>
    /// <param name="cells">The notebook's cells, in order.</param>
    /// <param name="catalog">The verbs a notebook knows.</param>
    /// <returns>The course; nothing when the blocks follow neither.</returns>
    public static PipelineCourse? FollowedBy(IReadOnlyList<CellModel> cells, StepCatalog catalog)
    {
        PipelineCourse? followed = null;
        var most = -1;

        foreach (var course in PipelineCourse.Named)
        {
            var progress = Of(course, cells, catalog);

            if (progress.Follows && progress.Matched > most)
            {
                followed = course;
                most = progress.Matched;
            }
        }

        return followed;
    }

    /// <summary>Whether a course is offered to a notebook: it follows the course, has steps to come, and no other says more.</summary>
    /// <param name="course">The course.</param>
    /// <param name="cells">The notebook's cells, in order.</param>
    /// <param name="catalog">The verbs a notebook knows.</param>
    /// <returns>
    /// <see langword="true"/> when the blocks follow the course and it has a step the blocks do not say, and no course that
    /// the blocks follow as well says more of them: a notebook that orders its rows is a series', not a table's.
    /// </returns>
    public static bool IsOfferedFor(PipelineCourse course, IReadOnlyList<CellModel> cells, StepCatalog catalog)
    {
        var progress = Of(course, cells, catalog);

        return progress.Follows
            && progress.Missing.Count > 0
            && PipelineCourse.Named
                .Select(other => Of(other, cells, catalog))
                .Where(other => other.Follows)
                .All(other => other.Matched <= progress.Matched);
    }

    /// <summary>The step of a course that belongs at a place among a notebook's blocks.</summary>
    /// <param name="course">The course the blocks follow.</param>
    /// <param name="cells">The notebook's cells, in order, as they stand before a block is added.</param>
    /// <param name="at">The place the block goes: how many cells stand above it.</param>
    /// <param name="catalog">The verbs a notebook knows.</param>
    /// <returns>
    /// The first step the blocks above the place do not say, so long as the blocks below it do not already stand past it;
    /// nothing when the blocks have gone their own way, or no step belongs in that place.
    /// </returns>
    /// <remarks>
    /// A block put between two others has to take the place the course gives it there, and not the one at the end: the
    /// step that follows the last block of the notebook, written above blocks that stand behind it, would take the notebook
    /// off its course.
    /// </remarks>
    public static CourseStep? NextAt(PipelineCourse course, IReadOnlyList<CellModel> cells, int at, StepCatalog catalog)
    {
        if (!Of(course, cells, catalog).Follows)
        {
            return null;
        }

        var above = Of(course, [.. cells.Take(at)], catalog);

        return above.Missing is [var next, ..] && above.NextRow < FirstRowOf(course, [.. cells.Skip(at)], catalog) ? next : null;
    }

    // The place in the course of the first block that moves the course on, behind a place; none, as a place nothing can stand
    // before, when there is no such block.
    private static int FirstRowOf(PipelineCourse course, IReadOnlyList<CellModel> cells, StepCatalog catalog)
    {
        var kinds = new Kinds(catalog);

        foreach (var cell in cells.Where(cell => cell.Type == StepCellType.StepType))
        {
            var verb = StepText.Of(cell.Source).Verb;
            var row = verb is not null && catalog.Knows(verb) ? RowOf(course, verb, kinds) : -1;

            if (row >= 0 && !kinds.Of(verb!).Drops)
            {
                return row;
            }
        }

        return int.MaxValue;
    }

    // The place in the course of the step a verb says; nothing, as minus one, when the course has no such step.
    private static int RowOf(PipelineCourse course, string verb, Kinds kinds)
    {
        for (var at = 0; at < course.Steps.Count; at++)
        {
            if (kinds.Stands(course.Steps[at].Verb, verb))
            {
                return at;
            }
        }

        return -1;
    }

    /// <summary>What a verb is to a course: the kind of step a declaration holds one of that it is, and whether it takes columns away.</summary>
    /// <param name="Kind">The kind, when the verb is of one a pipeline holds only one of and a course may stand another verb for.</param>
    /// <param name="Drops">Whether it takes columns away.</param>
    private readonly record struct VerbKind(Type? Kind, bool Drops);

    // The verbs of one reading of a notebook, each read as the step its template is only once. Every verb asked of is one the
    // notebook knows: a block's is checked before it is asked, and the steps of the named courses are held to it by a test.
    private sealed class Kinds(StepCatalog catalog)
    {
        private readonly Dictionary<string, VerbKind> _read = new(StringComparer.Ordinal);

        public VerbKind Of(string verb)
        {
            if (!_read.TryGetValue(verb, out var kind))
            {
                var step = catalog.ReadStep(catalog.Describe(verb).Template);

                kind = new VerbKind(Interchangeable.FirstOrDefault(each => each.IsInstanceOfType(step)), step is IDropsColumns);
                _read[verb] = kind;
            }

            return kind;
        }

        public bool Stands(string rowVerb, string verb) =>
            rowVerb == verb || (Of(rowVerb).Kind is { } kind && kind == Of(verb).Kind);
    }
}
