// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// One column known to be there at a step: its name, what it holds, and whether it is surely there.
/// </summary>
/// <param name="Name">The column's name.</param>
/// <param name="Kind">What it holds.</param>
/// <param name="Surely">
/// Whether it is there whatever the rows say: a column the schema allows to be absent is known, and may still
/// turn out not to be there when the rows are read.
/// </param>
/// <param name="HalfOf">
/// The signed value this column is one half of, when a split by sign made it; nothing otherwise. A half is the
/// last form its value takes, so a step that would scale it on its own asks for this.
/// </param>
public readonly record struct KnownColumn(string Name, ColumnKind Kind, bool Surely, string? HalfOf = null);

/// <summary>
/// A column a step reads, and the kinds of column it can work on.
/// </summary>
/// <param name="Column">The column's name.</param>
/// <param name="Accepts">The kinds the step can work on.</param>
/// <remarks>Taken from the step's parameters that name columns, and from nowhere else: there is no second list to keep in step.</remarks>
public readonly record struct ColumnRead(string Column, IReadOnlyList<ColumnKind> Accepts);

/// <summary>
/// Which columns there are at one step of a declaration, known before anything runs.
/// </summary>
/// <remarks>
/// Followed from the schema down, each step saying what it leaves behind. Three things are known besides the
/// columns themselves. Whether other columns may be there too — a schema that keeps the rest, or a step that
/// does not say what it leaves behind, opens the set. The families an encoder makes, whose members are known by
/// the start of their name alone until the training rows have been seen, with any member dropped by name
/// remembered as gone. And which columns may turn out not to be there once the rows are read.
/// </remarks>
public sealed class ColumnState
{
    private readonly KnownColumn[] _columns;
    private readonly string[] _families;
    private readonly string[] _gone;
    private readonly Dictionary<string, string> _conditions;

    private ColumnState(KnownColumn[] columns, string[] families, string[] gone, bool open, Dictionary<string, string>? conditions = null)
    {
        _columns = columns;
        _families = families;
        _gone = gone;
        Open = open;
        _conditions = conditions ?? new Dictionary<string, string>(StringComparer.Ordinal);
    }

    /// <summary>Nothing known, and nothing else there: the state before the columns are declared.</summary>
    public static ColumnState None { get; } = new([], [], [], open: false);

    /// <summary>The columns a table holds, each surely there.</summary>
    /// <param name="table">The table.</param>
    /// <returns>Its columns, in order.</returns>
    public static ColumnState Of(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        return new([.. table.Columns.Select(column => new KnownColumn(column.Name, column.Kind, Surely: true))], [], [], open: false);
    }

    /// <summary>Why a known column may turn out not to be there, when it may.</summary>
    /// <param name="name">The column's name.</param>
    /// <returns>The reason, or nothing when the column is surely there or not known at all.</returns>
    public string? WhyItMayBeGone(string name) => _conditions.GetValueOrDefault(name);

    /// <summary>The columns known to be there, in the order they stand.</summary>
    public IReadOnlyList<KnownColumn> Columns => _columns;

    /// <summary>The starts of the names of columns an encoder makes, whose members only a fit knows.</summary>
    public IReadOnlyList<string> Families => _families;

    /// <summary>Whether columns nobody named may be there too.</summary>
    public bool Open { get; }

    /// <summary>The column of that name, when it is known to be there.</summary>
    /// <param name="name">The column's name.</param>
    /// <returns>The column, or nothing.</returns>
    public KnownColumn? Find(string name)
    {
        foreach (var column in _columns)
        {
            if (column.Name == name)
            {
                return column;
            }
        }

        return null;
    }

    /// <summary>Whether a step may read a column of that name here.</summary>
    /// <param name="name">The column's name.</param>
    /// <returns>
    /// <see langword="true"/> when it is known, or it was not taken away by name and belongs to a family or the
    /// set is open.
    /// </returns>
    public bool Allows(string name) =>
        Find(name) is not null || (!_gone.Contains(name, StringComparer.Ordinal) && (InAFamily(name) || Open));

    /// <summary>Whether a name belongs to one of the families, and was not taken away by name.</summary>
    /// <param name="name">The column's name.</param>
    /// <returns><see langword="true"/> when it is a member nobody dropped.</returns>
    public bool InAFamily(string name) =>
        !_gone.Contains(name, StringComparer.Ordinal)
        && _families.Any(family => name.StartsWith(family, StringComparison.Ordinal));

    /// <summary>The same columns, with one added at the end or, when it is there, holding another kind in its place.</summary>
    /// <param name="name">The column's name.</param>
    /// <param name="kind">What it holds from here on.</param>
    /// <returns>The state with the column.</returns>
    public ColumnState With(string name, ColumnKind kind)
    {
        var at = Array.FindIndex(_columns, column => column.Name == name);

        KnownColumn[] columns = at < 0
            ? [.. _columns, new KnownColumn(name, kind, Surely: true)]
            : [.. _columns[..at], _columns[at] with { Kind = kind }, .. _columns[(at + 1)..]];

        return new(columns, _families, [.. _gone.Where(each => each != name)], Open, _conditions);
    }

    /// <summary>The same columns, with one added as a half of a signed value split by its sign.</summary>
    /// <param name="name">The half's name.</param>
    /// <param name="of">The signed value it is one half of.</param>
    /// <returns>The state with the half, a number.</returns>
    /// <exception cref="ArgumentException">The value it is half of is not named.</exception>
    /// <remarks>
    /// A step writing the column again later — a fill, say — leaves it a half: what it is half of does not change
    /// by writing its gaps.
    /// </remarks>
    public ColumnState WithHalf(string name, string of)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(of);

        var state = With(name, ColumnKind.Number);

        return new(
            [.. state._columns.Select(column => column.Name == name ? column with { HalfOf = of } : column)],
            state._families, state._gone, state.Open, state._conditions);
    }

    /// <summary>The same columns without one, remembered as gone: no family and no open set brings it back.</summary>
    /// <param name="name">The column's name.</param>
    /// <returns>The state without the column.</returns>
    public ColumnState Without(string name) =>
        new([.. _columns.Where(column => column.Name != name)], _families, [.. _gone, name], Open, Except(name));

    /// <summary>The same columns, one of them no longer surely there, and why.</summary>
    /// <param name="name">The column's name.</param>
    /// <param name="why">What decides whether it is there, in the words of the step that decides it.</param>
    /// <returns>The state with the column known and not sure.</returns>
    /// <remarks>
    /// A step may read it, since it may well be there; when the rows are run and it is not, the refusal says
    /// why, in these words.
    /// </remarks>
    public ColumnState MaybeGone(string name, string why)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(why);

        var conditions = new Dictionary<string, string>(_conditions, StringComparer.Ordinal) { [name] = why };

        return new([.. _columns.Select(column => column.Name == name ? column with { Surely = false } : column)], _families, _gone, Open, conditions);
    }

    /// <summary>The same columns, and a family whose members are known by the start of their names.</summary>
    /// <param name="start">The start every member's name has.</param>
    /// <returns>The state with the family.</returns>
    public ColumnState WithFamily(string start) => new(_columns, [.. _families, start], _gone, Open, _conditions);

    /// <summary>The same columns, with others nobody named allowed beside them.</summary>
    /// <returns>The state, open.</returns>
    public ColumnState Opened() => new(_columns, _families, _gone, open: true, _conditions);

    /// <summary>The columns a schema declares, with the rest kept or not.</summary>
    /// <param name="columns">The declared columns.</param>
    /// <param name="remainder">What becomes of the columns the schema does not name.</param>
    /// <returns>The state after the schema.</returns>
    internal static ColumnState Declared(IEnumerable<ColumnDeclaration> columns, Remainder remainder)
    {
        ColumnDeclaration[] declared = [.. columns];

        return new(
            [.. declared.Select(column => new KnownColumn(column.Name, column.Kind, !column.Optional))],
            [],
            [],
            remainder == Remainder.Keep,
            declared.Where(column => column.Optional).ToDictionary(
                column => column.Name, _ => "the schema allows the rows not to have it", StringComparer.Ordinal));
    }

    private Dictionary<string, string> Except(string name) =>
        _conditions.Where(each => each.Key != name).ToDictionary(StringComparer.Ordinal);
}

/// <summary>
/// A step that says which columns it leaves behind.
/// </summary>
/// <remarks>
/// What it leaves behind and nothing more: which columns a step reads is said by the parameters that name
/// columns. Every step this library ships says it, so a declaration can follow the columns from the schema
/// down; a step that does not leaves any column possible from there on.
/// </remarks>
public interface IDescribesColumns : IPipelineStep
{
    /// <summary>The columns there are after this step, given the ones there were before it.</summary>
    /// <param name="before">The columns before it.</param>
    /// <returns>The columns after it.</returns>
    ColumnState After(ColumnState before);

    /// <summary>Why this step cannot stand where the columns are what they are, when it cannot.</summary>
    /// <param name="before">The columns before it.</param>
    /// <returns>The reason, or nothing when it can stand there.</returns>
    /// <remarks>Nothing, for most steps: whether the columns they read are there is asked of every step alike.</remarks>
    string? Refusal(ColumnState before) => null;
}

/// <summary>
/// Following the columns through a declaration: from the schema down, the columns there are at each step, and
/// every step that reads a column that is not there.
/// </summary>
/// <remarks>
/// One walk, used twice: when a declaration is made, from the columns the schema declares; and when rows are
/// read, from the columns the rows turned out to have — so a column the schema allowed to be absent, and the
/// rows lack, is refused at every step that reads it before any of them runs.
/// </remarks>
internal static class ColumnFlow
{
    /// <summary>Follows the columns from one step to another.</summary>
    /// <param name="steps">The steps, in the order they were written.</param>
    /// <param name="start">The columns before the first step followed.</param>
    /// <param name="from">The first step to follow.</param>
    /// <param name="until">The step to stop before.</param>
    /// <param name="declared">Whether the columns are already declared before the first step followed.</param>
    /// <returns>The columns there are before the step it stopped at, and every fault met on the way.</returns>
    internal static ColumnsFollowed Follow(IReadOnlyList<IPipelineStep> steps, ColumnState start, int from, int until, bool declared)
    {
        var faults = new List<DeclarationFault>();
        var state = start;
        string? target = null;

        for (var at = from; at < until; at++)
        {
            var step = steps[at];

            // A step above the schema is the schema rule's to name; one mistake is one fault.
            declared |= step is IBindsColumns;

            if (!declared)
            {
                continue;
            }

            if (step is not IBindsColumns)
            {
                faults.AddRange(Missing(state, step, at));
            }

            if (step is IDescribesColumns describes && describes.Refusal(state) is { } refusal)
            {
                faults.Add(new DeclarationFault(at, step.Verb, refusal));
            }

            var after = step is IDescribesColumns said ? said.After(state) : state.Opened();

            if (target is not null && state.Allows(target) && !after.Allows(target))
            {
                faults.Add(new DeclarationFault(
                    at, step.Verb,
                    $"takes away '{target}', the column the target above it names; nothing after the target may."));
            }

            target = step is TargetStep named ? named.Column : target;
            state = after;
        }

        return new ColumnsFollowed(state, faults);
    }

    private static IEnumerable<DeclarationFault> Missing(ColumnState state, IPipelineStep step, int at)
    {
        foreach (var read in step.ColumnsRead)
        {
            if (state.Find(read.Column) is { } known)
            {
                if (!read.Accepts.Contains(known.Kind))
                {
                    yield return new DeclarationFault(
                        at, step.Verb,
                        $"reads '{read.Column}', which holds {known.Kind.ToString().ToLowerInvariant()}, and it works on "
                        + $"{string.Join(" or ", read.Accepts.Select(kind => kind.ToString().ToLowerInvariant()))}.");
                }

                continue;
            }

            if (!state.Allows(read.Column))
            {
                yield return new DeclarationFault(
                    at, step.Verb,
                    $"reads '{read.Column}', and no column of that name is here: the schema does not declare it, no "
                    + "step above makes it, or a step above took it away.");
            }
        }
    }
}

/// <summary>What following the columns found: the columns at the end, and every fault on the way.</summary>
/// <param name="State">The columns there are where the walk stopped.</param>
/// <param name="Faults">Every step that reads a column that is not there, or cannot stand where it is.</param>
internal readonly record struct ColumnsFollowed(ColumnState State, IReadOnlyList<DeclarationFault> Faults);
