// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// One thing wrong with a declaration, and the step it is wrong at.
/// </summary>
/// <param name="At">The step's place in the declaration, counting from nought as <see cref="PipelineDeclaration.Steps"/> does.</param>
/// <param name="Verb">The verb of the step at that place.</param>
/// <param name="Message">What is wrong, in the words a person reads it in.</param>
/// <remarks>
/// A declaration reports every fault it has at once rather than the first one it meets: somebody handed one
/// fault at a time, five times over, stops using the thing. The place is what lets a notebook put the fault
/// on the block it belongs to.
/// </remarks>
public readonly record struct DeclarationFault(int At, string Verb, string Message)
{
    /// <summary>The fault as it is read: which step, counting from one, and what is wrong with it.</summary>
    /// <returns>The step, its verb and the message.</returns>
    public override string ToString() => $"Step {At + 1}, '{Verb}': {Message}";
}

/// <summary>
/// A declaration that breaks a rule every declaration keeps, or does not fit the rows it is run over, with
/// every fault it has.
/// </summary>
/// <remarks>
/// An <see cref="InvalidOperationException"/>, because that is what a declaration refusing its steps has
/// always thrown; the faults come with it as data, so a notebook can put each one on the block it belongs to
/// and a file door can put each one at its line, without reading the message apart.
/// </remarks>
public sealed class DeclarationException : InvalidOperationException
{
    /// <summary>A refusal carrying every fault the steps have.</summary>
    /// <param name="faults">The faults, in the order of the steps they are at.</param>
    public DeclarationException(IReadOnlyList<DeclarationFault> faults)
        : base(string.Join(Environment.NewLine, faults ?? throw new ArgumentNullException(nameof(faults)))) =>
        Faults = faults;

    /// <summary>Every fault, in the order of the steps they are at.</summary>
    public IReadOnlyList<DeclarationFault> Faults { get; }
}

/// <summary>
/// Something every declaration has to be true of, whichever door it came through.
/// </summary>
internal interface IDeclarationRule
{
    /// <summary>Every place these steps break the rule.</summary>
    /// <param name="steps">The steps, in the order they were written.</param>
    /// <returns>One fault per step that breaks it; none when the steps keep it.</returns>
    IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps);
}

/// <summary>
/// A declaration has at most one step of a kind: one source, one schema, one split, one output.
/// </summary>
/// <typeparam name="TStep">The kind there may be only one of.</typeparam>
/// <param name="what">What the kind is called, for the message.</param>
/// <remarks>
/// A second one was never an error before; it was ignored. The run took the first of each and the rest
/// were inert, so a file said one thing and the numbers came from another — and a second target was worse
/// than inert, because the first one was then handed to the model as a feature.
/// </remarks>
internal sealed class AtMostOne<TStep>(string what) : IDeclarationRule
    where TStep : IPipelineStep
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        var first = -1;

        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is not TStep)
            {
                continue;
            }

            if (first < 0)
            {
                first = at;
                continue;
            }

            yield return new DeclarationFault(
                at, steps[at].Verb,
                $"a pipeline has one {what}, and this is a second one; the first is '{steps[first].Verb}' at step {first + 1}.");
        }
    }
}

/// <summary>
/// Every step does something the run acts on, except an output that only names the answer.
/// </summary>
/// <remarks>
/// A step with no acting capability used to be carried along and ignored: it was in the file and in the
/// chain, and in nothing the pipeline did. Two capabilities on one step cannot compile outside this library,
/// so what this rule has left to find is a step that does nothing.
/// </remarks>
internal sealed class EveryStepActs : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is not (IActsInAWalk or INamesTheAnswer))
            {
                yield return new DeclarationFault(
                    at, steps[at].Verb,
                    "does nothing a pipeline runs: it opens no rows, declares no columns, adds, drops or orders "
                    + "nothing, divides nothing and learns nothing. A step says what it does by what it implements.");
            }
        }
    }
}

/// <summary>
/// The source, when a declaration names one, is its first step.
/// </summary>
/// <remarks>
/// Every other step works on the rows the source opens. A declaration that hands its rows in at run time
/// has no source at all, and starts at its schema.
/// </remarks>
internal sealed class TheSourceComesFirst : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        var source = -1;

        for (var at = 0; at < steps.Count && source < 0; at++)
        {
            if (steps[at] is IOpensRows)
            {
                source = at;
            }
        }

        if (source > 0)
        {
            yield return new DeclarationFault(
                source, steps[source].Verb,
                "is where the rows come from, so it is the first step: nothing can work on rows before they are read.");
        }
    }
}

/// <summary>
/// The columns are declared directly after the source, and every other step stands after them.
/// </summary>
/// <remarks>
/// Everything after the source works on columns, and there are none until the schema says which. A step
/// written above the schema used to fail a whole run later, as a column nobody could find.
/// </remarks>
internal sealed class ColumnsAreDeclaredFirst : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        var declared = -1;

        for (var at = 0; at < steps.Count && declared < 0; at++)
        {
            if (steps[at] is IBindsColumns)
            {
                declared = at;
            }
        }

        for (var at = 0; at < steps.Count; at++)
        {
            // Sources are the rule above's to place, and a second schema is a fault of its own.
            if (steps[at] is IOpensRows or IBindsColumns || (declared >= 0 && at > declared))
            {
                continue;
            }

            yield return new DeclarationFault(
                at, steps[at].Verb,
                declared < 0
                    ? "works on columns, and no step declares any. The schema, 'declare', comes directly after the source."
                    : $"works on columns, and none are declared until step {declared + 1}. The schema, 'declare', comes directly after the source.");
        }
    }
}

/// <summary>
/// Nothing that learns from the data stands before the rows are split.
/// </summary>
/// <remarks>
/// The rule the whole library exists to keep. It arrives through three doors — the chain, the extension
/// point every other package uses, and a file somebody edited by hand — and the declaration is the only
/// place all three pass through.
/// </remarks>
internal sealed class NothingLearnsBeforeTheSplit : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        var split = FirstSplit(steps);

        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is not IFittedStep || (split >= 0 && split < at))
            {
                continue;
            }

            yield return new DeclarationFault(
                at, steps[at].Verb,
                "learns from the data, so it cannot stand before the rows are split. "
                + (split < 0
                    ? "This declaration never splits them."
                    : $"The split at step {split + 1} comes after it."));
        }
    }

    internal static int FirstSplit(IReadOnlyList<IPipelineStep> steps)
    {
        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is ISplitStep)
            {
                return at;
            }
        }

        return -1;
    }
}

/// <summary>
/// Every column a step reads is there where it reads it, of a kind it can work on.
/// </summary>
/// <remarks>
/// The columns are followed from the schema down, each step saying what it leaves behind. A column the schema
/// left out, or a step above took away, used to fail a whole run later as a column nobody could find; it is
/// refused where it is read, naming the step. The same walk refuses what only the columns can decide — an
/// encoder with no category to encode, a step after the output that takes one of its answers away.
/// </remarks>
internal sealed class ColumnsAreThereWhereTheyAreRead : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps) =>
        ColumnFlow.Follow(steps, ColumnState.None, from: 0, until: steps.Count, declared: false).Faults;
}

/// <summary>
/// Rows are dropped and put in order before they are divided, never after.
/// </summary>
/// <remarks>
/// A split divides the rows it is given, once. Dropping some of them afterwards changes what each part holds
/// without the parts knowing, and putting them in another order moves rows under steps that learned from them
/// in the first — a run and a replay then disagree about which rows there are.
/// </remarks>
internal sealed class RowsAreSettledBeforeTheSplit : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        var split = NothingLearnsBeforeTheSplit.FirstSplit(steps);

        for (var at = split + 1; split >= 0 && at < steps.Count; at++)
        {
            if (steps[at] is IDropsRows or IOrdersRows)
            {
                yield return new DeclarationFault(
                    at, steps[at].Verb,
                    $"rows are {(steps[at] is IDropsRows ? "dropped" : "put in order")} before they are divided, and the split at step {split + 1} has already divided them.");
            }
        }
    }
}

/// <summary>
/// A step that reads the rows in their order stands below the step that declares it.
/// </summary>
/// <remarks>
/// The rows before a row are whichever the file put there, unless an order is declared above: the same
/// prices reversed gave a five-day average of 98.352 where the right one is 99.74.
/// </remarks>
internal sealed class RowOrderIsDeclaredBeforeItIsRead : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        var ordered = false;

        for (var at = 0; at < steps.Count; at++)
        {
            ordered |= steps[at] is IOrdersRows;

            if (steps[at] is IReadsRowOrder && !ordered)
            {
                yield return new DeclarationFault(
                    at, steps[at].Verb,
                    "reads the rows in their order, and nothing above it says what that order is. "
                    + "Put them in order first, with 'order.by'.");
            }
        }
    }
}

/// <summary>
/// An output that acts on the rows makes its answer from them.
/// </summary>
/// <remarks>
/// Naming the answer is not doing something to the data; the one thing an output may do is make its answer, and it
/// says so by making it. An output acting on the rows any other way would change the data a model is shown under the
/// name of saying what it is asked.
/// </remarks>
internal sealed class AnActingOutputMakesItsAnswer : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is INamesTheAnswer and IActsInAWalk and not IMakesTheAnswer)
            {
                yield return new DeclarationFault(
                    at, steps[at].Verb,
                    "is an output that acts on the rows, and the one thing an output does is name its answer or make it: "
                    + "an output that acts makes its answer.");
            }
        }
    }
}

/// <summary>
/// Only an output reads rows after its own.
/// </summary>
/// <remarks>
/// A feature that knows the future is a leak in mathematical dress: it scores beautifully on every row it is measured
/// on, and on no row it is ever asked about, since those have no later rows yet.
/// </remarks>
internal sealed class OnlyAnOutputReadsAhead : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is IReadsRowsAhead and not INamesTheAnswer)
            {
                yield return new DeclarationFault(
                    at, steps[at].Verb,
                    "reads rows after the row it is on, and only an output may: a feature that knows the future is a leak "
                    + "that scores well on every row it is measured on and on none it is asked about.");
            }
        }
    }
}

/// <summary>
/// An answer read from later rows stands below a split in time whose gap is at least as wide, the rows ordered by the
/// column that split divides by, alone.
/// </summary>
/// <remarks>
/// Without the gap, the last rows a model learns from read their answers from the rows it is measured on. Later rows
/// are later in time only when the rows are ordered by the column the split divides by and nothing else.
/// </remarks>
internal sealed class RowsAheadAreKeptApart : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        var split = NothingLearnsBeforeTheSplit.FirstSplit(steps);
        var order = steps.OfType<IOrdersRows>().FirstOrDefault();

        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is not IReadsRowsAhead ahead)
            {
                continue;
            }

            if (split < 0 || split > at || steps[split] is not IDividesInTime time)
            {
                yield return new DeclarationFault(
                    at, ahead.Verb,
                    $"reads {ahead.Ahead} rows ahead, so the rows it learns from and the rows it is measured on are divided in time "
                    + $"above it, with a gap of at least {ahead.Ahead}: split.byTime.");
                continue;
            }

            if (time.Gap < ahead.Ahead)
            {
                yield return new DeclarationFault(
                    at, ahead.Verb,
                    $"reads {ahead.Ahead} rows ahead, and the split at step {split + 1} keeps a gap of {time.Gap}: the last rows a "
                    + $"model learns from would read their answers from the rows it is measured on. Keep a gap of at least {ahead.Ahead}.");
                continue;
            }

            if (order?.OrderedBy is not [var only] || only != time.Column)
            {
                yield return new DeclarationFault(
                    at, ahead.Verb,
                    $"reads rows ahead in the order the rows stand in, and the split divides them by '{time.Column}': order them "
                    + $"by '{time.Column}' alone, so the rows after a row are the ones that came after it.");
            }
        }
    }
}

/// <summary>
/// A return is made from its column as it was read: no step above it changes that column.
/// </summary>
/// <remarks>
/// A return comes back as a value by the row's own value as it was read, so the return has to be on that value. Made
/// from a scaled price, it came back as a price that was never there.
/// </remarks>
internal sealed class AReturnIsMadeFromItsColumnAsRead : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is not AheadStep { IsMadeFromItsColumnAsRead: true } ahead)
            {
                continue;
            }

            for (var above = 0; above < at; above++)
            {
                var changes = steps[above] switch
                {
                    IFittedStep fitted => fitted.ColumnsRead.Any(read => read.Column == ahead.Column),
                    IUndoesItself undoes => undoes.Undoes(ahead.Column),
                    _ => false,
                };

                if (changes)
                {
                    yield return new DeclarationFault(
                        at, ahead.Verb,
                        $"is a return on '{ahead.Column}' as it was read, and step {above + 1}, '{steps[above].Verb}', changes "
                        + $"'{ahead.Column}' above it. Put the return above that step.");
                }
            }
        }
    }
}
