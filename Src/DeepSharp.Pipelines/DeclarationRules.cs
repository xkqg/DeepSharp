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
/// Every step does something the run acts on, except an output that only names the answer, a report that only names
/// what a model's answers are measured by, and a step that only names the learner the pipeline is declared for.
/// </summary>
/// <remarks>
/// A step with no acting capability used to be carried along and ignored: it was in the file and in the
/// chain, and in nothing the pipeline did. Two capabilities on one step cannot compile outside this library,
/// so what this rule has left to find is a step that does nothing. Naming the answer, naming its measures and naming the
/// learner change no row, and each is read by what comes after the pipeline: the handover, the measures of a trained
/// model, and the package that trains it.
/// </remarks>
internal sealed class EveryStepActs : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is not (IActsInAWalk or INamesTheAnswer or INamesTheMeasures or INamesTheLearner))
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
/// A column whose gaps were settled is not settled or filled again.
/// </summary>
/// <remarks>
/// Settling destroys the difference between absent and measured, permanently: afterwards the column holds no gap, so a
/// second verb reaching for the same column has nothing to do. The marking column keeps saying where the gaps were,
/// because a later marking adds to the first rather than replacing it, but the line itself never acts — and a line that
/// never acts is a person who meant one of the two and wrote both. Said at the declaration, where the line is, rather
/// than left to a run that quietly does nothing.
/// </remarks>
internal sealed class ASettledColumnIsNotFilledAgain : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        ArgumentNullException.ThrowIfNull(steps);

        var settled = new Dictionary<string, int>(StringComparer.Ordinal);

        for (var at = 0; at < steps.Count; at++)
        {
            if (Column(steps[at]) is not { } column)
            {
                continue;
            }

            if (settled.TryGetValue(column, out var first))
            {
                yield return new DeclarationFault(
                    at, steps[at].Verb,
                    $"would fill the gaps in '{column}', and settle.gaps at step {first + 1} already settled them: "
                    + "there is no gap left for it to find. Keep the settling, or keep the fill and settle another column.");
            }
            else if (steps[at] is SettleGapsStep)
            {
                settled[column] = at;
            }
        }
    }

    // The column a verb would put a value into, for the two verbs that fill a gap; nothing for every other step.
    private static string? Column(IPipelineStep step) => step switch
    {
        SettleGapsStep settle => settle.Column,
        FillMissingStep fill => fill.Column,
        _ => null,
    };
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
        ColumnFlow.Follow(steps, ColumnState.None, .., declared: false).Faults;
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
/// A report stands below the output whose answers it measures.
/// </summary>
/// <remarks>
/// What it measures are that output's answers, so without an output it measures nothing, and above one it would name
/// measures of an answer not named yet where it stands.
/// </remarks>
internal sealed class AReportStandsBelowItsOutput : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        var output = -1;

        for (var at = 0; at < steps.Count && output < 0; at++)
        {
            if (steps[at] is INamesTheAnswer)
            {
                output = at;
            }
        }

        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is not INamesTheMeasures || (output >= 0 && output < at))
            {
                continue;
            }

            yield return new DeclarationFault(
                at, steps[at].Verb,
                output < 0
                    ? "names the measures of a model's answers, and this pipeline names no answer: a report stands below the output whose answers it measures."
                    : $"measures the answers the output at step {output + 1} names, and stands above it: a report stands below the output whose answers it measures.");
        }
    }
}

/// <summary>
/// A learner stands below the output it learns to answer.
/// </summary>
/// <remarks>
/// What it learns is that output's answers, so without an output it learns nothing, and above one it would name a learner
/// for an answer not named yet where it stands.
/// </remarks>
internal sealed class ALearnerStandsBelowItsOutput : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        var output = -1;

        for (var at = 0; at < steps.Count && output < 0; at++)
        {
            if (steps[at] is INamesTheAnswer)
            {
                output = at;
            }
        }

        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is not INamesTheLearner || (output >= 0 && output < at))
            {
                continue;
            }

            yield return new DeclarationFault(
                at, steps[at].Verb,
                output < 0
                    ? "names the learner this pipeline is declared for, and this pipeline names no answer: a learner stands below the output it learns to answer."
                    : $"learns the answers the output at step {output + 1} names, and stands above it: a learner stands below the output it learns to answer.");
        }
    }
}

/// <summary>
/// A report measures the parts a split makes, so a pipeline with a report divides its rows.
/// </summary>
/// <remarks>
/// Rows nothing divides are neither the rows a model learns from nor the rows it is chosen or tested on, so a report over
/// them would name measures no part could ever be measured by.
/// </remarks>
internal sealed class AReportMeasuresDividedRows : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        if (NothingLearnsBeforeTheSplit.FirstSplit(steps) >= 0)
        {
            yield break;
        }

        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is INamesTheMeasures)
            {
                yield return new DeclarationFault(
                    at, steps[at].Verb,
                    "measures a model on the parts a split divides the rows into, and this pipeline never divides them: split the rows above it.");
            }
        }
    }
}

/// <summary>
/// A measure that counts classes is named only where the output's answers can be classes.
/// </summary>
/// <remarks>
/// A share of a whole and a return are amounts: an accuracy of shares would count how often a share came out at exactly
/// nought or one, which says nothing about the model. The output says whether its answers can be classes, so an output
/// another package brings says it too.
/// </remarks>
internal sealed class ClassesAreCountedWhereTheAnswersAreClasses : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        if (steps.OfType<INamesTheAnswer>().FirstOrDefault() is not { } output || output.Takes(MetricFamily.Classes))
        {
            yield break;
        }

        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is not INamesTheMeasures report || !report.Metrics.Any(MetricExtensions.CountsClasses))
            {
                continue;
            }

            yield return new DeclarationFault(
                at, steps[at].Verb,
                $"counts classes with {report.Metrics.Where(MetricExtensions.CountsClasses).Listed()}, and the answers '{output.Verb}' "
                + $"names are amounts, not classes of nought or one: measure them with rmse, mae or r2{output.SharesMeasured()}.");
        }
    }
}

/// <summary>
/// A measure of shares is named only where the output's answers are shares of a whole, and one that follows their order
/// only where they are in one.
/// </summary>
/// <remarks>
/// A divergence of a price from a prediction, or a distance along an order the bands do not have, is a number that means
/// nothing; the output says whether its answers are shares in an order, so an output another package brings says it too.
/// </remarks>
internal sealed class SharesAreMeasuredWhereTheAnswersAreShares : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        if (steps.OfType<INamesTheAnswer>().FirstOrDefault() is not { } output)
        {
            yield break;
        }

        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is not INamesTheMeasures report)
            {
                continue;
            }

            if (report.Metrics.Where(metric => metric.Family() == MetricFamily.OrderedShares).ToArray() is [_, ..] ordered && !output.Takes(MetricFamily.OrderedShares))
            {
                yield return new DeclarationFault(
                    at, report.Verb,
                    $"measures shares along their order with {ordered.Listed()}, and the answers '{output.Verb}' names are no shares in an order: "
                    + "they are when 'target.distribution' says its columns are, with 'ordered': true.");
            }

            if (report.Metrics.Where(metric => metric.Family() == MetricFamily.Shares).ToArray() is [_, ..] shares && !output.Takes(MetricFamily.Shares))
            {
                yield return new DeclarationFault(
                    at, report.Verb,
                    $"measures shares with {shares.Listed()}, and the answers '{output.Verb}' names are no shares of a whole: "
                    + "name the columns a whole is divided among with 'target.distribution'.");
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
                if (Changes(steps[above], ahead.Column))
                {
                    yield return new DeclarationFault(
                        at, ahead.Verb,
                        $"is a return on '{ahead.Column}' as it was read, and step {above + 1}, '{steps[above].Verb}', changes "
                        + $"'{ahead.Column}' above it. Put the return above that step.");
                }
            }
        }
    }

    /// <summary>Whether a step changes a column where it stands: learns from it in place, or has a way back for it.</summary>
    /// <param name="step">The step.</param>
    /// <param name="column">The column.</param>
    /// <returns><see langword="true"/> when the column is not as it was read below the step.</returns>
    /// <remarks>One rule for every answer made from columns as they were read.</remarks>
    internal static bool Changes(IPipelineStep step, string column) => step switch
    {
        IFittedStep fitted => fitted.ColumnsRead.Any(read => read.Column == column),
        IUndoesItself undoes => undoes.Undoes(column),
        _ => false,
    };
}

/// <summary>
/// What is left of a distribution's whole is made from its counts as they were read: no step above it changes the bands or
/// how many there were.
/// </summary>
/// <remarks>
/// Each band's share and what is left are the counts over the whole, and they come back as birds by the whole as it was
/// read. Made from a scaled whole, they would come back as birds that were never there.
/// </remarks>
internal sealed class ARemainderIsMadeFromItsCountsAsRead : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        for (var at = 0; at < steps.Count; at++)
        {
            // A remainder is named only with the whole it is left of, which the output refuses to be without.
            if (steps[at] is not DistributionStep { Remainder: { } remainder } distribution)
            {
                continue;
            }

            for (var above = 0; above < at; above++)
            {
                if (distribution.Columns.Append(distribution.ScaleBy!).FirstOrDefault(column => AReturnIsMadeFromItsColumnAsRead.Changes(steps[above], column)) is { } changed)
                {
                    yield return new DeclarationFault(
                        at, distribution.Verb,
                        $"makes '{remainder}' from the counts as they were read, and step {above + 1}, '{steps[above].Verb}', changes "
                        + $"'{changed}' above it. Put the output above that step.");
                }
            }
        }
    }
}

/// <summary>
/// The column the schema says names each row is not an answer.
/// </summary>
/// <remarks>
/// An id is carried beside the rows and never shown to a model, so asking a model for it would ask for what no row taught
/// it: refused where the output names it, by name.
/// </remarks>
internal sealed class TheIdIsNoAnswer : IDeclarationRule
{
    public IEnumerable<DeclarationFault> FaultsIn(IReadOnlyList<IPipelineStep> steps)
    {
        if (steps.OfType<DeclareStep>().FirstOrDefault()?.IdColumn is not { } id)
        {
            yield break;
        }

        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is INamesTheAnswer output && output.Answers.Contains(id, StringComparer.Ordinal))
            {
                yield return new DeclarationFault(
                    at, steps[at].Verb,
                    $"names '{id}' as an answer, and the schema says '{id}' is the id: an id names each row and is carried beside it, "
                    + "never asked for. Name another column, or say the id is another.");
            }
        }
    }
}
