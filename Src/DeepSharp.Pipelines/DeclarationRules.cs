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
