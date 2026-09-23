// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// A pipeline being written, before it has been told how to split.
/// </summary>
/// <remarks>
/// Everything offered here is arithmetic on a row: where the data comes from, which columns are derived
/// from which. Nothing here learns anything from the data as a whole — the operations that do are not
/// methods on this type, and the one door that takes a step from another package refuses a step that says
/// it learns. They arrive with <see cref="FittingBuilder"/>, which is only reachable by splitting.
/// </remarks>
public sealed class PipelineBuilder
{
    private readonly List<IPipelineStep> _steps = [];
    private bool _split;

    internal PipelineBuilder()
    {
    }

    /// <summary>What has been declared so far.</summary>
    public PipelineDeclaration Declaration => new(_steps);

    /// <summary>Adds a declared step. Every verb, including one from another package, comes through here.</summary>
    /// <param name="step">The step to declare. It must not be one that learns from the data.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <exception cref="InvalidOperationException">
    /// The step learns from the data, or this builder has already been split and is finished.
    /// </exception>
    public PipelineBuilder Add(IPipelineStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        ThrowIfSplit();

        if (step is IFittedStep)
        {
            throw new InvalidOperationException(
                $"'{step.Verb}' learns from the data, so it belongs after the split, not before it.");
        }

        _steps.Add(step);

        return this;
    }

    /// <summary>Declares which columns take part, what they hold, and what becomes of the rest.</summary>
    /// <param name="schema">Names the columns, in the order they should reach a model.</param>
    /// <param name="remainder">What becomes of the columns the schema does not name.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <exception cref="ArgumentException">There are no columns, or one is declared twice.</exception>
    public PipelineBuilder Declare(Action<SchemaBuilder> schema, Remainder remainder = Remainder.Drop)
    {
        ArgumentNullException.ThrowIfNull(schema);

        var builder = new SchemaBuilder();
        schema(builder);

        return Add(new DeclareStep(builder.Columns, remainder));
    }

    /// <summary>Adds a column worked out from two others.</summary>
    /// <param name="name">What the new column is called.</param>
    /// <param name="left">The column on the left.</param>
    /// <param name="arithmetic">What to do with them.</param>
    /// <param name="right">The column on the right.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public PipelineBuilder AddFeature(string name, string left, Arithmetic arithmetic, string right) =>
        Add(new AddFeatureStep(name, left, arithmetic, right));

    /// <summary>Writes a moment in time as a place on a circle, so its ends meet.</summary>
    /// <param name="column">The column holding the moment.</param>
    /// <param name="period">Which cycle to place it on.</param>
    /// <param name="form">How to write the two values down.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public PipelineBuilder Cyclical(string column, Period period, Form form = Form.Signed) =>
        Add(new CyclicalStep(column, period, form));

    /// <summary>Splits the rows by where they sit in time, and opens the half of the chain that learns.</summary>
    /// <param name="column">The column that says when a row happened.</param>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models.</param>
    /// <param name="test">The share kept back until the end.</param>
    /// <returns>The builder that offers the steps which are fitted on the training rows.</returns>
    /// <exception cref="ArgumentException">The column has no name, or the shares do not make a whole.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A share is not a share.</exception>
    /// <exception cref="InvalidOperationException">This builder has already been split.</exception>
    public FittingBuilder SplitByTime(string column, double train, double validation, double test)
    {
        return Split(new SplitByTimeStep(column, train, validation, test));
    }

    /// <summary>Splits the rows at random, and opens the half of the chain that learns.</summary>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models.</param>
    /// <param name="test">The share kept back until the end.</param>
    /// <param name="seed">The number that makes the shuffle repeatable.</param>
    /// <returns>The builder that offers the steps which are fitted on the training rows.</returns>
    /// <remarks>The right split for rows that do not depend on one another.</remarks>
    public FittingBuilder SplitAtRandom(double train, double validation, double test, int seed = 20260923) =>
        Split(new SplitAtRandomStep(new SplitShares(train, validation, test), seed));

    /// <summary>Splits at random while keeping the mixture of one column the same in every part.</summary>
    /// <param name="column">The column whose mixture is kept.</param>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models.</param>
    /// <param name="test">The share kept back until the end.</param>
    /// <param name="seed">The number that makes the shuffle repeatable.</param>
    /// <returns>The builder that offers the steps which are fitted on the training rows.</returns>
    /// <remarks>The right split when an answer is rare enough that a plain shuffle could lose it.</remarks>
    public FittingBuilder SplitStratified(
        string column, double train, double validation, double test, int seed = 20260923) =>
        Split(new SplitStratifiedStep(column, new SplitShares(train, validation, test), seed));

    /// <summary>Finishes the pipeline, so it can be run.</summary>
    /// <returns>The declaration with the means to carry it out.</returns>
    public Pipeline Build() => new(Declaration);

    private FittingBuilder Split(ISplitStep step)
    {
        ThrowIfSplit();

        _steps.Add(step);
        _split = true;

        return new FittingBuilder(_steps);
    }

    private void ThrowIfSplit()
    {
        // Holding on to this builder after splitting used to let a feature be declared into the same list,
        // after the split — the leak arriving from the side — and a second split made one declaration that
        // claimed to divide the rows twice. Two pipelines are two chains, said out loud.
        if (_split)
        {
            throw new InvalidOperationException(
                "This pipeline has been split; carry on with the builder the split handed back, "
                + "or start another pipeline for a second arrangement.");
        }
    }
}

/// <summary>
/// A pipeline being written, after it has been told how to split.
/// </summary>
/// <remarks>
/// This is where everything that learns from the data lives, and each of those things is fitted on the
/// training rows alone. There is no way back to <see cref="PipelineBuilder"/>, and the builder the chain
/// started with is finished the moment it is split, so nothing can be added on the far side of the line.
/// </remarks>
public sealed class FittingBuilder
{
    private readonly List<IPipelineStep> _steps;

    internal FittingBuilder(List<IPipelineStep> steps) => _steps = steps;

    /// <summary>What has been declared so far.</summary>
    public PipelineDeclaration Declaration => new(_steps);

    /// <summary>Adds a declared step, which may be one that is fitted on the training rows.</summary>
    /// <param name="step">The step to declare.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder Add(IPipelineStep step)
    {
        ArgumentNullException.ThrowIfNull(step);

        _steps.Add(step);

        return this;
    }

    /// <summary>Fills the gaps in a column, the named way.</summary>
    /// <param name="column">The column with gaps in it.</param>
    /// <param name="strategy">What to put in them — <see cref="With"/> has the names.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    /// <exception cref="ArgumentException">The column has no name, or the strategy is not one of the names.</exception>
    public FittingBuilder FillMissing(string column, FillStrategy strategy) =>
        Add(new FillMissingStep(column, strategy));

    /// <summary>Brings a column onto a comparable scale, by numbers learned from the training rows.</summary>
    /// <param name="column">The column to scale.</param>
    /// <param name="scale">Which kind of scaling.</param>
    /// <param name="outOfRange">What happens to a value outside the range the fit learned.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder Normalise(
        string column, Scale scale = Scale.Standard, OutOfRange outOfRange = OutOfRange.Pass) =>
        Add(new NormaliseStep(column, scale, outOfRange));

    /// <summary>Brings several columns onto a comparable scale, by numbers learned from the training rows.</summary>
    /// <param name="columns">The columns to scale, each on its own.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder Normalise(params string[] columns)
    {
        ArgumentNullException.ThrowIfNull(columns);

        foreach (var column in columns)
        {
            Add(new NormaliseStep(column));
        }

        return this;
    }

    /// <summary>Writes a column of words down as numbers, using the categories the training rows held.</summary>
    /// <param name="column">The column of words.</param>
    /// <param name="how">One column per category, or one column of places.</param>
    /// <param name="unseen">What happens to a category the training rows never held.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder Encode(string column, As how = As.OneHot, Unseen unseen = Unseen.Reserve) =>
        Add(new EncodeStep(column, how, unseen));

    /// <summary>Brings each row onto a comparable scale, learning nothing.</summary>
    /// <param name="norm">How the row's size is measured.</param>
    /// <param name="columns">The columns that make up the row.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public FittingBuilder NormaliseRow(Norm norm, params string[] columns) =>
        Add(new NormaliseRowStep(columns, norm));

    /// <summary>Finishes the pipeline, so it can be run.</summary>
    /// <returns>The declaration with the means to carry it out.</returns>
    public Pipeline Build() => new(Declaration);
}
