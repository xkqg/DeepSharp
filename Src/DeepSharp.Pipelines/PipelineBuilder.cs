// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// A pipeline being written, before it has been told how to split.
/// </summary>
/// <remarks>
/// Everything offered here is arithmetic on a row: where the data comes from, which columns are derived
/// from which. Nothing here learns anything from the data as a whole, and that is not a convention — the
/// operations that do learn are not methods on this type. They arrive with <see cref="FittingBuilder"/>,
/// which is only reachable by saying how the data is split.
/// </remarks>
public sealed class PipelineBuilder
{
    private readonly List<IPipelineStep> _steps = [];

    internal PipelineBuilder()
    {
    }

    /// <summary>What has been declared so far.</summary>
    public PipelineDeclaration Declaration => new(_steps);

    /// <summary>Adds a declared step. Every verb, including one from another package, comes through here.</summary>
    /// <param name="step">The step to declare.</param>
    /// <returns>This builder, so the next verb can be written after it.</returns>
    public PipelineBuilder Add(IPipelineStep step)
    {
        ArgumentNullException.ThrowIfNull(step);

        _steps.Add(step);

        return this;
    }

    /// <summary>Splits the rows by where they sit in time, and opens the half of the chain that learns.</summary>
    /// <param name="column">The column that says when a row happened.</param>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models.</param>
    /// <param name="test">The share kept back until the end.</param>
    /// <returns>The builder that offers the steps which are fitted on the training rows.</returns>
    /// <exception cref="ArgumentException">The column has no name, or the shares do not make a whole.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A share is not a share.</exception>
    public FittingBuilder SplitByTime(string column, double train, double validation, double test)
    {
        Add(new SplitByTimeStep(column, train, validation, test));

        return new FittingBuilder(_steps);
    }
}

/// <summary>
/// A pipeline being written, after it has been told how to split.
/// </summary>
/// <remarks>
/// This is where everything that learns from the data lives, and each of those things is fitted on the
/// training rows alone. There is no way back to <see cref="PipelineBuilder"/>: declaring a feature after
/// the split is the same leak from the other side, since the feature would be computed over rows the model
/// is meant never to have seen.
/// </remarks>
public sealed class FittingBuilder
{
    private readonly List<IPipelineStep> _steps;

    internal FittingBuilder(List<IPipelineStep> steps) => _steps = steps;

    /// <summary>What has been declared so far.</summary>
    public PipelineDeclaration Declaration => new(_steps);

    /// <summary>Adds a declared step that is fitted on the training rows.</summary>
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
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public FittingBuilder FillMissing(string column, FillStrategy strategy) =>
        Add(new FillMissingStep(column, strategy));
}
