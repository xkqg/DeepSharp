// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// A step that knows where rows come from.
/// </summary>
/// <remarks>
/// Declaring a source and opening one are two different moments, which is the whole reason a pipeline can
/// be written on a machine that holds no data. This is the second moment.
/// </remarks>
public interface IOpensRows : IPipelineStep
{
    /// <summary>Opens the source and hands back its rows.</summary>
    /// <returns>The rows, as text, with their column names.</returns>
    IRowSource Open();
}

/// <summary>
/// A step that turns rows of text into named, typed columns.
/// </summary>
public interface IBindsColumns : IPipelineStep
{
    /// <summary>Reads a source into the columns this step declares.</summary>
    /// <param name="source">The rows to read.</param>
    /// <returns>The table the pipeline carries from here on.</returns>
    Table Bind(IRowSource source);
}

/// <summary>
/// A pipeline that has been written down and can be run.
/// </summary>
/// <remarks>
/// The declaration is what a person wrote; this is that declaration with the means to carry it out. It
/// holds no data of its own, so running it twice on the same source gives the same answer and running it on
/// another source is the ordinary thing to do rather than a trick.
/// </remarks>
public sealed class Pipeline
{
    /// <summary>A pipeline that carries out this declaration.</summary>
    /// <param name="declaration">The steps, in the order they were written.</param>
    /// <remarks>
    /// Public because a declaration read back from a file is exactly as runnable as one written in C#,
    /// which is the whole promise: <c>new Pipeline(PipelineDeclaration.FromJson(text)).Run()</c>.
    /// </remarks>
    public Pipeline(PipelineDeclaration declaration)
    {
        ArgumentNullException.ThrowIfNull(declaration);

        Declaration = declaration;
    }

    /// <summary>The steps, exactly as they were declared.</summary>
    public PipelineDeclaration Declaration { get; }

    /// <summary>Opens the source and reads it into the declared columns.</summary>
    /// <returns>The table, with every declared column and nothing else unless the schema said otherwise.</returns>
    /// <exception cref="InvalidOperationException">
    /// The declaration names no source, or names no columns, or a declared column is not in the source.
    /// </exception>
    /// <exception cref="FormatException">A cell cannot be read as the kind its column was declared to be.</exception>
    public Table Prepare()
    {
        var source = Declaration.Steps.OfType<IOpensRows>().FirstOrDefault()
            ?? throw new InvalidOperationException(
                "This pipeline never says where its rows come from, so there is nothing to prepare.");

        var schema = Declaration.Steps.OfType<IBindsColumns>().FirstOrDefault()
            ?? throw new InvalidOperationException(
                "This pipeline never says which columns take part. Declare them, and the rest is dropped.");

        return schema.Bind(source.Open());
    }

    /// <summary>Runs the whole declaration: reads, divides the rows, fits on training, replays everywhere.</summary>
    /// <returns>The data, where every row landed, and what each step learned.</returns>
    /// <exception cref="InvalidOperationException">
    /// The declaration names no source, no columns, or no split while something in it learns.
    /// </exception>
    /// <remarks>
    /// The order is the point. Rows are divided before anything is fitted, every fit sees the training
    /// rows alone, and what it learned is then applied to all of them — so validation, test and a row that
    /// arrives a year from now meet the same numbers.
    /// </remarks>
    public PreparedData Run()
    {
        var table = Prepare();
        var splits = Assign(table);
        var fitted = new Dictionary<int, FittedStepValues>();

        for (var at = 0; at < Declaration.Steps.Count; at++)
        {
            if (Declaration.Steps[at] is not ILearnsFromData step)
            {
                continue;
            }

            var learned = step.Fit(table, splits);
            step.ApplyTo(table, learned);
            fitted[at] = learned;
        }

        return new PreparedData(Declaration, table, splits, fitted);
    }

    private Split[] Assign(Table table)
    {
        var split = Declaration.Steps.OfType<IAssignsSplits>().FirstOrDefault();

        if (split is not null)
        {
            return split.Assign(table);
        }

        if (Declaration.Steps.Any(step => step is IFittedStep))
        {
            throw new InvalidOperationException(
                "Something in this pipeline learns from the data, and the rows have not been divided.");
        }

        // A pipeline that learns nothing needs no split, and every row is simply itself.
        return [.. Enumerable.Repeat(Split.Train, table.RowCount)];
    }
}
