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
        : this(declaration, rows: null)
    {
    }

    /// <summary>A pipeline that carries out this declaration over rows handed in.</summary>
    /// <param name="declaration">The steps, in the order they were written.</param>
    /// <param name="rows">The rows, when the declaration says they are handed in.</param>
    public Pipeline(PipelineDeclaration declaration, IRowSource? rows)
    {
        ArgumentNullException.ThrowIfNull(declaration);

        Declaration = declaration;
        Rows = rows;
    }

    /// <summary>The rows handed in with this pipeline, when there are any.</summary>
    public IRowSource? Rows { get; }

    /// <summary>The steps, exactly as they were declared.</summary>
    public PipelineDeclaration Declaration { get; }

    /// <summary>Opens the source and reads it into the declared columns.</summary>
    /// <returns>The table, with every declared column and nothing else unless the schema said otherwise.</returns>
    /// <exception cref="InvalidOperationException">
    /// The declaration names no source, or names no columns, or a declared column is not in the source.
    /// </exception>
    /// <exception cref="FormatException">A cell cannot be read as the kind its column was declared to be.</exception>
    public Table Prepare() => Prepare(Rows);

    /// <summary>Opens the given rows and reads them into the declared columns.</summary>
    /// <param name="rows">The rows to read, or nothing to use the source the declaration names.</param>
    /// <returns>The table, with every declared column and nothing else unless the schema said otherwise.</returns>
    /// <exception cref="InvalidOperationException">
    /// The declaration names no source, or names no columns, or a declared column is not in the source.
    /// </exception>
    public Table Prepare(IRowSource? rows)
    {
        var schema = Declaration.Steps.OfType<IBindsColumns>().FirstOrDefault()
            ?? throw new InvalidOperationException(
                "This pipeline never says which columns take part. Declare them, and the rest is dropped.");

        if (rows is not null)
        {
            return schema.Bind(rows);
        }

        var source = Declaration.Steps.OfType<IOpensRows>().FirstOrDefault()
            ?? throw new InvalidOperationException(
                "This pipeline never says where its rows come from, so there is nothing to prepare.");

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
    public PreparedData Run() => Run(Rows);

    /// <summary>Runs the whole declaration over the given rows.</summary>
    /// <param name="rows">The rows to read, or nothing to use the source the declaration names.</param>
    /// <returns>The data, where every row landed, and what each step learned.</returns>
    /// <exception cref="InvalidOperationException">The declaration is not one that can be run.</exception>
    public PreparedData Run(IRowSource? rows)
    {
        var table = Prepare(rows);
        var steps = Declaration.Steps;
        var line = steps.Count;

        for (var at = 0; at < steps.Count; at++)
        {
            if (steps[at] is ISplitStep)
            {
                line = at;
                break;
            }
        }

        // Everything before the split is arithmetic on a row, and it has to happen before the rows are
        // divided: a feature is what the split then divides, not something added to one part of it.
        for (var at = 0; at < line; at++)
        {
            if (steps[at] is IAddsColumns adds)
            {
                adds.AddTo(table);
            }
        }

        var splits = Assign(table);
        var fitted = new Dictionary<int, FittedStepValues>();

        for (var at = line; at < steps.Count; at++)
        {
            switch (steps[at])
            {
                case ILearnsFromData learns:
                    var learned = learns.Fit(table, splits);
                    learns.ApplyTo(table, learned);
                    fitted[at] = learned;
                    break;

                case IAddsColumns adds:
                    adds.AddTo(table);
                    break;
            }
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

        // Nothing here needs to refuse a pipeline that learns without splitting: a declaration carrying a
        // step that learns and no split is refused when it is built, whichever door it came through.
        // A pipeline that learns nothing needs no split, and every row is simply itself.
        return [.. Enumerable.Repeat(Split.Train, table.RowCount)];
    }
}
