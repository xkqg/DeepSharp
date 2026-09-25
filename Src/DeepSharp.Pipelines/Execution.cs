// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>
/// A step that knows where rows come from.
/// </summary>
/// <remarks>
/// Declaring a source and opening one are two different moments, which is the whole reason a pipeline can
/// be written on a machine that holds no data. This is the second moment.
/// </remarks>
public interface IOpensRows : IActsInAWalk
{
    /// <summary>Opens the source and hands back its rows.</summary>
    /// <param name="folder">Where a relative path the step holds is read from: the pipeline's folder.</param>
    /// <returns>The rows, as text, with their column names.</returns>
    IRowSource Open(SourceFolder folder);

    /// <inheritdoc />
    void IActsInAWalk.ActOn(Walk walk) => walk.Open(Open);
}

/// <summary>
/// A step that turns rows of text into named, typed columns.
/// </summary>
public interface IBindsColumns : IActsInAWalk
{
    /// <summary>Reads a source into the columns this step declares.</summary>
    /// <param name="source">The rows to read.</param>
    /// <returns>The table the pipeline carries from here on.</returns>
    Table Bind(IRowSource source);

    /// <inheritdoc />
    void IActsInAWalk.ActOn(Walk walk) => walk.Bind(Bind);
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
    /// which is the whole promise: <c>new Pipeline(PipelineDeclaration.FromJson(text, catalog)).Run()</c>.
    /// </remarks>
    public Pipeline(PipelineDeclaration declaration)
        : this(declaration, rows: null)
    {
    }

    /// <summary>A pipeline that carries out this declaration over rows handed in.</summary>
    /// <param name="declaration">The steps, in the order they were written.</param>
    /// <param name="rows">The rows, when the declaration says they are handed in.</param>
    public Pipeline(PipelineDeclaration declaration, IRowSource? rows)
        : this(declaration, rows, SourceFolder.WorkingDirectory)
    {
    }

    /// <summary>A pipeline that carries out this declaration, reading a relative path from a given folder.</summary>
    /// <param name="declaration">The steps, in the order they were written.</param>
    /// <param name="rows">The rows, when the declaration says they are handed in.</param>
    /// <param name="folder">The folder the pipeline sits in: that of the file it was read from, or of its notebook.</param>
    public Pipeline(PipelineDeclaration declaration, IRowSource? rows, SourceFolder folder)
    {
        ArgumentNullException.ThrowIfNull(declaration);
        ArgumentNullException.ThrowIfNull(folder);

        Declaration = declaration;
        Rows = rows;
        Folder = folder;
    }

    /// <summary>The rows handed in with this pipeline, when there are any.</summary>
    public IRowSource? Rows { get; }

    /// <summary>Where a relative path in the declaration is read from.</summary>
    public SourceFolder Folder { get; }

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
    public Table Prepare(IRowSource? rows) => new Walk(Declaration, new FitOnTheTrainingRows(), Folder).Bound(rows);

    /// <summary>The data as it stands after the first so many steps, with where each row stands.</summary>
    /// <param name="steps">How many steps from the start: at least one, at most all of them.</param>
    /// <returns>The rows there, and where each stands: in the part the split puts it in, dropped before it, or undivided.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There are not that many steps.</exception>
    /// <remarks>
    /// What the grid under a block of a notebook shows. Above the split, the rows are followed on down to it, so a
    /// range or a profile drawn here is drawn over the rows it trains on. Before the columns are declared, the rows
    /// are what the source says, every column as words.
    /// </remarks>
    public PipelineView ViewAt(int steps) => ViewAt(steps, Rows);

    /// <summary>The data as it stands after the first so many steps, over the given rows.</summary>
    /// <param name="steps">How many steps from the start: at least one, at most all of them.</param>
    /// <param name="rows">The rows to read, or nothing to use the source the declaration names.</param>
    /// <returns>The rows there, and where each stands.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There are not that many steps.</exception>
    public PipelineView ViewAt(int steps, IRowSource? rows)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(steps, Declaration.Steps.Count);

        if (steps > Declaration.ColumnsAt && Declaration.ColumnsAt >= 0)
        {
            return new Walk(Declaration, new FitOnTheTrainingRows(), Folder).Viewed(rows, steps);
        }

        // Before the schema: the rows as the source holds them, every column as words.
        var source = rows ?? ((IOpensRows)Declaration.Steps[0]).Open(Folder);
        var asText = new DeclareStep(
            [.. source.ColumnNames.Select(name => new ColumnDeclaration(name, ColumnKind.Text, Optional: false))]);

        return new Walk(Declaration, new FitOnTheTrainingRows(), Folder).Placed(source, SchemaBinding.Bind(asText, source), steps);
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
    /// <remarks>
    /// Every step acts where it was written: rows are dropped where the drop stands, divided where the split
    /// stands, and each step that learns is fitted on the training rows as they stand at its place. The run
    /// up to any one step — the grid under a notebook block — is this same walk over the steps up to it.
    /// </remarks>
    public PreparedData Run(IRowSource? rows)
    {
        var fitting = new FitOnTheTrainingRows();
        var walked = new Walk(Declaration, fitting, Folder).Through(rows);
        var prepared = new PreparedData(Declaration, walked.Table, walked.Parts, fitting.Fitted, walked.Evidence)
        {
            AsRead = fitting.AsRead,
        };

        // Scale what a model predicts and the predictions come back scaled; the way back has to be real, and
        // the only way to know that is to try it on values whose answer is already known.
        ThrowIfTheWayBackIsNotReal(prepared);

        return prepared;
    }

    private static void ThrowIfTheWayBackIsNotReal(PreparedData prepared)
    {
        foreach (var answer in prepared.Declaration.Output?.Answers ?? [])
        {
            var chain = UndoChain.For(prepared.Declaration, prepared.Fitted, answer);

            // A way back with nothing on it has nothing to check, and one that comes back to a column that was not
            // numbers as read — words, a moment, a column a step made — has no values to check it against.
            if (chain.Links.Count == 0 || !prepared.AsRead.Holds(chain.End))
            {
                continue;
            }

            var now = prepared.Table.NumbersOf(answer);

            // Each row against the row it was read as, not against whatever row now sits at its place: rows
            // dropped at the start used to shift every comparison onto a different row. An answer read ahead lands
            // on the row that many rows later, in the order the rows stand in.
            for (var row = 0; row + chain.Ahead < now.Length; row++)
            {
                var readAt = prepared.Table.Identities[row].ReadAt;
                var landsAt = prepared.Table.Identities[row + chain.Ahead].ReadAt;

                if (prepared.AsRead.At(chain.End, landsAt) is not { } was || now[row] is not { } is_)
                {
                    continue;
                }

                var back = chain.Back(is_, new RowAsRead(prepared.AsRead, readAt));

                if (Math.Abs(back - was) <= 1e-6 * Math.Max(1, Math.Abs(was)))
                {
                    continue;
                }

                throw new InvalidOperationException(chain.Ahead == 0
                    ? string.Create(
                        CultureInfo.InvariantCulture,
                        $"The way back for '{answer}' does not lead back: row {readAt + 1} was {was}, became {is_}, and comes back as {back}. A prediction from this pipeline would be in units nobody can name.")
                    : string.Create(
                        CultureInfo.InvariantCulture,
                        $"The way back for '{answer}' does not lead back: the answer on row {readAt + 1} is {is_} and comes back as {back}, where row {landsAt + 1}, {chain.Ahead} rows later, was {was}. A prediction from this pipeline would be in units nobody can name."));
            }
        }
    }
}
