// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Read the rows from a comma-separated file.
/// </summary>
/// <remarks>
/// Declaring where the data comes from is not the same act as going to get it: nothing is opened until the
/// pipeline runs, so a declaration can be written, saved and checked on a machine that has no data on it.
/// </remarks>
public sealed record ReadCsvStep : IPipelineStep<ReadCsvStep>, IOpensRows
{
    /// <summary>Declares that the rows come from the file at this path.</summary>
    /// <param name="path">Where the file will be, when the pipeline runs.</param>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    public ReadCsvStep(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("A CSV step needs a path to read from.", nameof(path));
        }

        Path = path;
    }

    /// <summary>Where the file will be, when the pipeline runs.</summary>
    public string Path { get; }

    /// <inheritdoc />
    public static string Name => "read.csv";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("path", Path);
        writer.WriteEndObject();
    }

    /// <inheritdoc />
    public IRowSource Open() => new CsvRowSource(Path);

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing or is not text.</exception>
    public static ReadCsvStep ReadFrom(JsonElement element) => new(element.RequiredString("path"));
}

/// <summary>
/// The rows are handed in rather than opened.
/// </summary>
/// <remarks>
/// A source that lives outside the file: a table already in memory, a reader over a query, anything a
/// package turns into rows. The declaration says so and says what it was, and the rows themselves are
/// handed to the pipeline when it runs — which is also exactly how serving works, so the same declaration
/// covers both without a second shape.
/// </remarks>
public sealed record ReadRowsStep : IPipelineStep<ReadRowsStep>, IOpensRows
{
    /// <summary>Declares that the rows are handed in.</summary>
    /// <param name="description">What the rows are, for whoever reads the file later.</param>
    /// <exception cref="ArgumentException">The description is empty.</exception>
    public ReadRowsStep(string description)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(description);

        Description = description;
    }

    /// <summary>What the rows are, for whoever reads the file later.</summary>
    public string Description { get; }

    /// <inheritdoc />
    public static string Name => "read.rows";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public IRowSource Open() =>
        throw new InvalidOperationException(
            $"This pipeline reads rows that are handed in ({Description}). "
            + "Hand them to Run or Prepare, or to Replay when the pipeline is already fitted.");

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("description", Description);
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static ReadRowsStep ReadFrom(JsonElement element) => new(element.RequiredString("description"));
}

/// <summary>
/// Split the rows into training, validation and test by where they sit in time.
/// </summary>
/// <remarks>
/// This is the line in the chain. Above it nothing may learn from the data; below it the operations that do
/// become available, and each of them is fitted on the training rows alone.
/// </remarks>
public sealed record SplitByTimeStep : ISplitStep, IAssignsSplits, IPipelineStep<SplitByTimeStep>
{
    /// <summary>Declares a split in time, by three shares that together make a whole.</summary>
    /// <param name="column">The column that says when a row happened.</param>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models.</param>
    /// <param name="test">The share kept back until the end.</param>
    /// <exception cref="ArgumentException">The column has no name, or the shares do not make a whole.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A share is not a share: nothing, or more than everything.</exception>
    public SplitByTimeStep(string column, double train, double validation, double test)
    {
        if (string.IsNullOrWhiteSpace(column))
        {
            throw new ArgumentException("A split in time needs the column that says when.", nameof(column));
        }

        // Each share first, then the sum. An empty split is not a split — a model measured on nothing
        // scores perfectly on nothing — and three shares can add to one while one of them is nonsense.
        ThrowIfNotAShare(train, nameof(train));
        ThrowIfNotAShare(validation, nameof(validation));
        ThrowIfNotAShare(test, nameof(test));

        var total = train + validation + test;
        if (Math.Abs(total - 1) > 1e-9)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"The three shares add up to {total:0.####} and a split has to use every row."),
                nameof(train));
        }

        Column = column;
        Train = train;
        Validation = validation;
        Test = test;
    }

    /// <summary>The column that says when a row happened.</summary>
    public string Column { get; }

    /// <summary>The share the model learns from.</summary>
    public double Train { get; }

    /// <summary>The share used while choosing between models.</summary>
    public double Validation { get; }

    /// <summary>The share kept back until the end.</summary>
    public double Test { get; }

    /// <inheritdoc />
    public Split[] Assign(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var column = table[Column];
        var order = new (int Row, double When)[table.RowCount];

        for (var row = 0; row < table.RowCount; row++)
        {
            if (column.IsMissing(row))
            {
                // A row with no time cannot be placed in a split by time, and guessing where it belongs is
                // how a row from next year ends up in the training data.
                throw new InvalidOperationException(
                    $"Row {row + 1} has no '{Column}', so a split in time has nowhere to put it.");
            }

            order[row] = (row, When(column, row));
        }

        Array.Sort(order, (left, right) => left.When.CompareTo(right.When));

        return new SplitShares(Train, Validation, Test)
            .Over(table.RowCount)
            .Placed([.. order.Select(each => each.Row)]);
    }

    private static double When(IColumn column, int row) => column switch
    {
        Column<DateTime> timestamps => timestamps[row]!.Value.Ticks,
        Column<long> numbers => numbers[row]!.Value,
        Column<double> numbers => numbers[row]!.Value,
        _ => throw new InvalidOperationException(
            $"'{column.Name}' holds {column.Kind.ToString().ToLowerInvariant()}, which has no order in time."),
    };

    /// <inheritdoc />
    public static string Name => "split.byTime";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("column", Column);
        writer.WriteNumber("train", Train);
        writer.WriteNumber("validation", Validation);
        writer.WriteNumber("test", Test);
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing or is of the wrong kind.</exception>
    public static SplitByTimeStep ReadFrom(JsonElement element) =>
        new(element.RequiredString("column"),
            element.RequiredNumber("train"),
            element.RequiredNumber("validation"),
            element.RequiredNumber("test"));

    private static void ThrowIfNotAShare(double share, string name)
    {
        // Written as the negation of what a share IS, because every comparison against a not-a-number is
        // false: a range test lets NaN through both this check and the sum, and the declaration that
        // results cannot even be written down.
        if (!(share > 0 && share <= 1))
        {
            throw new ArgumentOutOfRangeException(
                name, share, "A share of the data is more than none of it and at most all of it.");
        }
    }
}

/// <summary>
/// Fill the gaps in a column, the named way.
/// </summary>
/// <remarks>
/// Missing is not the same as not-a-number: a value is missing when it was never there, which is data,
/// while a not-a-number is arithmetic that produced no number, which is a fault further upstream. They get
/// different verbs because they deserve different answers.
/// </remarks>
public sealed record FillMissingStep : IFittedStep, ILearnsFromData, IPipelineStep<FillMissingStep>
{
    /// <summary>Declares that the gaps in a column are filled the named way.</summary>
    /// <param name="column">The column with gaps in it.</param>
    /// <param name="strategy">What to put in them, learned from the training rows.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public FillMissingStep(string column, FillStrategy strategy)
    {
        if (string.IsNullOrWhiteSpace(column))
        {
            throw new ArgumentException("Filling gaps needs the column they are in.", nameof(column));
        }

        // A strategy is a name, so a default one carries no name at all and a hand-written file can carry
        // any word. Both are caught here, at the one point a strategy enters a declaration.
        if (string.IsNullOrWhiteSpace(strategy.Name))
        {
            throw new ArgumentException("Filling gaps needs a strategy; With has the names.", nameof(strategy));
        }

        if (!With.Knows(strategy.Name))
        {
            throw new ArgumentException(
                $"'{strategy.Name}' is not a way of filling a gap. With has the names.", nameof(strategy));
        }

        if (With.TakesAValue(strategy.Name) != strategy.Value.HasValue)
        {
            throw new ArgumentException(
                $"The strategy '{strategy.Name}' is written {(strategy.Value.HasValue ? "without" : "with")} a number.",
                nameof(strategy));
        }

        Column = column;
        Strategy = strategy;
    }

    /// <summary>The column with gaps in it.</summary>
    public string Column { get; }

    /// <summary>What goes in them, learned from the training rows.</summary>
    public FillStrategy Strategy { get; }

    /// <summary>The column written beside a filled one, saying where the gaps were.</summary>
    /// <remarks>
    /// Always, and not behind a flag. Filling destroys the difference between "absent" and "the value
    /// happened to be that" permanently, so the difference is written down before it goes.
    /// </remarks>
    public string MarkerColumn => $"{Column}_was_missing";

    /// <inheritdoc />
    public FittedStepValues Fit(Table table, IReadOnlyList<Split> splits)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(splits);

        var column = table[Column];
        var learned = new FittedStepValues();

        // Every number here comes from the training rows and from nowhere else. Fit on all of them and the
        // validation rows have quietly taught the model about themselves, and nothing goes red.
        var training = Enumerable.Range(0, table.RowCount)
            .Where(row => splits[row] == Split.Train && !column.IsMissing(row))
            .Select(row => NumberAt(column, row))
            .ToArray();

        learned.Learned("gaps", Enumerable.Range(0, table.RowCount).Count(column.IsMissing));

        switch (Strategy.Name)
        {
            case "mean":
                learned.Learned("value", Refuse.IfEmpty(training, Column).Average());
                break;

            case "median":
                var sorted = Refuse.IfEmpty(training, Column).Order().ToArray();
                learned.Learned("value", sorted.Length % 2 == 1
                    ? sorted[sorted.Length / 2]
                    : (sorted[(sorted.Length / 2) - 1] + sorted[sorted.Length / 2]) / 2);
                break;

            case "zero":
                learned.Learned("value", 0);
                break;

            case "constant":
                learned.Learned("value", Strategy.Value!.Value);
                break;

            case "refuse" when learned.Number("gaps") > 0:
                throw new InvalidOperationException(
                    $"'{Column}' has {learned.Number("gaps"):0} gaps, and this pipeline says there should be none.");

            default:
                // Carrying the previous value forward learns nothing, and the value it uses depends on the
                // row above rather than on the training set.
                break;
        }

        return learned;
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var column = table[Column];
        var marker = new Column<double>(
            MarkerColumn, ColumnKind.Number,
            Enumerable.Range(0, table.RowCount).Select(row => (double?)(column.IsMissing(row) ? 1 : 0)));

        switch (column)
        {
            case Column<double> numbers:
                Fill(numbers, fitted, value => value);
                break;

            case Column<long> whole:
                Fill(whole, fitted, value => (long)Math.Round(value, MidpointRounding.AwayFromZero));
                break;

            default:
                throw new InvalidOperationException(
                    $"'{Column}' holds {column.Kind.ToString().ToLowerInvariant()}, and a gap in it is not filled with a number.");
        }

        table.Put(marker);
    }

    private void Fill<T>(Column<T> column, FittedStepValues fitted, Func<double, T> asValue)
        where T : struct
    {
        T? previous = null;

        for (var row = 0; row < column.Count; row++)
        {
            if (!column.IsMissing(row))
            {
                previous = column[row];
                continue;
            }

            column[row] = Strategy.Name == "previous"
                ? previous ?? throw new InvalidOperationException(
                    $"Row {row + 1} of '{Column}' is a gap with nothing before it to carry forward.")
                : asValue(fitted.Number("value"));
        }
    }

    private static double NumberAt(IColumn column, int row) => column switch
    {
        Column<double> numbers => numbers[row]!.Value,
        Column<long> whole => whole[row]!.Value,
        _ => throw new InvalidOperationException(
            $"'{column.Name}' holds {column.Kind.ToString().ToLowerInvariant()}, and a gap in it is not filled with a number."),
    };

    /// <inheritdoc />
    public static string Name => "fill.missing";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("column", Column);

        // A strategy with no number is written as the word alone, which is what a person reads best. One
        // that carries a number becomes an object, so the number has somewhere to live.
        if (Strategy.Value is { } value)
        {
            writer.WriteStartObject("with");
            writer.WriteString("kind", Strategy.Name);
            writer.WriteNumber("value", value);
            writer.WriteEndObject();
        }
        else
        {
            writer.WriteString("with", Strategy.Name);
        }

        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing or is not text.</exception>
    public static FillMissingStep ReadFrom(JsonElement element)
    {
        var column = element.RequiredString("column");

        if (!element.TryGetProperty("with", out var with))
        {
            throw new FormatException("The step is missing a text value for 'with'.");
        }

        return with.ValueKind switch
        {
            JsonValueKind.String => new FillMissingStep(column, new FillStrategy(with.GetString()!)),
            JsonValueKind.Object => new FillMissingStep(
                column, new FillStrategy(with.RequiredString("kind"), with.RequiredNumber("value"))),
            _ => throw new FormatException("The step is missing a text value for 'with'."),
        };
    }
}
