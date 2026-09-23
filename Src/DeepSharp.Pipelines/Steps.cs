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
public sealed record ReadCsvStep : IPipelineStep
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
    public string Verb => "read.csv";

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("path", Path);
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing or is not text.</exception>
    public static ReadCsvStep ReadFrom(JsonElement element) => new(element.RequiredString("path"));
}

/// <summary>
/// Split the rows into training, validation and test by where they sit in time.
/// </summary>
/// <remarks>
/// This is the line in the chain. Above it nothing may learn from the data; below it the operations that do
/// become available, and each of them is fitted on the training rows alone.
/// </remarks>
public sealed record SplitByTimeStep : IPipelineStep
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
    public string Verb => "split.byTime";

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
        if (share is <= 0 or > 1)
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
public sealed record FillMissingStep : IPipelineStep
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

        Column = column;
        Strategy = strategy;
    }

    /// <summary>The column with gaps in it.</summary>
    public string Column { get; }

    /// <summary>What goes in them, learned from the training rows.</summary>
    public FillStrategy Strategy { get; }

    /// <inheritdoc />
    public string Verb => "fill.missing";

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("column", Column);
        writer.WriteString("with", Strategy.Name);
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing or is not text.</exception>
    public static FillMissingStep ReadFrom(JsonElement element) =>
        new(element.RequiredString("column"), new FillStrategy(element.RequiredString("with")));
}
