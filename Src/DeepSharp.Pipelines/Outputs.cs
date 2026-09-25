// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// A step that names what a model is asked to predict: the output of the pipeline.
/// </summary>
/// <remarks>
/// A pipeline has one output, and the kind of output decides everything about the answer: which columns hold it,
/// what the handover hands over, what a row served later waits for. The kinds are the verbs that implement this
/// — one column, several, a column a number of rows ahead — and a new kind is a new verb, registered like any
/// other.
/// <para>
/// Naming the answer is not doing something to the data, so an output that only names its answers acts on
/// nothing, and it is the one kind of step the run is allowed to leave alone.
/// </para>
/// </remarks>
public interface INamesTheAnswer : IPipelineStep
{
    /// <summary>The columns that hold the answer, in their order, as they stand after this step.</summary>
    IReadOnlyList<string> Answers { get; }

    /// <summary>Why a row's answers cannot be handed over as this output's answers, when they cannot.</summary>
    /// <param name="answers">One row's answers, in the order <see cref="Answers"/> names them, each a finite number.</param>
    /// <returns>The reason, or nothing when the row may be handed over.</returns>
    /// <remarks>
    /// Nothing, for an output that takes any number as its answer. A kind of output that promises more — a
    /// distribution that sums to one, labels that are nought or one — says so here, and the handover asks it of
    /// every row it hands over: a model trained towards an answer its output could not have meant learns the
    /// wrong thing, and says nothing about it.
    /// </remarks>
    string? Refusal(IReadOnlyList<double> answers) => null;
}

/// <summary>
/// Names the column a model is being asked to predict.
/// </summary>
/// <remarks>
/// The answer is not a feature, so it is taken out of what a model is shown and handed over separately. A
/// pipeline that left it among the inputs would produce a model that scores perfectly and knows nothing —
/// and the same mistake wears a quieter costume when a column merely restates the answer, which is what
/// declaring the columns is for.
/// </remarks>
public sealed record TargetStep : IPipelineStep<TargetStep>, INamesTheAnswer, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column a model is asked to predict, handed over apart from the numbers it is shown.", "answer", ColumnKinds.Any);

    /// <summary>Declares which column holds the answer.</summary>
    /// <param name="column">The column being predicted.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public TargetStep(string column) => Column = ColumnKey.Require(column);

    /// <summary>The column being predicted.</summary>
    public string Column { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> Answers => [Column];

    /// <inheritdoc />
    public static string Name => "target";

    /// <inheritdoc />
    public static string Purpose => "Names the column a model is asked to predict, which is handed over apart from the numbers it is shown.";

    /// <inheritdoc />
    public static StepParameters<TargetStep> Parameters { get; } =
        new StepParameters<TargetStep>().With(ColumnKey, step => step.Column);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static TargetStep ReadFrom(JsonElement element) => new(ColumnKey.Read(element));
}
