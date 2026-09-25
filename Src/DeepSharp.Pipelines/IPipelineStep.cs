// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// One declared step of a pipeline.
/// </summary>
/// <remarks>
/// A step says what is to be done, never does it. That is what allows a pipeline to be written down, handed
/// to somebody, saved beside a model and replayed a year later on data that did not exist yet — none of
/// which is possible for a chain that performed the work while it was being typed.
/// <para>
/// Implement this as a <see langword="record"/>. Two declarations are compared step by step, so a step that
/// compares by reference makes a pipeline read back from a file unequal to the one it was written from,
/// with nothing to point at the cause.
/// </para>
/// </remarks>
public interface IPipelineStep
{
    /// <summary>The name this step is written under, in a file and in a catalog. Lowercase, dotted.</summary>
    string Verb { get; }

    /// <summary>Writes this step as one JSON object, its verb included.</summary>
    /// <param name="writer">The writer positioned where the object belongs.</param>
    void WriteTo(Utf8JsonWriter writer);

    /// <summary>The columns this step reads, each with the kinds of column it can work on.</summary>
    /// <remarks>
    /// Taken from the parameters that name columns, so a step says it once, where it says what it takes; a
    /// step that describes no parameters reads nothing anybody can check.
    /// </remarks>
    IReadOnlyList<ColumnRead> ColumnsRead => [];
}

/// <summary>
/// A step that knows its own name, its parameters, and how to read itself back.
/// </summary>
/// <typeparam name="TSelf">The step itself.</typeparam>
/// <remarks>
/// The verb used to be written twice — once in the step, once in the line that registered it — and nothing
/// required the two to agree or to both exist. Here the name, what it does, the parameters and the reader
/// belong to the type, so registering a step is one token and a step that forgot to register is something a
/// test can find. A step is written from its parameters, so a key is typed in exactly one place.
/// <para>
/// What every step has belongs to its type, and a step without one of these does not compile. What only some
/// steps do — open rows, learn from the training rows, divide them — is a capability the step implements.
/// </para>
/// </remarks>
public interface IPipelineStep<TSelf> : IPipelineStep
    where TSelf : IPipelineStep<TSelf>
{
    /// <summary>The single place this step's verb is written.</summary>
    static abstract string Name { get; }

    /// <summary>What the step does, in one sentence a person reads: the line under its name in the reference, a form and a hover.</summary>
    static abstract string Purpose { get; }

    /// <summary>The version of the pipeline file from which this verb means what it says now.</summary>
    /// <remarks>
    /// The first version, unless the verb is newer than that or its meaning changed. A file written against
    /// an older version that names the verb is refused by name rather than loaded: it would run something the
    /// person who wrote the file never saw.
    /// </remarks>
    static virtual int Since => 1;

    /// <summary>The step's parameters, each bound to where its value lives, in the order they are written.</summary>
    static abstract StepParameters<TSelf> Parameters { get; }

    /// <summary>Reads this step back out of the JSON object it was written as.</summary>
    /// <param name="element">The object, including its <c>step</c> key.</param>
    /// <returns>The step the file describes.</returns>
    static abstract TSelf ReadFrom(JsonElement element);

    /// <inheritdoc />
    void IPipelineStep.WriteTo(Utf8JsonWriter writer) => TSelf.Parameters.Write(writer, (TSelf)this);

    /// <inheritdoc />
    IReadOnlyList<ColumnRead> IPipelineStep.ColumnsRead => TSelf.Parameters.ReadBy((TSelf)this);
}

/// <summary>
/// A step the run acts on: it opens rows, names the columns, adds or drops something, divides the rows, or
/// learns from them.
/// </summary>
/// <remarks>
/// Every step does exactly one of those, and says which by the capability it implements; an output that only
/// names the answer acts on nothing, because it names the answer rather than changing the data. Each capability tells the walk
/// what doing it means, so the walk hands every step to itself and asks nothing — a walk that asked each step
/// what it could do was a list that every new capability had to be added to, and a list that had to agree
/// with the rules about which steps act. A step cannot implement two of them: it would be one step the run
/// could not place, and outside this library it does not compile.
/// </remarks>
public interface IActsInAWalk : IPipelineStep
{
    /// <summary>Does what this step does, at its place in the walk.</summary>
    /// <param name="walk">The walk it stands in.</param>
    internal void ActOn(Walk walk);
}

/// <summary>
/// A step that learns something from the training rows and then replays what it learned.
/// </summary>
/// <remarks>
/// A mean, the value that fills a gap, the categories an encoder knows, the bounds of a clip — each is a
/// parameter learned from the training rows alone. Saying so in the type is what lets one rule, in one
/// place, refuse such a step wherever it arrives from: the chain, the extension point, a hand-written file or
/// a notebook. A step that is arithmetic on a single row does not implement this and needs no split before it.
/// <para>
/// Fitting and applying are two separate acts on purpose. The fit sees the training rows and nothing else;
/// applying sees every row and learns nothing, so validation, test and a row arriving in production a year
/// from now are all treated with the same numbers. They are the whole of what it means to learn here, so
/// saying a step learns and not saying how is not something a step can do.
/// </para>
/// </remarks>
public interface IFittedStep : IActsInAWalk
{
    /// <summary>Learns whatever this step needs, from the training rows alone.</summary>
    /// <param name="table">The data.</param>
    /// <param name="parts">Which part each row belongs to.</param>
    /// <returns>What was learned.</returns>
    FittedStepValues Fit(Table table, IReadOnlyList<Part> parts);

    /// <summary>Applies what was learned to every row.</summary>
    /// <param name="table">The data, changed in place.</param>
    /// <param name="fitted">What the fit learned.</param>
    void ApplyTo(Table table, FittedStepValues fitted);

    /// <inheritdoc />
    void IActsInAWalk.ActOn(Walk walk) => walk.Learn(Fit, ApplyTo);
}

/// <summary>
/// A step that divides the rows into training, validation, test and a part to predict on.
/// </summary>
/// <remarks>
/// The line in the declaration. Everything an <see cref="IFittedStep"/> learns is fitted on the training
/// side of it, so nothing that learns may stand before one. Saying a step splits and working out which row
/// goes where are one thing: a split that divided nothing used to leave every row in training.
/// <para>
/// A split writes an entry of its own in the fitted half, beside what the steps below it learned: how many
/// rows went to each part and a digest of the rows it divided, which does not depend on the order they came
/// in. That is what a fit saw, and it travels with what the fit learned.
/// </para>
/// </remarks>
public interface ISplitStep : IActsInAWalk
{
    /// <summary>Works out which part every row belongs to.</summary>
    /// <param name="table">The rows to divide.</param>
    /// <returns>One part per row, in row order.</returns>
    Part[] Assign(Table table);

    /// <summary>Adds what only this kind of split can say about the rows it divided.</summary>
    /// <param name="table">The rows it divided.</param>
    /// <param name="parts">Where each of them went.</param>
    /// <param name="seen">What every split writes down: how many rows each part holds, and a digest of them.</param>
    /// <remarks>Nothing, unless the split has more to say; a split in time says where each part starts and ends.</remarks>
    void Describe(Table table, IReadOnlyList<Part> parts, FittedStepValues seen)
    {
    }

    /// <inheritdoc />
    void IActsInAWalk.ActOn(Walk walk) => walk.Divide(Assign, Describe);
}

/// <summary>
/// A step as the one line of JSON it writes itself as.
/// </summary>
internal static class PipelineStepExtensions
{
    /// <summary>The step as it writes itself, on one line with nothing between the tokens.</summary>
    /// <param name="step">The step.</param>
    /// <returns>The step's JSON, as UTF-8.</returns>
    /// <remarks>
    /// The form a step is compared and keyed by. The step writes it from what it holds, so it is the same
    /// however a file happened to space, order or spell what the step was read from.
    /// </remarks>
    public static byte[] Canonical(this IPipelineStep step)
    {
        var buffer = new ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            step.WriteTo(writer);
        }

        return buffer.WrittenSpan.ToArray();
    }
}
