// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

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
}

/// <summary>
/// A step that learns something from the data.
/// </summary>
/// <remarks>
/// A mean, the value that fills a gap, the categories an encoder knows, the bounds of a clip — each is a
/// parameter learned from the training rows alone. Saying so in the type is what lets one rule, in one
/// place, refuse such a step wherever it arrives from: the chain, the extension point, or a hand-written
/// file. A step that is arithmetic on a single row does not implement this and needs no split before it.
/// </remarks>
public interface IFittedStep : IPipelineStep;

/// <summary>
/// A step that divides the rows into training, validation and test.
/// </summary>
/// <remarks>
/// The line in the declaration. Everything an <see cref="IFittedStep"/> learns is fitted on the training
/// side of it, so nothing that learns may stand before one.
/// </remarks>
public interface ISplitStep : IPipelineStep;
