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
/// </remarks>
public interface IPipelineStep
{
    /// <summary>The name this step is written under, in a file and in a catalog. Lowercase, dotted.</summary>
    string Verb { get; }

    /// <summary>Writes this step as one JSON object, its verb included.</summary>
    /// <param name="writer">The writer positioned where the object belongs.</param>
    void WriteTo(Utf8JsonWriter writer);
}
