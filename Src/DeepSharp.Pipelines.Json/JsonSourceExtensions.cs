// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// Reading rows from a JSON file holding an array of records, and teaching a catalog the verb a pipeline file names it by.
/// </summary>
/// <remarks>
/// A reader is an extension method shipped by a package of its own, so the verb exists exactly when that package is
/// referenced. .NET reads JSON itself, so this one brings the reader and nothing else.
/// </remarks>
public static class JsonSourceExtensions
{
    /// <summary>Declares that the rows come from a JSON file holding an array of records, one object a row.</summary>
    /// <param name="pipeline">The pipeline being written.</param>
    /// <param name="path">Where the file will be, when the pipeline runs; a relative path is read from the pipeline's folder.</param>
    /// <returns>The pipeline, so the next verb can be written after it.</returns>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    /// <remarks>
    /// Nothing is opened until the pipeline runs. The columns are the keys the records use, in the order they first
    /// appear, and a key a record leaves out is a gap on its row.
    /// </remarks>
    public static PipelineBuilder ReadJson(this PipelineBuilder pipeline, string path)
    {
        ArgumentNullException.ThrowIfNull(pipeline);

        return pipeline.Add(new ReadJsonStep(path));
    }

    /// <summary>Teaches a catalog to read <c>read.json</c> back out of a pipeline file.</summary>
    /// <param name="catalog">The catalog being assembled.</param>
    /// <returns>The same catalog, so registration reads as one sentence.</returns>
    public static StepCatalog WithJson(this StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);

        catalog.Register<ReadJsonStep>();

        return catalog;
    }
}

/// <summary>
/// The JSON reader's verb, offered to whichever catalog an application builds.
/// </summary>
public sealed class JsonSteps : IStepContribution
{
    /// <inheritdoc />
    public void AddTo(StepCatalog catalog) => catalog.WithJson();
}
