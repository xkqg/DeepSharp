// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// Reading rows from an Apache Parquet file, and teaching a catalog the verb a pipeline file names it by.
/// </summary>
/// <remarks>
/// A reader is an extension method shipped by the package that brings what the format needs, so the verb exists exactly
/// when this package is referenced, and a project that reads no Parquet carries none of it.
/// </remarks>
public static class ParquetSourceExtensions
{
    /// <summary>Declares that the rows come from an Apache Parquet file.</summary>
    /// <param name="pipeline">The pipeline being written.</param>
    /// <param name="path">Where the file will be, when the pipeline runs; a relative path is read from the pipeline's folder.</param>
    /// <returns>The pipeline, so the next verb can be written after it.</returns>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    /// <remarks>
    /// Nothing is opened until the pipeline runs. The file says what each of its columns holds, and
    /// <see cref="PipelineBuilder.ProposedKinds"/> proposes what it says.
    /// </remarks>
    public static PipelineBuilder ReadParquet(this PipelineBuilder pipeline, string path)
    {
        ArgumentNullException.ThrowIfNull(pipeline);

        return pipeline.Add(new ReadParquetStep(path));
    }

    /// <summary>Teaches a catalog to read <c>read.parquet</c> back out of a pipeline file.</summary>
    /// <param name="catalog">The catalog being assembled.</param>
    /// <returns>The same catalog, so registration reads as one sentence.</returns>
    public static StepCatalog WithParquet(this StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);

        catalog.Register<ReadParquetStep>();

        return catalog;
    }
}

/// <summary>
/// The Parquet reader's verb, offered to whichever catalog an application builds.
/// </summary>
public sealed class ParquetSteps : IStepContribution
{
    /// <inheritdoc />
    public void AddTo(StepCatalog catalog) => catalog.WithParquet();
}
