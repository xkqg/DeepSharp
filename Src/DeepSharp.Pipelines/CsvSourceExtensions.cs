// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// Reading rows from a comma-separated file.
/// </summary>
/// <remarks>
/// A reader is an extension method, shipped by the package that brings the dependency the format needs. CSV
/// needs none, so it travels with the pipeline itself; Parquet, Excel and the rest arrive the same way from
/// their own packages, and a verb exists exactly when its package is referenced.
/// </remarks>
public static class CsvSourceExtensions
{
    /// <summary>Declares that the rows come from a comma-separated file.</summary>
    /// <param name="pipeline">The pipeline being written.</param>
    /// <param name="path">Where the file will be, when the pipeline runs.</param>
    /// <returns>The pipeline, so the next verb can be written after it.</returns>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    public static PipelineBuilder ReadCsv(this PipelineBuilder pipeline, string path)
    {
        ArgumentNullException.ThrowIfNull(pipeline);

        return pipeline.Add(new ReadCsvStep(path));
    }
}
