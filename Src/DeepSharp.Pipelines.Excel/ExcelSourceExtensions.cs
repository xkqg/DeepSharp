// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// Reading rows from a sheet of an Excel workbook, and teaching a catalog the verb a pipeline file names it by.
/// </summary>
/// <remarks>
/// A reader is an extension method shipped by the package that brings what the format needs, so the verb exists exactly
/// when this package is referenced, and a project that reads no workbook carries none of it.
/// </remarks>
public static class ExcelSourceExtensions
{
    extension(PipelineBuilder pipeline)
    {
        /// <summary>Declares that the rows come from the first sheet of an Excel workbook.</summary>
        /// <param name="path">
        /// Where the workbook — .xlsx, .xls or .xlsb — will be, when the pipeline runs; a relative path is read from the
        /// pipeline's folder.
        /// </param>
        /// <returns>The pipeline, so the next verb can be written after it.</returns>
        /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
        /// <remarks>Nothing is opened until the pipeline runs. The sheet's first row names its columns.</remarks>
        public PipelineBuilder ReadExcel(string path)
        {
            ArgumentNullException.ThrowIfNull(pipeline);

            return pipeline.Add(new ReadExcelStep(path));
        }

        /// <summary>Declares that the rows come from a named sheet of an Excel workbook.</summary>
        /// <param name="path">Where the workbook will be, when the pipeline runs; a relative path is read from the pipeline's folder.</param>
        /// <param name="sheet">The sheet the rows are on; empty, or nothing but spaces, for the first.</param>
        /// <returns>The pipeline, so the next verb can be written after it.</returns>
        /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
        /// <remarks>A workbook without a sheet of that name is refused when the pipeline runs, naming the sheets it has.</remarks>
        public PipelineBuilder ReadExcel(string path, string sheet)
        {
            ArgumentNullException.ThrowIfNull(pipeline);

            return pipeline.Add(new ReadExcelStep(path, sheet));
        }
    }

    extension(StepCatalog catalog)
    {
        /// <summary>Teaches a catalog to read <c>read.excel</c> back out of a pipeline file.</summary>
        /// <returns>The same catalog, so registration reads as one sentence.</returns>
        public StepCatalog WithExcel()
        {
            ArgumentNullException.ThrowIfNull(catalog);

            catalog.Register<ReadExcelStep>();

            return catalog;
        }
    }
}

/// <summary>
/// The Excel reader's verb, offered to whichever catalog an application builds.
/// </summary>
public sealed class ExcelSteps : IStepContribution
{
    /// <inheritdoc />
    public void AddTo(StepCatalog catalog) => catalog.WithExcel();
}
