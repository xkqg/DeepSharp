// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Notebooks.Verso;

/// <summary>Reading a cell's text as a step, where text that is none is an answer rather than a fault.</summary>
internal static class StepCatalogExtensions
{
    /// <summary>The step a text reads as, through the catalog a pipeline file is read with.</summary>
    /// <param name="catalog">The verbs the text may use.</param>
    /// <param name="text">The text.</param>
    /// <returns>The step, or nothing when the text does not read as one.</returns>
    public static IPipelineStep? TryReadStep(this StepCatalog catalog, string text)
    {
        try
        {
            return catalog.ReadStep(text);
        }
        catch (PipelineFileException)
        {
            return null;
        }
    }
}
