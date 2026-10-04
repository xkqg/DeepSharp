// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.ML;

/// <summary>
/// The verb this package brings to a catalog that reads pipeline files.
/// </summary>
public static class MLStepExtensions
{
    extension(StepCatalog catalog)
    {
        /// <summary>Teaches a catalog the trainer from ML.NET a pipeline can be declared for.</summary>
        /// <returns>The catalog, so verbs are taught one after another.</returns>
        /// <exception cref="ArgumentNullException">There is no catalog.</exception>
        /// <exception cref="InvalidOperationException">It already knows the verb.</exception>
        public StepCatalog WithML()
        {
            ArgumentNullException.ThrowIfNull(catalog);

            catalog.Register<LearnMLStep>();

            return catalog;
        }
    }
}

/// <summary>The verb this package brings, for a host that assembles its catalog from what it finds.</summary>
public sealed class MLSteps : IStepContribution
{
    /// <inheritdoc />
    public void AddTo(StepCatalog catalog) => catalog.WithML();
}
