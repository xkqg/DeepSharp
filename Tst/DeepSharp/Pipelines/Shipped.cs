// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Every verb DeepSharp's packages bring, said once for every test that holds all of them to one rule: the pipeline's own,
/// and those of each package that brings more.
/// </summary>
/// <remarks>
/// Each such test named the packages itself, and a package added later was held to none of those rules until somebody
/// remembered every list.
/// </remarks>
internal static class Shipped
{
    /// <summary>What each package that brings verbs offers a catalog.</summary>
    public static IReadOnlyList<IStepContribution> Contributions { get; } =
        [new IndicatorSteps(), new ParquetSteps(), new ExcelSteps(), new JsonSteps(), new NetworkSteps()];

    /// <summary>The assemblies the steps are defined in: the pipeline's and each package's.</summary>
    public static IReadOnlyList<Assembly> StepAssemblies { get; } =
        [typeof(Pdd).Assembly, .. Contributions.Select(contribution => contribution.GetType().Assembly)];

    /// <summary>A catalog of every verb there is.</summary>
    /// <returns>A new catalog.</returns>
    public static StepCatalog Catalog()
    {
        var catalog = StepCatalog.BuiltIn();

        foreach (var contribution in Contributions)
        {
            contribution.AddTo(catalog);
        }

        return catalog;
    }
}
