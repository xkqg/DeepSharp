// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace DeepSharp.Pipelines;

/// <summary>
/// Registering pipelines with an application that has a host, its services and an app object.
/// </summary>
/// <remarks>
/// This is a door, never a requirement. The library depends on the dependency-injection abstractions and on
/// no container at all, so a console program that writes <c>Pdd.Create()</c> with no host anywhere stays a
/// first-class way to use it. What is registered here holds no state between pipelines: the factory is
/// shared, everything it hands out is its own.
/// </remarks>
public static class ServiceCollectionExtensions
{
    /// <summary>Adds the pipeline factory and the catalog of verbs to an application's services.</summary>
    /// <param name="services">The services being assembled.</param>
    /// <returns>The same collection, so registration reads as one sentence.</returns>
    /// <remarks>
    /// Registering twice adds nothing twice. Two factories would be harmless and two catalogs would not:
    /// a host that taught one of them a verb would be left with a file that reads differently depending on
    /// which of the two answered.
    /// </remarks>
    public static IServiceCollection AddDeepSharpPipelines(this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);

        services.TryAddSingleton(_ => StepCatalog.BuiltIn());
        services.TryAddSingleton<IPipelineFactory, PipelineFactory>();

        return services;
    }
}
