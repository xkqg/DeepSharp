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

        // Every contribution is applied, whoever registered first. The obvious shape -- each package
        // registering a catalog of its own and the container keeping one -- makes the order of two lines in
        // a startup file decide which verbs exist, silently, and the error message then blames a package
        // that is present.
        services.TryAddSingleton(provider =>
        {
            var catalog = StepCatalog.BuiltIn();

            foreach (var contribution in provider.GetServices<IStepContribution>())
            {
                contribution.AddTo(catalog);
            }

            return catalog;
        });

        services.TryAddSingleton<IPipelineFactory, PipelineFactory>();

        return services;
    }
}

/// <summary>
/// A package's verbs, offered to whichever catalog an application builds.
/// </summary>
/// <remarks>
/// A package that brings verbs registers one of these rather than a catalog of its own, so every package's
/// verbs end up in the one catalog no matter which order the registrations were written in. A genuine
/// duplicate still throws where it should, in <see cref="StepCatalog.Register"/>.
/// </remarks>
public interface IStepContribution
{
    /// <summary>Teaches the catalog the verbs this package brings.</summary>
    /// <param name="catalog">The catalog being assembled.</param>
    void AddTo(StepCatalog catalog);
}
