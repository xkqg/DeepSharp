// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Microsoft.Extensions.DependencyInjection;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// An application that already has a host, its services and an app object should be able to reach a
/// pipeline the way it reaches everything else. What this must not become is a library that only works
/// inside a container: a console program with <c>new</c> stays a first-class way to use it, so these tests
/// hold both doors open and check that what comes out of the container carries no shared state.
/// </summary>
public class ServiceRegistrationTests
{
    [Fact]
    public void AHostThatAsksForAPipelineFactory_GetsOne()
    {
        var services = new ServiceCollection();

        services.AddDeepSharpPipelines();

        using var provider = services.BuildServiceProvider();
        Assert.NotNull(provider.GetRequiredService<IPipelineFactory>());
    }

    [Fact]
    public void TheCatalogIsResolvable_SoAHostCanTeachItANewVerb()
    {
        var services = new ServiceCollection();
        services.AddDeepSharpPipelines();

        using var provider = services.BuildServiceProvider();
        var catalog = provider.GetRequiredService<StepCatalog>();

        Assert.True(catalog.Knows("read.csv"));
        Assert.False(catalog.Knows("read.avro"));
    }

    [Fact]
    public void EveryPipelineFromTheFactory_IsItsOwn()
    {
        // The factory is shared; what it hands out is not. A builder that carried state from the one
        // before it would make two pipelines in one process quietly contaminate each other.
        var services = new ServiceCollection();
        services.AddDeepSharpPipelines();

        using var provider = services.BuildServiceProvider();
        var factory = provider.GetRequiredService<IPipelineFactory>();

        var first = factory.Create().ReadCsv("first.csv");
        var second = factory.Create();

        Assert.Single(first.Declaration.Steps);
        Assert.Empty(second.Declaration.Steps);
    }

    [Fact]
    public void RegisteringTwice_DoesNotGiveTheHostTwoOfEverything()
    {
        var services = new ServiceCollection();

        services.AddDeepSharpPipelines();
        services.AddDeepSharpPipelines();

        using var provider = services.BuildServiceProvider();
        Assert.Single(provider.GetServices<IPipelineFactory>());
    }

    [Fact]
    public void APackagesVerbsSurvive_WhicheverOrderTheRegistrationsAreWrittenIn()
    {
        // Two lines in a startup file used to decide which verbs existed: whichever registered a catalog
        // first won, the other package's verbs vanished, and the file that then refused to load blamed a
        // package that was right there.
        foreach (var coreFirst in new[] { true, false })
        {
            var services = new ServiceCollection();

            if (coreFirst)
            {
                services.AddDeepSharpPipelines();
                services.AddSingleton<IStepContribution, ScalingSteps>();
            }
            else
            {
                services.AddSingleton<IStepContribution, ScalingSteps>();
                services.AddDeepSharpPipelines();
            }

            using var provider = services.BuildServiceProvider();
            var catalog = provider.GetRequiredService<StepCatalog>();

            Assert.True(catalog.Knows("scale.by"), $"core first: {coreFirst}");
            Assert.True(catalog.Knows("read.csv"), $"core first: {coreFirst}");
        }
    }

    /// <summary>A package bringing one verb, registered the way another package would register its own.</summary>
    private sealed class ScalingSteps : IStepContribution
    {
        public void AddTo(StepCatalog catalog) =>
            catalog.Register("scale.by", _ => new ReadCsvStep("scaled.csv"));
    }

    [Fact]
    public void AndNoneOfThatIsRequired()
    {
        // The same pipeline, with no container anywhere. If this ever stops compiling, the library has
        // grown a dependency on being hosted.
        var builder = Pdd.Create().ReadCsv("btceur-1d.csv");

        Assert.Single(builder.Declaration.Steps);
    }
}
