// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Xml.Linq;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The first cut in the library runs between preparing data and learning from it, and it is referenced one
/// way only: something that learns may know the pipeline, the pipeline may know nothing that learns. A
/// direction like that is never kept by intention — it is kept by a test that fails the day somebody adds
/// the reference that would have been convenient.
/// </summary>
public class PackageBoundaryTests
{
    [Fact]
    public void ThePipelineDoesNotKnowTheTensors()
    {
        // A service that prepares data and hands it to a trainer from somewhere else should not be made to
        // carry an engine it never calls.
        var referenced = typeof(Pdd).Assembly.GetReferencedAssemblies().Select(assembly => assembly.Name);

        Assert.DoesNotContain("DeepSharp", referenced);
    }

    [Fact]
    public void AndTheTensorsDoNotKnowThePipeline()
    {
        // The other direction is just as wrong, and for the mirror reason: a model training on data
        // somebody else prepared should not carry a CSV reader.
        var referenced = typeof(Tensor).Assembly.GetReferencedAssemblies().Select(assembly => assembly.Name);

        Assert.DoesNotContain("DeepSharp.Pipelines", referenced);
    }

    [Fact]
    public void AndNeitherProjectEvenAsksForTheOther()
    {
        // The metadata test above fires one commit late: the compiler leaves an unused assembly out of the
        // emitted references entirely, so a project reference that nothing has called yet is invisible to
        // it. The reference itself is what a person adds, so that is what this reads.
        foreach (var (project, forbidden) in new[]
                 {
                     (Path.Join("Src", "DeepSharp.Pipelines", "DeepSharp.Pipelines.csproj"), "DeepSharp.csproj"),
                     (Path.Join("Src", "DeepSharp", "DeepSharp.csproj"), "DeepSharp.Pipelines.csproj"),
                 })
        {
            var references = XDocument.Load(Path.Join(Repository.Root, project))
                .Descendants("ProjectReference")
                .Select(reference => reference.Attribute("Include")?.Value ?? string.Empty)
                .ToArray();

            Assert.DoesNotContain(references, reference => reference.EndsWith(forbidden, StringComparison.Ordinal));
        }
    }

    /// <summary>The one package the pipeline core may take beside the runtime: the container abstractions.</summary>
    private static readonly string[] CorePackages = ["Microsoft.Extensions.DependencyInjection.Abstractions"];

    /// <summary>The references of an assembly that are neither the runtime itself nor on a given list.</summary>
    private static string[] Outside(System.Reflection.Assembly assembly, IEnumerable<string> allowed)
    {
        // The runtime's own assemblies sit in one folder, so "part of .NET" is a fact that can be looked up
        // rather than a prefix that a package called System.Something would also match.
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();

        return [.. assembly.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(allowed)];
    }

    [Fact]
    public void ThePipelineCoreReferencesTheRuntimeAndAClosedListOfPackages()
    {
        // Naming the one forbidden reference catches the one mistake somebody already made. A closed list
        // catches the next one too: a drawing library, a data frame, a notebook host — each of them is a
        // package of its own, and the core that travels inside an application carries none of them.
        var outside = Outside(typeof(Pdd).Assembly, CorePackages);

        Assert.True(outside.Length == 0, $"The pipeline core references {string.Join(", ", outside)}.");

        // And the check is a real filter rather than one nothing could fail: a satellite that carries a
        // dependency of its own is caught by it.
        Assert.NotEmpty(Outside(typeof(AddIndicatorStep).Assembly, [.. CorePackages, "DeepSharp.Pipelines"]));
    }

    [Fact]
    public void TheyAreTwoAssemblies_NotOneWithTwoNamespaces()
    {
        Assert.NotEqual(typeof(Pdd).Assembly, typeof(Tensor).Assembly);
        Assert.Equal("DeepSharp.Pipelines", typeof(Pdd).Assembly.GetName().Name);
        Assert.Equal("DeepSharp", typeof(Tensor).Assembly.GetName().Name);
    }
}
