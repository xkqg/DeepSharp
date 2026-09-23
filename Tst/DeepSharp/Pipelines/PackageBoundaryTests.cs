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
            var references = XDocument.Load(Path.Join(RepoRoot(), project))
                .Descendants("ProjectReference")
                .Select(reference => reference.Attribute("Include")?.Value ?? string.Empty)
                .ToArray();

            Assert.DoesNotContain(references, reference => reference.EndsWith(forbidden, StringComparison.Ordinal));
        }
    }

    private static string RepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);

        while (directory is not null && !File.Exists(Path.Join(directory.FullName, "DeepSharp.slnx")))
        {
            directory = directory.Parent;
        }

        Assert.NotNull(directory);
        return directory!.FullName;
    }

    [Fact]
    public void TheyAreTwoAssemblies_NotOneWithTwoNamespaces()
    {
        Assert.NotEqual(typeof(Pdd).Assembly, typeof(Tensor).Assembly);
        Assert.Equal("DeepSharp.Pipelines", typeof(Pdd).Assembly.GetName().Name);
        Assert.Equal("DeepSharp", typeof(Tensor).Assembly.GetName().Name);
    }
}
