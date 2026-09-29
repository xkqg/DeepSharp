// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Xml.Linq;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The bridge is the one place a network and a pipeline know each other: it references both and nothing else, and neither
/// of them references it — so a service that only prepares data, or only trains on tensors, carries none of the other.
/// </summary>
public class NetworkBridgePackageTests
{
    private static readonly Assembly Bridge = typeof(TrainedNetwork).Assembly;

    [Fact]
    public void TheBridgeReferencesTheNetworkAndThePipeline_AndNothingElseButTheRuntime()
    {
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();
        var outside = Bridge.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(["DeepSharp", "DeepSharp.Pipelines"])
            .ToArray();

        Assert.Equal("DeepSharp.Learners.Networks", Bridge.GetName().Name);
        Assert.True(outside.Length == 0, $"The bridge references {string.Join(", ", outside)}.");
    }

    [Fact]
    public void NeitherTheNetworkNorThePipelineKnowsTheBridge()
    {
        Assert.DoesNotContain("DeepSharp.Learners.Networks", typeof(Tensor).Assembly.GetReferencedAssemblies().Select(reference => reference.Name));
        Assert.DoesNotContain("DeepSharp.Learners.Networks", typeof(Pdd).Assembly.GetReferencedAssemblies().Select(reference => reference.Name));

        foreach (var project in new[] { Path.Join("Src", "DeepSharp", "DeepSharp.csproj"), Path.Join("Src", "DeepSharp.Pipelines", "DeepSharp.Pipelines.csproj") })
        {
            var references = XDocument.Load(Path.Join(Repository.Root, project)).Descendants("ProjectReference").Select(reference => reference.Attribute("Include")?.Value ?? string.Empty);

            Assert.DoesNotContain(references, reference => reference.EndsWith("DeepSharp.Learners.Networks.csproj", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void TheBridgeIsAPackage_FoundUnderItsTags()
    {
        var project = XDocument.Load(Path.Join(Repository.Root, "Src", "DeepSharp.Learners.Networks", "DeepSharp.Learners.Networks.csproj"));
        var tags = project.Descendants("PackageTags").Single().Value.Split(';');

        Assert.Equal("DeepSharp.Learners.Networks", project.Descendants("PackageId").Single().Value);
        Assert.Contains("deepsharp", tags);
        Assert.Contains("pipeline", tags);
    }
}
