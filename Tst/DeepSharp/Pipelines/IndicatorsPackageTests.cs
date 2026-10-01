// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Xml.Linq;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Financial and signal-processing indicators are a package of their own, so a pipeline that adds none never carries
/// them: what the package references is a closed list — the arithmetic they are borrowed from, <c>MatPlotLibNet</c>,
/// and the frame it works on, <c>Microsoft.Data.Analysis</c>, through the door shared with the frame reader,
/// <c>MatPlotLibNet.DataFrame</c>.
/// </summary>
public class IndicatorsPackageTests
{
    private static readonly Assembly Package = typeof(AddIndicatorStep).Assembly;

    [Fact]
    public void TheIndicatorsReferenceThePipelineAndTheFrame_AndNothingElseButTheRuntime()
    {
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();
        var outside = Package.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(["DeepSharp.Pipelines", "MatPlotLibNet.DataFrame", "Microsoft.Data.Analysis", "MatPlotLibNet"])
            .ToArray();

        Assert.Equal("DeepSharp.Pipelines.Indicators", Package.GetName().Name);
        Assert.True(outside.Length == 0, $"The indicators reference {string.Join(", ", outside)}.");
    }

    [Fact]
    public void ThePackageIsFoundUnderItsTags()
    {
        var project = XDocument.Load(Path.Join(Repository.Root, "Src", "DeepSharp.Pipelines.Indicators", "DeepSharp.Pipelines.Indicators.csproj"));
        var tags = project.Descendants("PackageTags").Single().Value.Split(';');

        Assert.Equal("DeepSharp.Pipelines.Indicators", project.Descendants("PackageId").Single().Value);
        Assert.Contains("indicators", tags);
    }
}
