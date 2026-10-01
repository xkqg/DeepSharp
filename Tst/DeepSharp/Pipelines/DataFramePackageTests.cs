// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Xml.Linq;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Reading rows out of Microsoft's DataFrame, or out of anything that can fill one, is a package of its own, so a
/// pipeline that never meets a frame never carries it: what the package references is a closed list — the frame
/// itself, <c>Microsoft.Data.Analysis</c>, stands in the reader's own public surface, and the door to it,
/// <c>MatPlotLibNet.DataFrame</c>, is shared with the indicators.
/// </summary>
public class DataFramePackageTests
{
    private static readonly Assembly Package = typeof(DataFrameRowSource).Assembly;

    [Fact]
    public void TheFrameReaderReferencesThePipelineAndTheFrame_AndNothingElseButTheRuntime()
    {
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();
        var outside = Package.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(["DeepSharp.Pipelines", "MatPlotLibNet.DataFrame", "Microsoft.Data.Analysis"])
            .ToArray();

        Assert.Equal("DeepSharp.Pipelines.DataFrame", Package.GetName().Name);
        Assert.True(outside.Length == 0, $"The frame reader references {string.Join(", ", outside)}.");
    }

    [Fact]
    public void ThePackageIsFoundUnderItsTags()
    {
        var project = XDocument.Load(Path.Join(Repository.Root, "Src", "DeepSharp.Pipelines.DataFrame", "DeepSharp.Pipelines.DataFrame.csproj"));
        var tags = project.Descendants("PackageTags").Single().Value.Split(';');

        Assert.Equal("DeepSharp.Pipelines.DataFrame", project.Descendants("PackageId").Single().Value);
        Assert.Contains("dataframe", tags);
    }
}
