// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Xml.Linq;
using DeepSharp.Notebooks.Verso;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The notebook package is installed by Verso itself, into a folder of its own, beside whatever else a person
/// has installed. What it may carry and which Verso it runs in are therefore facts about the package, and each
/// is held by a test rather than remembered: the abstractions it is compiled against decide the oldest Verso
/// that loads it, and every reference it takes is copied in with it.
/// </summary>
public class PackageTests
{
    private static readonly Assembly Package = typeof(StepCellType).Assembly;

    private static XDocument Project() =>
        XDocument.Load(Path.Join(Repository.Root, "Src", "DeepSharp.Notebooks.Verso", "DeepSharp.Notebooks.Verso.csproj"));

    [Fact]
    public void ThePackageIsCompiledAgainstTheOldestAbstractionsThatCarryWhatItCalls()
    {
        // Verso refuses an extension compiled against a newer minor than the one it runs. Compiled against
        // 1.1.0 it loads in every Verso from 1.1.0 on; a restore that quietly took a newer version would lock
        // out everyone still on an older one, and nothing else would notice.
        var abstractions = Package.GetReferencedAssemblies().Single(reference => reference.Name == "Verso.Abstractions");

        Assert.Equal(new Version(1, 1, 0, 0), abstractions.Version);
    }

    [Fact]
    public void ThePackageReferencesTheRuntimeAndAClosedListOfPackages()
    {
        // Everything the package references is installed with it into Verso's folder for it. The list is closed
        // so that a reference added for convenience is a decision somebody makes, not a surprise in every install.
        string[] allowed = ["DeepSharp.Pipelines", "DeepSharp.Pipelines.Indicators", "MatPlotLibNet", "Verso.Abstractions"];
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();

        var outside = Package.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(allowed)
            .ToArray();

        Assert.True(outside.Length == 0, $"The notebook package references {string.Join(", ", outside)}.");
    }

    [Fact]
    public void ThePackageIsFoundUnderItsTags_AndTakesTheAbstractionsAsARange()
    {
        var project = Project();
        var tags = project.Descendants("PackageTags").Single().Value.Split(';');
        var abstractions = project.Descendants("PackageReference")
            .Single(reference => reference.Attribute("Include")!.Value == "Verso.Abstractions");

        Assert.Equal("DeepSharp.Notebooks.Verso", project.Descendants("PackageId").Single().Value);
        Assert.Contains("verso", tags);
        Assert.Contains("notebook", tags);
        Assert.Contains("pdd", tags);
        Assert.Equal("[1.1.0, 2.0.0)", abstractions.Attribute("Version")!.Value);
    }
}
