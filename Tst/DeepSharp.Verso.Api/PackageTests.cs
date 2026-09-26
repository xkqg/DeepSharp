// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Xml.Linq;
using DeepSharp.Verso.Api;

namespace DeepSharp.Tests.Api;

/// <summary>
/// What the package an application hosts notebooks with brings into that application: Verso's engine, at the version
/// its tests run on, the notebook's own package, and nothing else — each held by a test rather than remembered, so a
/// reference added for convenience is a decision somebody makes, not a surprise in every application.
/// </summary>
public class PackageTests
{
    private static XDocument Project() =>
        XDocument.Load(Path.Join(Repository.Root, "Src", "DeepSharp.Verso.Api", "DeepSharp.Verso.Api.csproj"));

    [Fact]
    public void ThePackageReferencesTheRuntimeAndAClosedListOfPackages()
    {
        string[] allowed = ["DeepSharp.Verso.Notebooks", "Verso", "Verso.Abstractions"];
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();

        var outside = typeof(NotebookHost).Assembly.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(allowed)
            .ToArray();

        Assert.True(outside.Length == 0, $"The host's package references {string.Join(", ", outside)}.");
    }

    [Fact]
    public void ThePackageIsFoundUnderItsTags_AndTakesTheEngineAtTheVersionItsTestsRunOn()
    {
        var project = Project();
        var tags = project.Descendants("PackageTags").Single().Value.Split(';');
        var engine = project.Descendants("PackageReference").Single(reference => reference.Attribute("Include")!.Value == "Verso");

        Assert.Equal("DeepSharp.Verso.Api", project.Descendants("PackageId").Single().Value);
        Assert.Contains("verso", tags);
        Assert.Contains("notebook", tags);
        Assert.Contains("hosting", tags);
        Assert.Equal("1.2.2", engine.Attribute("Version")!.Value);
        Assert.Equal(typeof(global::Verso.Scaffold).Assembly.GetName().Version!.ToString(3), engine.Attribute("Version")!.Value);
    }
}
