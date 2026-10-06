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
        // The notebook's package carries the pipeline library, so the host reaches it without bringing anything of its own:
        // it writes a notebook whole by the one routine the core has, instead of by a copy of that routine. The decision is
        // held by the test below, which says the host depends on the engine and the notebook's package alone.
        string[] allowed = ["DeepSharp.Pipelines", "DeepSharp.Verso.Notebooks", "Verso", "Verso.Abstractions"];
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();

        var outside = typeof(NotebookHost).Assembly.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(allowed)
            .ToArray();

        Assert.True(outside.Length == 0, $"The host's package references {string.Join(", ", outside)}.");
    }

    [Fact]
    public void WhatThePackageDependsOn_IsVersosEngineAndTheNotebooksPackageAlone_TheCoreReachingTheHostThroughTheLatter()
    {
        var project = Project();

        Assert.Equal(["Verso"], project.Descendants("PackageReference").Select(reference => reference.Attribute("Include")!.Value));
        Assert.Equal(
            [@"..\DeepSharp.Verso.Notebooks\DeepSharp.Verso.Notebooks.csproj"],
            project.Descendants("ProjectReference").Select(reference => reference.Attribute("Include")!.Value));
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

    [Fact]
    public void EveryEnumTheHostNamesAfterVersos_HoldsTheNamesAndValuesVersosDoes()
    {
        // The host hands an application its own enums rather than the engine's, read across by name: an engine that adds
        // a name the host lacks would fail that read, so the two are held equal here.
        AssertSame<global::Verso.Abstractions.LayoutCapabilities, LayoutAllows>();
        AssertSame<global::Verso.Abstractions.ToolbarPlacement, ToolbarPlace>();
        AssertSame<global::Verso.Abstractions.PropertyFieldType, FieldKind>();
    }

    private static void AssertSame<TVerso, THost>()
        where TVerso : struct, Enum
        where THost : struct, Enum
    {
        Assert.Equal(Enum.GetNames<TVerso>(), Enum.GetNames<THost>());
        Assert.Equal(Enum.GetValues<TVerso>().Select(value => Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture)),
            Enum.GetValues<THost>().Select(value => Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture)));
    }
}
