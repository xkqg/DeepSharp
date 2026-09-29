// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Xml.Linq;
using DeepSharp.Charts;

namespace DeepSharp.Tests.Charts;

/// <summary>
/// The charts are a package of their own, so a trainer on a machine with no screen never carries a renderer: what it may
/// reference is a closed list, and nothing of the drawing library it borrows shows in what it hands out — a chart comes
/// out as the text of an SVG, so the library underneath can change without a line of anybody's code changing with it.
/// </summary>
public class ChartsPackageTests
{
    private static readonly Assembly Package = typeof(HistoryCharts).Assembly;

    [Fact]
    public void TheChartsReferenceTheRuntimeAndAClosedListOfPackages()
    {
        string[] allowed = ["DeepSharp", "DeepSharp.Pipelines", "MatPlotLibNet"];

        Assert.Equal("DeepSharp.Charts", Package.GetName().Name);
        Assert.Empty(Outside(Package, allowed));
    }

    [Fact]
    public void NothingTheChartsHandOut_IsATypeOfTheDrawingLibrary()
    {
        var shown = Package.GetExportedTypes()
            .SelectMany(type => Mentioned(type).Select(mentioned => $"{type.Name}: {mentioned.FullName}"))
            .Where(mention => mention.Contains(": MatPlotLibNet", StringComparison.Ordinal))
            .ToArray();

        Assert.NotEmpty(Package.GetExportedTypes());
        Assert.True(shown.Length == 0, $"The charts hand out {string.Join("; ", shown)}.");
    }

    [Fact]
    public void TheCoreReferencesTheRuntimeAndAClosedListOfPackages()
    {
        // The engine travels inside an application: the vector maths it runs on, and nothing else.
        Assert.Empty(Outside(typeof(DeepSharp.Tensors.Tensor).Assembly, ["System.Numerics.Tensors"]));
    }

    [Fact]
    public void TheChartsAreAPackage_FoundUnderItsTags()
    {
        var project = XDocument.Load(Path.Join(Repository.Root, "Src", "DeepSharp.Charts", "DeepSharp.Charts.csproj"));
        var tags = project.Descendants("PackageTags").Single().Value.Split(';');

        Assert.Equal("DeepSharp.Charts", project.Descendants("PackageId").Single().Value);
        Assert.Contains("deepsharp", tags);
        Assert.Contains("charts", tags);
    }

    // The references of an assembly that are neither the runtime itself nor on the list.
    private static string[] Outside(Assembly assembly, IEnumerable<string> allowed)
    {
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();

        return [.. assembly.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(allowed)];
    }

    // Every type a public type shows: what it derives from and implements, and what its public and protected members take
    // and give, generic arguments and element types included.
    private static IEnumerable<Type> Mentioned(Type type)
    {
        const BindingFlags Shown = BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        IEnumerable<Type> direct =
        [
            .. type.BaseType is { } baseType ? [baseType] : Array.Empty<Type>(),
            .. type.GetInterfaces(),
            .. type.GetMethods(Shown | BindingFlags.NonPublic).Where(method => method.IsPublic || method.IsFamily)
                .SelectMany(method => method.GetParameters().Select(parameter => parameter.ParameterType).Append(method.ReturnType)),
            .. type.GetProperties(Shown).Select(property => property.PropertyType),
            .. type.GetFields(Shown).Select(field => field.FieldType),
            .. type.GetConstructors(Shown).SelectMany(constructor => constructor.GetParameters().Select(parameter => parameter.ParameterType)),
        ];

        return direct.SelectMany(Unwrapped);
    }

    private static IEnumerable<Type> Unwrapped(Type type) =>
        type.HasElementType ? Unwrapped(type.GetElementType()!)
        : type.IsGenericType ? type.GetGenericArguments().SelectMany(Unwrapped).Prepend(type)
        : [type];
}
