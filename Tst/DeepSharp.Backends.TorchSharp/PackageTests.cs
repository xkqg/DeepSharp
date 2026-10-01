// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Xml.Linq;
using DeepSharp.Backends.TorchSharp;
using TorchSharp;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// What the engine's package brings into an application: DeepSharp and TorchSharp, at the version its tests run on, and
/// nothing else — never a libtorch, which is the application's to bring — and nothing of TorchSharp's in what it publishes,
/// so a model written against the seam never names the engine underneath. Each is held by a test rather than remembered.
/// </summary>
public class PackageTests
{
    private static XDocument Project() =>
        XDocument.Load(Path.Join(Repository.Root, "Src", "DeepSharp.Backends.TorchSharp", "DeepSharp.Backends.TorchSharp.csproj"));

    [Fact]
    public void ThePackageReferencesTorchSharpAlone_AtTheVersionItsTestsRunOn_AndBringsNoLibtorch()
    {
        var project = Project();
        var packages = project.Descendants("PackageReference").Select(reference => $"{reference.Attribute("Include")!.Value} {reference.Attribute("Version")!.Value}");
        var projects = project.Descendants("ProjectReference").Select(reference => reference.Attribute("Include")!.Value);

        Assert.Equal(["TorchSharp 0.107.0"], packages);
        Assert.Equal([@"..\DeepSharp\DeepSharp.csproj"], projects);
        Assert.Equal(Libtorch.TorchSharpVersion, typeof(torch).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()!.InformationalVersion.Split('+')[0]);
    }

    [Fact]
    public void TheAssemblyReferencesTheRuntimeDeepSharpAndTorchSharp_AndNothingElse()
    {
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();

        var outside = typeof(TorchBackend).Assembly.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Order(StringComparer.Ordinal);

        Assert.Equal(["DeepSharp", "TorchSharp"], outside);
    }

    [Fact]
    public void NoTypeOfTorchSharps_StandsInAnythingThePackagePublishes()
    {
        var published = typeof(TorchBackend).Assembly.GetExportedTypes();
        var named = published.SelectMany(Named).Where(type => type.Assembly == typeof(torch).Assembly).Select(type => type.FullName).Distinct();

        Assert.Equal([typeof(TorchBackend)], published);
        Assert.Empty(named);
    }

    [Fact]
    public void ThePackageIsFoundUnderItsTags()
    {
        var project = Project();
        var tags = project.Descendants("PackageTags").Single().Value.Split(';');

        Assert.Equal("DeepSharp.Backends.TorchSharp", project.Descendants("PackageId").Single().Value);
        Assert.Contains("torchsharp", tags);
        Assert.Contains("libtorch", tags);
        Assert.Contains("cuda", tags);
    }

    // Every type a published type names where a caller sees it: what it derives from and implements, and every public or
    // protected member's type, parameters and return, generic arguments included.
    private static IEnumerable<Type> Named(Type type)
    {
        const BindingFlags Seen = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        var members = type.GetMembers(Seen).Where(member => member switch
        {
            MethodBase method => method.IsPublic || method.IsFamily || method.IsFamilyOrAssembly,
            FieldInfo field => field.IsPublic || field.IsFamily || field.IsFamilyOrAssembly,
            PropertyInfo property => property.GetAccessors(nonPublic: true).Any(accessor => accessor.IsPublic || accessor.IsFamily || accessor.IsFamilyOrAssembly),
            EventInfo happening => happening.AddMethod is { } add && (add.IsPublic || add.IsFamily),
            Type nested => nested.IsNestedPublic || nested.IsNestedFamily,
            _ => false,
        });

        var named = new List<Type> { type };
        named.AddRange(type.GetInterfaces());

        if (type.BaseType is { } baseType)
        {
            named.Add(baseType);
        }

        foreach (var member in members)
        {
            named.AddRange(member switch
            {
                MethodInfo method => [method.ReturnType, .. method.GetParameters().Select(parameter => parameter.ParameterType)],
                ConstructorInfo constructor => constructor.GetParameters().Select(parameter => parameter.ParameterType),
                FieldInfo field => [field.FieldType],
                PropertyInfo property => [property.PropertyType, .. property.GetIndexParameters().Select(parameter => parameter.ParameterType)],
                EventInfo happening => [happening.EventHandlerType!],
                _ => [],
            });
        }

        return named.SelectMany(Unwrapped);
    }

    // A type and every type it is made of: an array's element, a reference's target, a generic's arguments.
    private static IEnumerable<Type> Unwrapped(Type type) =>
        type.HasElementType
            ? Unwrapped(type.GetElementType()!)
            : type.IsGenericType
                ? [type.GetGenericTypeDefinition(), .. type.GetGenericArguments().SelectMany(Unwrapped)]
                : [type];
}
