// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Xml.Linq;
using DeepSharp.Import.PyTorch;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import;

/// <summary>
/// Reading what PyTorch saved is a package of its own, so a network that never meets a file of PyTorch's never carries its
/// reader: it references the core and the safetensors reader it borrows and nothing else, nothing it hands out is a type of
/// that reader, the core knows nothing of it, and nothing in it looks a type up or builds one by a name a file could hold.
/// </summary>
public class PyTorchImportPackageTests
{
    private static readonly Assembly Package = typeof(SafetensorsFile).Assembly;

    private static XDocument Project() =>
        XDocument.Load(Path.Join(Repository.Root, "Src", "DeepSharp.Import.PyTorch", "DeepSharp.Import.PyTorch.csproj"));

    [Fact]
    public void ThePackageReferencesTheCoreAndTheReaderItBorrows_AndNothingElseButTheRuntime()
    {
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();
        var outside = Package.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(["DeepSharp", "Onnxify.Safetensors"])
            .ToArray();

        Assert.Equal("DeepSharp.Import.PyTorch", Package.GetName().Name);
        Assert.True(outside.Length == 0, $"The package references {string.Join(", ", outside)}.");
        Assert.Equal(
            ["Onnxify.Safetensors 0.3.11"],
            Project().Descendants("PackageReference").Select(reference => $"{reference.Attribute("Include")!.Value} {reference.Attribute("Version")!.Value}"));
    }

    [Fact]
    public void NothingThePackageHandsOut_IsATypeOfTheReaderItBorrows()
    {
        const BindingFlags Shown = BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        var shown = Package.GetExportedTypes()
            .SelectMany(type => type.GetInterfaces()
                .Append(type.BaseType!)
                .Concat(type.GetConstructors(Shown).SelectMany(constructor => constructor.GetParameters().Select(parameter => parameter.ParameterType)))
                .Concat(type.GetMethods(Shown).SelectMany(method => method.GetParameters().Select(parameter => parameter.ParameterType).Append(method.ReturnType)))
                .Concat(type.GetProperties(Shown).Select(property => property.PropertyType))
                .SelectMany(Unwrapped)
                .Select(mentioned => $"{type.Name}: {mentioned.FullName}"))
            .Where(mention => mention.Contains(": Onnxify", StringComparison.Ordinal))
            .ToArray();

        Assert.Equal([typeof(PyTorchFile), typeof(SafetensorsFile), typeof(TorchSaveFile)], Package.GetExportedTypes().OrderBy(type => type.Name, StringComparer.Ordinal));
        Assert.True(shown.Length == 0, $"The package hands out {string.Join("; ", shown)}.");
    }

    [Fact]
    public void TheCoreKnowsNothingOfThePackage()
    {
        Assert.DoesNotContain("DeepSharp.Import.PyTorch", typeof(Tensor).Assembly.GetReferencedAssemblies().Select(reference => reference.Name));
        Assert.DoesNotContain(
            XDocument.Load(Path.Join(Repository.Root, "Src", "DeepSharp", "DeepSharp.csproj")).Descendants("ProjectReference").Select(reference => reference.Attribute("Include")!.Value),
            reference => reference.EndsWith("DeepSharp.Import.PyTorch.csproj", StringComparison.Ordinal));
    }

    [Fact]
    public void ThePackageIsFoundUnderItsTags()
    {
        var tags = Project().Descendants("PackageTags").Single().Value.Split(';');

        Assert.Equal("DeepSharp.Import.PyTorch", Project().Descendants("PackageId").Single().Value);
        Assert.Contains("deepsharp", tags);
        Assert.Contains("pytorch", tags);
        Assert.Contains("safetensors", tags);
        Assert.Contains("torch-save", tags);
        Assert.Contains("pickle", tags);
    }

    [Fact]
    public void ThePackageCallsNothingThatLooksATypeUpOrBuildsOneByName()
    {
        // Read from the package's own metadata: every member it calls outside itself, by the type that declares it.
        using var file = File.OpenRead(Package.Location);
        using var image = new PEReader(file);
        var metadata = image.GetMetadataReader();
        var called = metadata.MemberReferences
            .Select(metadata.GetMemberReference)
            .Select(member => $"{Declaring(metadata, member.Parent)}.{metadata.GetString(member.Name)}")
            .ToArray();
        // A type named by a string, a member found by one, or anything built or called through what was found; typeof, which
        // the compiler resolves, is none of them.
        string[] byName =
        [
            "System.Type.GetType", "System.Type.GetMethod", "System.Type.GetConstructor", "System.Type.GetMember", "System.Type.InvokeMember",
            "System.Reflection.MethodBase.Invoke", "System.Reflection.MethodInfo.Invoke", "System.Reflection.ConstructorInfo.Invoke",
            "System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject",
        ];
        string[] everything = ["System.Activator.", "System.Reflection.Assembly."];

        Assert.Contains("System.IO.Compression.ZipArchive..ctor", called);
        Assert.DoesNotContain(called, member => byName.Contains(member) || everything.Any(type => member.StartsWith(type, StringComparison.Ordinal)));
    }

    private static string Declaring(MetadataReader metadata, EntityHandle parent) =>
        parent.Kind == HandleKind.TypeReference
            ? $"{metadata.GetString(metadata.GetTypeReference((TypeReferenceHandle)parent).Namespace)}.{metadata.GetString(metadata.GetTypeReference((TypeReferenceHandle)parent).Name)}"
            : parent.Kind.ToString();

    private static IEnumerable<Type> Unwrapped(Type type) =>
        type.HasElementType ? Unwrapped(type.GetElementType()!)
        : type.IsGenericType ? type.GetGenericArguments().SelectMany(Unwrapped).Prepend(type)
        : [type];
}
