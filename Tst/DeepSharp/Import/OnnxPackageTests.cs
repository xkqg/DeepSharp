// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Xml.Linq;
using DeepSharp.Import.Onnx;
using DeepSharp.Networks;

namespace DeepSharp.Tests.Import;

/// <summary>
/// Reading an ONNX graph is a package of its own, so an application that never reads one never carries a protobuf reader:
/// what it references is a closed list, the reader it borrows is named with its licence and with what it brings, and
/// nothing it hands out is a type of that reader — what comes back is a network and a loss, as from any other importer.
/// </summary>
public class OnnxPackageTests
{
    private static readonly Assembly Package = typeof(OnnxFile).Assembly;

    [Fact]
    public void TheOnnxReaderReferencesTheRuntimeTheCoreAndItsProtobufReader_AndNothingElse()
    {
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();
        var outside = Package.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(["DeepSharp", "OnnxSharp", "Google.Protobuf"])
            .ToArray();

        Assert.Equal("DeepSharp.Import.Onnx", Package.GetName().Name);
        Assert.True(outside.Length == 0, $"The ONNX reader references {string.Join(", ", outside)}.");
    }

    [Fact]
    public void TheReaderHandsOutOneType_TheImporter_AndNoTypeOfTheProtobufReader()
    {
        var shown = Package.GetExportedTypes();

        Assert.Equal([typeof(OnnxFile)], shown);
        Assert.Equal([typeof(IImporter)], typeof(OnnxFile).GetInterfaces());
        Assert.DoesNotContain(
            typeof(OnnxFile).GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly)
                .OfType<MethodBase>()
                .SelectMany(member => member.GetParameters().Select(parameter => parameter.ParameterType).Append((member as MethodInfo)?.ReturnType ?? typeof(void))),
            type => type.Namespace is "Onnx" || type.Namespace?.StartsWith("Google.Protobuf", StringComparison.Ordinal) == true);
    }

    [Fact]
    public void TheReaderIsAPackage_FoundUnderItsTags_NamingTheReaderItBorrows_WhatThatBrings_AndTheirLicences()
    {
        var path = Path.Join(Repository.Root, "Src", "DeepSharp.Import.Onnx", "DeepSharp.Import.Onnx.csproj");
        var project = XDocument.Load(path);
        var text = File.ReadAllText(path);
        var tags = project.Descendants("PackageTags").Single().Value.Split(';');
        var borrowed = project.Descendants("PackageReference").Single();

        Assert.Equal("DeepSharp.Import.Onnx", project.Descendants("PackageId").Single().Value);
        Assert.Contains("deepsharp", tags);
        Assert.Contains("onnx", tags);
        Assert.Equal("OnnxSharp", borrowed.Attribute("Include")!.Value);
        Assert.Equal("0.3.2", borrowed.Attribute("Version")!.Value);
        Assert.Contains("OnnxSharp 0.3.2, MIT", text, StringComparison.Ordinal);
        Assert.Contains("Google.Protobuf 3.29.3, BSD-3-Clause", text, StringComparison.Ordinal);
        Assert.Equal("true", project.Descendants("EnableTrimAnalyzer").Single().Value);
        Assert.Equal(["DeepSharp.csproj"], project.Descendants("ProjectReference").Select(reference => Path.GetFileName(reference.Attribute("Include")!.Value.Replace('\\', '/'))));
    }
}
