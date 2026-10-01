// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Xml.Linq;
using DeepSharp.Import.Keras;
using DeepSharp.Networks;

namespace DeepSharp.Tests.Import;

/// <summary>
/// Reading what Keras saved is a package of its own, so an application that never reads a Keras file never carries an
/// HDF5 reader: what it references is a closed list, the reader it borrows is named with its licence, and nothing it hands
/// out is a type of that reader — what comes back is a network and a loss, as from any other importer.
/// </summary>
public class KerasPackageTests
{
    private static readonly Assembly Package = typeof(KerasFile).Assembly;

    [Fact]
    public void TheKerasReaderReferencesTheRuntimeTheCoreAndItsHdf5Reader_AndNothingElse()
    {
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();
        var outside = Package.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")))
            .Except(["DeepSharp", "PureHDF"])
            .ToArray();

        Assert.Equal("DeepSharp.Import.Keras", Package.GetName().Name);
        Assert.True(outside.Length == 0, $"The Keras reader references {string.Join(", ", outside)}.");
    }

    [Fact]
    public void TheReaderHandsOutOneType_TheImporter_AndNoTypeOfTheHdf5Reader()
    {
        var shown = Package.GetExportedTypes();

        Assert.Equal([typeof(KerasFile)], shown);
        Assert.Equal([typeof(IImporter)], typeof(KerasFile).GetInterfaces());
        Assert.DoesNotContain(
            typeof(KerasFile).GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly)
                .OfType<MethodBase>()
                .SelectMany(member => member.GetParameters().Select(parameter => parameter.ParameterType).Append((member as MethodInfo)?.ReturnType ?? typeof(void))),
            type => type.Namespace?.StartsWith("PureHDF", StringComparison.Ordinal) == true);
    }

    [Fact]
    public void TheReaderIsAPackage_FoundUnderItsTags_NamingTheReaderItBorrowsAndItsLicence()
    {
        var path = Path.Join(Repository.Root, "Src", "DeepSharp.Import.Keras", "DeepSharp.Import.Keras.csproj");
        var project = XDocument.Load(path);
        var tags = project.Descendants("PackageTags").Single().Value.Split(';');
        var borrowed = project.Descendants("PackageReference").Single();

        Assert.Equal("DeepSharp.Import.Keras", project.Descendants("PackageId").Single().Value);
        Assert.Contains("deepsharp", tags);
        Assert.Contains("keras", tags);
        Assert.Equal("PureHDF", borrowed.Attribute("Include")!.Value);
        Assert.Equal("2.2.0", borrowed.Attribute("Version")!.Value);
        Assert.Contains("PureHDF 2.2.0, MIT", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.Equal(["DeepSharp.csproj"], project.Descendants("ProjectReference").Select(reference => Path.GetFileName(reference.Attribute("Include")!.Value)));
    }
}
