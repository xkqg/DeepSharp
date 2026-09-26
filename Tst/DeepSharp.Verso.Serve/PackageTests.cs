// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Xml.Linq;
using DeepSharp.Verso.Serve;
using Microsoft.AspNetCore.Builder;

namespace DeepSharp.Tests.Serve;

/// <summary>
/// What <c>deepsharp-serve</c> is as a package: a .NET tool, told to pack at all — the web SDK packs nothing otherwise
/// and says so only in a warning — told to keep starting on the newer runtime once .NET 8 is gone, and carrying the
/// runtime, ASP.NET, Verso's engine through the host's package, and a closed list of anything else.
/// </summary>
public class PackageTests
{
    private static XDocument Project() =>
        XDocument.Load(Path.Join(Repository.Root, "Src", "DeepSharp.Verso.Serve", "DeepSharp.Verso.Serve.csproj"));

    private static string Property(XDocument project, string name) => project.Descendants(name).Single().Value;

    [Fact]
    public void TheToolPacks_AsDeepSharpServe_AndKeepsStartingOnTheNewerRuntime()
    {
        var project = Project();

        Assert.Equal("Microsoft.NET.Sdk.Web", project.Root!.Attribute("Sdk")!.Value);
        Assert.Equal("true", Property(project, "IsPackable"));
        Assert.Equal("true", Property(project, "PackAsTool"));
        Assert.Equal("deepsharp-serve", Property(project, "ToolCommandName"));
        Assert.Equal("Major", Property(project, "RollForward"));
        Assert.Equal("DeepSharp.Verso.Serve", Property(project, "PackageId"));
    }

    [Fact]
    public void TheToolReferencesTheRuntime_AspNet_AndAClosedListOfPackages()
    {
        // On .NET 8 the stream's writer and the async LINQ that puts the stream together come from their packages; on
        // .NET 10 they are the runtime's own.
        string[] allowed = ["DeepSharp.Verso.Api", "Verso", "Verso.Abstractions", "System.Net.ServerSentEvents", "System.Linq.AsyncEnumerable"];
        var runtime = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();
        var aspNet = Path.GetDirectoryName(typeof(WebApplication).Assembly.Location)!;

        var outside = typeof(NotebookServer).Assembly.GetReferencedAssemblies()
            .Select(reference => reference.Name!)
            .Where(name => !File.Exists(Path.Join(runtime, $"{name}.dll")) && !File.Exists(Path.Join(aspNet, $"{name}.dll")))
            .Except(allowed)
            .ToArray();

        Assert.True(outside.Length == 0, $"The tool references {string.Join(", ", outside)}.");
    }
}
