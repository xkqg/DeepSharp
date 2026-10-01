// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A C# cell as a person writes it — naming DeepSharp's packages by their NuGet ids — run against the packages as this
/// repository built them.
/// </summary>
/// <remarks>
/// A notebook resolves <c>#r "nuget: Id"</c> by fetching the package, which the release puts there; before it, and in a
/// test, the same cell is handed the assembly the repository built for that package, for the runtime and the
/// configuration this suite runs on. Nothing else in the cell changes.
/// </remarks>
internal static partial class PackageReferenceExtensions
{
    [GeneratedRegex("^#r \"nuget: (?<id>DeepSharp[A-Za-z.]*)\"$", RegexOptions.Multiline)]
    private static partial Regex NuGetReference();

    extension(string cell)
    {
        /// <summary>The cell, each DeepSharp package it names by its NuGet id referenced as the repository built it.</summary>
        /// <returns>The cell's text, every other line as it was.</returns>
        internal string WithPackagesAsBuilt() =>
            NuGetReference().Replace(cell.ReplaceLineEndings("\n"), reference => $"#r \"{Built(reference.Groups["id"].Value)}\"");
    }

    // A package's assembly as the repository built it: its project's output for this suite's configuration and runtime.
    private static string Built(string id)
    {
        var framework = new DirectoryInfo(Path.TrimEndingDirectorySeparator(AppContext.BaseDirectory));
        var built = Path.Join(Repository.Root, "Src", id, "bin", framework.Parent!.Name, framework.Name, $"{id}.dll");

        Assert.True(File.Exists(built), $"The cell names the package {id}, and the repository has not built it: {built}.");

        return built;
    }
}
