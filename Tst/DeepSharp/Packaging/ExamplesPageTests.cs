// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace DeepSharp.Tests.Packaging;

/// <summary>
/// The changelog says what each release brought; the wiki's examples page shows how to call it — a section to each release,
/// the newest first, what is new and what was improved, each in code written against the library as it is now. The code is
/// compiled with every other block of the wiki; this holds the page's shape: the release this build is comes first and shows
/// something, every section names a release the changelog has, and the start page's row for the release leads to it.
/// </summary>
public sealed partial class ExamplesPageTests
{
    [GeneratedRegex("^## (?<version>\\d+\\.\\d+\\.\\d+)$", RegexOptions.Multiline)]
    private static partial Regex Release();

    private static string Page => File.ReadAllText(Path.Join(Wiki.Folder, "Examples.md")).ReplaceLineEndings("\n");

    private static string Building =>
        XDocument.Load(Path.Join(Repository.Root, "Directory.Build.props")).Descendants("Version").Single().Value;

    [Fact]
    public void TheReleaseThisBuildIs_ComesFirst_AndShowsWhatItBroughtInCode()
    {
        var sections = Release().Matches(Page).ToArray();
        var first = sections[0];
        var end = sections.Length > 1 ? sections[1].Index : Page.Length;

        Assert.Equal(Building, first.Groups["version"].Value);
        Assert.NotEmpty(Wiki.Read("Examples.md", Page[first.Index..end]).Programs);
    }

    [Fact]
    public void EverySection_IsAReleaseTheChangelogHas_NewestFirst_SaidAsNewOrImproved()
    {
        var changelog = File.ReadAllText(Path.Join(Repository.Root, "CHANGELOG.md"));
        var sections = Release().Matches(Page).ToArray();
        Version[] versions = [.. sections.Select(section => Version.Parse(section.Groups["version"].Value))];

        Assert.Equal(versions.OrderDescending(), versions);
        Assert.All(versions, version => Assert.Contains($"## [{version}]", changelog, StringComparison.Ordinal));

        for (var at = 0; at < sections.Length; at++)
        {
            var end = at + 1 < sections.Length ? sections[at + 1].Index : Page.Length;
            var section = Page[sections[at].Index..end];

            Assert.True(
                section.Contains("\n### New\n", StringComparison.Ordinal) || section.Contains("\n### Improved\n", StringComparison.Ordinal),
                $"The examples of {versions[at]} say neither what is new nor what was improved.");
        }
    }

    [Fact]
    public void TheStartPagesRowForTheRelease_LeadsToItsExamples()
    {
        var home = File.ReadAllText(Path.Join(Wiki.Folder, "Home.md")).ReplaceLineEndings("\n");
        var minor = string.Join('.', Building.Split('.').Take(2));
        var row = home.Split('\n').Single(line => line.StartsWith($"| {minor} |", StringComparison.Ordinal));

        Assert.Contains($"(Examples#{Building.Replace(".", string.Empty, StringComparison.Ordinal)})", row, StringComparison.Ordinal);
    }
}
