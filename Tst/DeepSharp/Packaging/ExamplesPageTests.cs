// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Xml.Linq;

namespace DeepSharp.Tests.Packaging;

/// <summary>
/// The changelog says what each release brought; the wiki's examples page shows how to call it — a section to each release,
/// the newest first, what is new and what was improved, each in code written against the library as it is now. The code is
/// compiled with every other block of the wiki; this holds the page's shape: the release this build is comes first and shows
/// something, every section names a release the changelog has, and the start page's row for the release leads to it.
/// </summary>
public sealed class ExamplesPageTests
{
    private static string Page => File.ReadAllText(Path.Join(Wiki.Folder, "Examples.md")).ReplaceLineEndings("\n");

    private static string Building =>
        XDocument.Load(Path.Join(Repository.Root, "Directory.Build.props")).Descendants("Version").Single().Value;

    [Fact]
    public void TheReleaseThisBuildIs_ComesFirst_AndShowsWhatItBroughtInCode()
    {
        var first = Sections()[0];

        Assert.Equal(Version.Parse(Building), first.Version);
        Assert.NotEmpty(Wiki.Read("Examples.md", first.Text).Programs);
    }

    [Fact]
    public void EverySection_IsAReleaseTheChangelogHas_NewestFirst_SaidAsNewOrImproved()
    {
        var changelog = File.ReadAllText(Path.Join(Repository.Root, "CHANGELOG.md"));
        var sections = Sections();
        Version[] versions = [.. sections.Select(section => section.Version)];

        Assert.Equal(versions.OrderDescending(), versions);
        Assert.All(versions, version => Assert.Contains($"## [{version}]", changelog, StringComparison.Ordinal));

        foreach (var section in sections)
        {
            Assert.True(
                section.Text.Contains("\n### New\n", StringComparison.Ordinal) || section.Text.Contains("\n### Improved\n", StringComparison.Ordinal),
                $"The examples of {section.Version} say neither what is new nor what was improved.");
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

    // The page's releases in the order they stand: every section whose heading is a version, up to the next heading.
    private static Section[] Sections() =>
    [
        .. Page.Split("\n## ").Skip(1)
            .Select(chunk => new Section(chunk[..chunk.IndexOf('\n', StringComparison.Ordinal)], "## " + chunk))
            .Where(section => section.Heading.Split('.').Length == 3 && Version.TryParse(section.Heading, out _)),
    ];

    /// <summary>One section of the page.</summary>
    /// <param name="Heading">What its heading says.</param>
    /// <param name="Text">The section, from its heading to the next.</param>
    private readonly record struct Section(string Heading, string Text)
    {
        /// <summary>The release the section is of.</summary>
        public Version Version => Version.Parse(Heading);
    }
}
