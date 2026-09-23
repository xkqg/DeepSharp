// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace DeepSharp.Tests.Packaging;

/// <summary>
/// A release is carried by a handful of files that have to agree with each other and that nothing compiles:
/// the version in the build, the heading in the changelog, the tag someone types, and the workflows that turn
/// those into a published package. Each agreement is a test here, because the alternative is remembering.
/// </summary>
public class ReleaseContractTests
{
    private static readonly string Root = Repository.Root;

    private static string Read(params string[] parts) => File.ReadAllText(Path.Join([Root, .. parts]));

    /// <summary>The one version number, from the file that hands it to every packable project.</summary>
    private static string DeclaredVersion() =>
        XDocument.Load(Path.Join(Root, "Directory.Build.props"))
            .Descendants("Version").Single().Value.Trim();

    [Fact]
    public void TheVersionInTheBuild_IsTheOneTheChangelogDescribes()
    {
        // These two drift in opposite directions: the build is bumped when work starts and the changelog is
        // written when work ends. A package whose notes describe the release before it is the one thing a
        // reader cannot check for themselves.
        string changelog = Read("CHANGELOG.md");
        var newest = Regex.Match(changelog, @"^## \[(?<version>[^\]]+)\]", RegexOptions.Multiline);

        Assert.True(newest.Success, "CHANGELOG.md has no version heading at all");
        Assert.Equal(DeclaredVersion(), newest.Groups["version"].Value);
    }

    [Fact]
    public void TheVersionIsDeclaredInExactlyOnePlace()
    {
        // A second <Version> in a project file wins over the shared one for that project alone, which ships a
        // package a version behind its siblings without anything going red.
        var strays = Directory.EnumerateFiles(Root, "*.csproj", SearchOption.AllDirectories)
            .Where(file => !file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}",
                                          StringComparison.Ordinal))
            .Where(file => XDocument.Load(file).Descendants("Version").Any())
            .Select(Path.GetFileName)
            .ToArray();

        Assert.True(strays.Length == 0,
            $"These projects declare their own version instead of taking the shared one: {string.Join(", ", strays)}");
    }

    [Fact]
    public void TheReleaseWorkflow_TakesItsNotesFromTheChangelogSectionForThatVersion()
    {
        // Release notes assembled from commit subjects describe the work; the changelog describes the change.
        // Only one of those is written for the person deciding whether to upgrade.
        string workflow = Read(".github", "workflows", "release.yml");

        Assert.Contains("CHANGELOG.md", workflow, StringComparison.Ordinal);
        Assert.Contains("notes-file", workflow, StringComparison.Ordinal);
        Assert.Contains("test -s notes.md", workflow, StringComparison.Ordinal);
    }

    [Fact]
    public void ThePublishWorkflow_CannotReportSuccessWhileItPushedNothing()
    {
        // A loop that pushes nothing exits exactly as happily as one that pushed everything, and an empty
        // pack directory is the quietest way for a release to not happen.
        string workflow = Read(".github", "workflows", "publish.yml");

        Assert.Contains("count=$(ls nupkgs/*.nupkg", workflow, StringComparison.Ordinal);
        Assert.Contains("exit 1", workflow, StringComparison.Ordinal);
    }

    [Fact]
    public void ThePublishWorkflow_RunsEverySuiteOnTheCommitItIsPublishing()
    {
        // The suites are found by their name, never listed: a workflow naming the first suite by hand keeps
        // passing when a second one lands beside it, and the release goes out with a suite nobody ran.
        string workflow = Read(".github", "workflows", "publish.yml");

        Assert.Contains("for suite in Tst/*/*.Tests.csproj", workflow, StringComparison.Ordinal);
        Assert.Contains("dotnet run --project \"$suite\"", workflow, StringComparison.Ordinal);
        Assert.Contains("|| exit 1", workflow, StringComparison.Ordinal);
        Assert.True(TestSuites().Length > 1);
    }

    [Fact]
    public void TheCoverageGate_MeasuresEverySuite_FoundByItsName()
    {
        // A suite the gate never ran leaves its package measured by nothing, while the total still says PASS.
        // So the gate finds the suites the way the workflow does, measures each, and judges them together.
        string gate = Read("tools", "coverage", "run.ps1");

        Assert.Contains("*.Tests.csproj", gate, StringComparison.Ordinal);
        Assert.Contains("dotnet-coverage merge", gate, StringComparison.Ordinal);
        Assert.All(TestSuites(), suite => Assert.DoesNotContain(Path.GetFileName(suite), gate, StringComparison.Ordinal));
    }

    /// <summary>Every suite in the repository, by the name every suite has.</summary>
    private static string[] TestSuites() =>
        [.. Directory.EnumerateFiles(Path.Join(Root, "Tst"), "*.Tests.csproj", SearchOption.AllDirectories)
            .Where(file => !file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}",
                                          StringComparison.Ordinal))
            .Select(file => Path.GetRelativePath(Root, file).Replace(Path.DirectorySeparatorChar, '/'))];

    [Fact]
    public void TheCoverageSettings_CanBeReadAtAllAndNameEveryLibrary()
    {
        // A settings file the collector cannot parse is ignored without a word, and the gate then measures
        // something else entirely and still says PASS. Measured: one double hyphen inside an XML comment
        // was enough, and the number went from 204 lines of library to 841 of library-and-tests.
        var settings = XDocument.Load(Path.Join(Root, "tools", "coverage", "coverage.runsettings"));

        var included = settings.Descendants("Include").Descendants("ModulePath")
            .Select(module => module.Value).ToArray();

        var libraries = Directory.EnumerateFiles(Path.Join(Root, "Src"), "*.csproj", SearchOption.AllDirectories)
            .Where(file => !file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}",
                                          StringComparison.Ordinal))
            .Select(file => $"{Path.GetFileNameWithoutExtension(file)}.dll");

        foreach (var library in libraries)
        {
            Assert.Contains(included, pattern => Regex.IsMatch(library, pattern));
        }
    }

    [Fact]
    public void ADependencyTwoPackagesShare_IsTakenAtOneVersion()
    {
        // Two packages naming the same dependency at two versions ship an application a conflict to
        // resolve, and which version wins then depends on which package the application happened to
        // reference first. Measured before this test: one pinned 1.17.1 and the other floated on 1.17.*.
        var versions = Directory.EnumerateFiles(Path.Join(Root, "Src"), "*.csproj", SearchOption.AllDirectories)
            .Where(file => !file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}",
                                          StringComparison.Ordinal))
            .SelectMany(file => XDocument.Load(file).Descendants("PackageReference"))
            .Where(reference => reference.Attribute("Version") is not null)
            .GroupBy(reference => reference.Attribute("Include")!.Value, StringComparer.OrdinalIgnoreCase)
            .Select(group => new
            {
                Package = group.Key,
                Versions = group.Select(reference => reference.Attribute("Version")!.Value).Distinct().ToArray(),
            })
            .Where(each => each.Versions.Length > 1)
            .Select(each => $"{each.Package} at {string.Join(" and ", each.Versions)}")
            .ToArray();

        Assert.True(versions.Length == 0, $"Taken at more than one version: {string.Join("; ", versions)}");
    }

    [Theory]
    [InlineData("ci.yml")]
    [InlineData("publish.yml")]
    public void EveryProjectThatBecomesAPackage_IsOneTheWorkflowPacks(string file)
    {
        // A workflow that names one project by hand keeps working perfectly when a second package is added,
        // and ships one of the two. Nothing goes red: the pack succeeds, the push succeeds, and the package
        // the release notes describe is simply not on the feed.
        string workflow = Read(".github", "workflows", file);
        var packsEverything = workflow.Contains("dotnet pack DeepSharp.slnx", StringComparison.Ordinal);

        var missing = PackableProjects()
            .Where(project => !packsEverything && !workflow.Contains(project, StringComparison.Ordinal))
            .ToArray();

        Assert.True(missing.Length == 0,
            $"{file} does not pack: {string.Join(", ", missing)}");
    }

    /// <summary>The projects that produce a package, named the way a workflow would name them.</summary>
    private static IEnumerable<string> PackableProjects() =>
        Directory.EnumerateFiles(Path.Join(Root, "Src"), "*.csproj", SearchOption.AllDirectories)
            .Where(file => !file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}",
                                          StringComparison.Ordinal))
            .Where(file => XDocument.Load(file).Descendants("IsPackable")
                               .All(packable => !string.Equals(packable.Value.Trim(), "false",
                                                               StringComparison.OrdinalIgnoreCase)))
            .Select(file => Path.GetRelativePath(Root, file).Replace(Path.DirectorySeparatorChar, '/'));

    [Fact]
    public void TheReadmeAndTheChangelog_AgreeOnWhichVersionThisIs()
    {
        // The README is the package's front page on NuGet. A number there that is behind the package it is
        // attached to is read as the truth, because nobody opens the changelog to check the README.
        Assert.Contains(DeclaredVersion(), Read("README.md"), StringComparison.Ordinal);
    }

    [Fact]
    public void ThePublishWorkflow_ProvesWhoItIsRatherThanCarryingAKey()
    {
        // nuget.org trusts this repository and this workflow file by name, and hands the run a key that lives
        // for an hour. Swapping back to a stored key would still publish, so nothing would go red — it would
        // just quietly reintroduce a long-lived credential that can leak and has to be rotated, and it would
        // bypass the policy that says only this workflow may publish these packages.
        string workflow = Read(".github", "workflows", "publish.yml");

        Assert.Contains("id-token: write", workflow, StringComparison.Ordinal);
        Assert.Contains("NuGet/login@", workflow, StringComparison.Ordinal);
        Assert.Contains("steps.login.outputs.NUGET_API_KEY", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("secrets.NUGET_API_KEY", workflow, StringComparison.Ordinal);
    }
}
