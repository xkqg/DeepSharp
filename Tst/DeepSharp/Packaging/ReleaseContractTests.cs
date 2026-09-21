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
    private static readonly string Root = RepoRoot();

    private static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Join(dir.FullName, "CHANGELOG.md")))
        {
            dir = dir.Parent;
        }

        Assert.NotNull(dir);
        return dir!.FullName;
    }

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
    public void ThePublishWorkflow_RunsTheTestsOnTheCommitItIsPublishing()
    {
        string workflow = Read(".github", "workflows", "publish.yml");

        Assert.Contains("DeepSharp.Tests.csproj", workflow, StringComparison.Ordinal);
    }

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
