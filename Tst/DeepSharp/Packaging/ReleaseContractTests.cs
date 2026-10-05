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
    public void TheReleaseWorkflow_StartsThePublishOfTheReleaseItMakes()
    {
        // GitHub starts no workflow from an event the bot's own token raised, but for a dispatch, and a release this
        // workflow makes carries the bot as its author: 0.1.0, 0.2.0 and 0.3.0 each reached NuGet only once somebody
        // started the publish by hand. So a tag publishes because the release it makes starts the publish itself.
        string workflow = Read(".github", "workflows", "release.yml");
        var made = workflow.IndexOf("gh release create", StringComparison.Ordinal);
        var dispatched = workflow.IndexOf("gh workflow run publish.yml", StringComparison.Ordinal);

        Assert.Contains("actions: write", workflow, StringComparison.Ordinal);
        Assert.True(made >= 0 && dispatched > made, "release.yml does not start publish.yml once it has made the release");
    }

    [Fact]
    public void ThePublishWorkflow_RefusesATagThatIsNotTheVersionItPacks()
    {
        // A tag publishes, and a package on NuGet cannot be taken back: a tag naming another version than the build's
        // would put that build's packages on NuGet under a release that says otherwise.
        string workflow = Read(".github", "workflows", "publish.yml");
        var held = workflow.IndexOf("test \"${tag#v}\" = \"$declared\"", StringComparison.Ordinal);

        Assert.True(held >= 0, "publish.yml does not hold the tag to the version the build declares");
        Assert.True(held < workflow.IndexOf("dotnet pack DeepSharp.slnx", StringComparison.Ordinal), "publish.yml packs before it holds the tag to the version");
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
        // So the gate finds the suites the way the workflow does, measures each, and judges them together — pooling
        // their reports itself, since the coverage tool's merge of the same reports came out differently from one run
        // to the next, once losing most of a class's branches.
        string gate = Read("tools", "coverage", "run.ps1");

        Assert.Contains("*.Tests.csproj", gate, StringComparison.Ordinal);
        Assert.Contains("foreach ($part in $parts)", gate, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet-coverage merge", gate, StringComparison.Ordinal);
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
    public void EveryProjectOnTheWebSdk_SaysItPacks()
    {
        // The web SDK packs nothing unless a project says it may, and says so only in a warning: the pack succeeds and
        // writes no package, and the release goes out without it (measured on the server's own project).
        var silent = Directory.EnumerateFiles(Path.Join(Root, "Src"), "*.csproj", SearchOption.AllDirectories)
            .Where(file => !file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .Select(file => XDocument.Load(file))
            .Where(project => project.Root!.Attribute("Sdk")?.Value == "Microsoft.NET.Sdk.Web")
            .Where(project => project.Descendants("IsPackable").All(packable => !string.Equals(packable.Value.Trim(), "true", StringComparison.OrdinalIgnoreCase)))
            .Select(project => project.Descendants("PackageId").Single().Value)
            .ToArray();

        Assert.True(silent.Length == 0, $"On the web SDK without saying they pack: {string.Join(", ", silent)}");
    }

    [Fact]
    public void TheSolution_HoldsEveryProjectAndEverySuite()
    {
        // The workflows pack and build the solution, so a project missing from it is neither packed nor tested, and
        // nothing says so.
        var solution = Read("DeepSharp.slnx");
        var missing = Directory.EnumerateFiles(Path.Join(Root, "Src"), "*.csproj", SearchOption.AllDirectories)
            .Concat(Directory.EnumerateFiles(Path.Join(Root, "Tst"), "*.Tests.csproj", SearchOption.AllDirectories))
            .Where(file => !file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .Select(file => Path.GetRelativePath(Root, file).Replace(Path.DirectorySeparatorChar, '/'))
            .Where(project => !solution.Contains($"Path=\"{project}\"", StringComparison.Ordinal))
            .ToArray();

        Assert.True(missing.Length == 0, $"Not in the solution: {string.Join(", ", missing)}");
    }

    [Fact]
    public void ThePublishWorkflow_PushesAPackageForEveryProjectInSrc_OrNone()
    {
        // More than none is not enough: a pack that quietly skipped one project still pushes the others, and the
        // release notes describe a package that is not on the feed.
        string workflow = Read(".github", "workflows", "publish.yml");

        Assert.Contains("expected=$(ls Src/*/*.csproj | wc -l)", workflow, StringComparison.Ordinal);
        Assert.Contains("test \"$count\" -eq \"$expected\"", workflow, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("ci.yml")]
    [InlineData("publish.yml")]
    public void TheServer_IsStartedFromThePackageJustMade_BeforeAnyPackageLeavesTheRun(string file)
    {
        // What a person runs is the package, and a package can lack what the build had while every suite, which runs the
        // build, stays green: a file left out, or the build for one runtime finding a dependency only on the other. So the
        // package just made is installed and started, after the pack and before anything is pushed — and from those
        // packages alone, since with the feed as a second source a version already published would install as well.
        string workflow = Read(".github", "workflows", file);
        var check = workflow.IndexOf("bash tools/serve/check.sh nupkgs", StringComparison.Ordinal);
        var push = workflow.IndexOf("dotnet nuget push", StringComparison.Ordinal);

        Assert.True(check > workflow.IndexOf("dotnet pack DeepSharp.slnx", StringComparison.Ordinal), $"{file} does not start the server from the package it made");
        Assert.True(push < 0 || check < push, $"{file} pushes the packages before it starts the server from one");
        Assert.Contains("dotnet tool install DeepSharp.Verso.Serve --tool-path \"$work/tool\" --source \"$packages\"", Read("tools", "serve", "check.sh"), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("ci.yml")]
    [InlineData("publish.yml")]
    public void TheNotebook_IsInstalledByVersoFromThePackageJustMade_BeforeAnyPackageLeavesTheRun(string file)
    {
        // Verso installs the notebook package into a folder of its own and loads it apart from everything else, while every
        // suite runs the build, where a dependency the package never declared is found all the same. So the package just
        // made is installed by Verso's own installer, loaded by its own loader and run by its own kernel, in a host that
        // holds nothing of DeepSharp — on .NET 8, and on the newest runtime, where Verso's VS Code extension runs that same
        // build — after the pack and before anything is pushed.
        string workflow = Read(".github", "workflows", file);
        var check = workflow.IndexOf("bash tools/verso/check.sh nupkgs", StringComparison.Ordinal);
        var push = workflow.IndexOf("dotnet nuget push", StringComparison.Ordinal);

        Assert.True(check > workflow.IndexOf("dotnet pack DeepSharp.slnx", StringComparison.Ordinal), $"{file} does not install the notebook from the package it made");
        Assert.True(push < 0 || check < push, $"{file} pushes the packages before Verso installs the notebook from one");
        Assert.Contains("dotnet exec --roll-forward LatestMajor", Read("tools", "verso", "check.sh"), StringComparison.Ordinal);
        Assert.Contains("MarketplaceLoader.InstallLocalFileAsync", Read("tools", "verso", "VersoHost", "Program.cs"), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("serve")]
    [InlineData("verso")]
    [InlineData("torch")]
    [InlineData("ml")]
    public void ACheckOfThePackagesJustMade_ExtractsThemIntoAFolderOfItsOwn_NeverIntoTheMachinesPackageCache(string check)
    {
        // A version is extracted into the machine's package cache once, and never again: a check run on packages packed
        // earlier in the same work leaves that build there, and a later check of the same version would then start the build
        // it found rather than the one just packed — green over code that is no longer there. So each check hands NuGet a
        // folder of its own for the packages it extracts, inside the folder it removes when it ends.
        var script = Read("tools", check, "check.sh");

        Assert.Matches(@"NUGET_PACKAGES=""(\$\(native "")?\$(work|run)/", script);
    }

    [Theory]
    [InlineData("ci.yml")]
    [InlineData("publish.yml")]
    public void TheTorchBackend_IsCheckedFromThePackageJustMade_BeforeAnyPackageLeavesTheRun(string file)
    {
        // An application referencing DeepSharp.Backends.TorchSharp is checked against the package just made, not the
        // build every suite ran: refused by name where it brings no libtorch, and a real step of the networks sample's
        // Titanic pipeline where it brings the processor's — after the pack and before anything is pushed.
        string workflow = Read(".github", "workflows", file);
        var check = workflow.IndexOf("bash tools/torch/check.sh nupkgs", StringComparison.Ordinal);
        var push = workflow.IndexOf("dotnet nuget push", StringComparison.Ordinal);

        Assert.True(check > workflow.IndexOf("dotnet pack DeepSharp.slnx", StringComparison.Ordinal), $"{file} does not check the engine package from the package it made");
        Assert.True(push < 0 || check < push, $"{file} pushes the packages before checking the engine package from one");
        Assert.Contains("PackageReference Include=\"DeepSharp.Backends.TorchSharp\"", Read("tools", "torch", "TorchCheckHost", "TorchCheckHost.csproj"), StringComparison.Ordinal);
    }

    [Fact]
    public void TheTorchSharpBackendSuite_RunsOnUbuntuAndWindows_BothFrameworks()
    {
        // The suite's own csproj brings the processor's libtorch for whichever platform it is built on, so `ci.yml`'s
        // one ubuntu job already reaches it through the coverage gate; a second, Windows-only job reaches the native
        // interop a Linux run never touches, on both frameworks a package ships, without repeating the whole gate there.
        string workflow = Read(".github", "workflows", "ci.yml");

        Assert.Contains("windows-latest", workflow, StringComparison.Ordinal);
        Assert.Contains("Tst/DeepSharp.Backends.TorchSharp/DeepSharp.Backends.TorchSharp.Tests.csproj", workflow, StringComparison.Ordinal);
        Assert.Contains("net8.0", workflow, StringComparison.Ordinal);
        Assert.Contains("net10.0", workflow, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryPackageThatExistedAtTheLastRelease_ValidatesAgainstItsBaseline_AndANewOneDoesNot()
    {
        // A package nuget.org already carries is checked against that build's own public surface, so a change nobody
        // meant to make breaks the build instead of somebody's upgrade; a package new this release has nothing yet to
        // compare against, and asking for a baseline that was never published would refuse the build for a package
        // that never shipped one. Every package shipped before the release this build is for is named here, and a
        // package that was not fails this test until somebody says which of the two it is.
        string[] publishedBefore =
        [
            "DeepSharp", "DeepSharp.Backends.TorchSharp", "DeepSharp.Charts", "DeepSharp.Import.Keras",
            "DeepSharp.Import.Onnx", "DeepSharp.Import.PyTorch", "DeepSharp.Learners.ML", "DeepSharp.Learners.MLNet",
            "DeepSharp.Learners.Networks", "DeepSharp.Pipelines", "DeepSharp.Pipelines.DataFrame",
            "DeepSharp.Pipelines.Excel", "DeepSharp.Pipelines.Indicators", "DeepSharp.Pipelines.Json",
            "DeepSharp.Pipelines.Parquet", "DeepSharp.Verso.Api", "DeepSharp.Verso.Notebooks", "DeepSharp.Verso.Serve",
        ];
        var props = Read("Directory.Build.props");
        var listed = Regex.Match(props, @"<PublishedBefore>(?<list>[^<]*)</PublishedBefore>").Groups["list"].Value
            .Split(';', StringSplitOptions.RemoveEmptyEntries);
        var baseline = Regex.Match(props, @"<PackageValidationBaselineVersion>(?<version>[^<]*)</PackageValidationBaselineVersion>")
            .Groups["version"].Value;
        var packable = PackableProjects().Select(project => Path.GetFileNameWithoutExtension(project)).ToArray();

        Assert.Contains("<EnablePackageValidation>true</EnablePackageValidation>", props, StringComparison.Ordinal);
        Assert.Equal(publishedBefore.Order(StringComparer.Ordinal), listed.Order(StringComparer.Ordinal));

        // Every package is packed, and every one of them shipped before this release: none is new, so none goes without a
        // baseline. A package that is new next release is added to the packable ones and left out of the list above, and
        // this test says so until it ships.
        Assert.Equal(packable.Order(StringComparer.Ordinal), publishedBefore.Order(StringComparer.Ordinal));

        // The baseline is the release nuget.org carries, which is the one before this build's. Left behind it, a surface
        // the last release added could be removed with nothing to say so.
        Assert.Equal("0.7.0", baseline);
        Assert.True(Version.Parse(baseline) < Version.Parse(DeclaredVersion()), "The baseline is the release before this one, never this one.");
    }

    [Fact]
    public void TheVersoTheNotebookIsInstalledIn_IsTheVersoItsHostsAreBuiltOn()
    {
        // A Verso that installs or loads a package otherwise is checked the day DeepSharp's own hosts move to it, not after:
        // the host the notebook is installed in takes the engine the application host and the notebook's suite take.
        static string Verso(params string[] project) => XDocument.Load(Path.Join([Root, .. project])).Descendants("PackageReference")
            .Single(reference => reference.Attribute("Include")!.Value == "Verso").Attribute("Version")!.Value;

        var installedIn = Verso("tools", "verso", "VersoHost", "VersoHost.csproj");

        Assert.Equal(Verso("Src", "DeepSharp.Verso.Api", "DeepSharp.Verso.Api.csproj"), installedIn);
        Assert.Equal(Verso("Tst", "DeepSharp.Verso.Notebooks", "DeepSharp.Verso.Notebooks.Tests.csproj"), installedIn);
    }

    [Fact]
    public void EveryScriptBashRuns_IsCheckedOutWithTheLineEndingsBashReads()
    {
        // Git on Windows checks text out with a carriage return ending each line unless it is told otherwise. The bash that
        // comes with Git forgives that (measured); bash on Linux reads the return as part of each command, so the script the
        // workflows run would fail in WSL, or in a container handed a checkout made on Windows.
        var scripts = Directory.EnumerateFiles(Path.Join(Root, "tools"), "*.sh", SearchOption.AllDirectories).ToArray();
        var returns = scripts
            .Where(script => File.ReadAllText(script).Contains('\r'))
            .Select(script => Path.GetRelativePath(Root, script))
            .ToArray();

        Assert.NotEmpty(scripts);
        Assert.True(returns.Length == 0, $"Checked out with carriage returns: {string.Join(", ", returns)}");
    }

    [Fact]
    public void EveryPackage_ShipsABuildForEachRuntimeAVersoSurfaceRuns()
    {
        // Verso's browser host stays on .NET 8 for as long as that runtime is installed, its VS Code host takes the
        // newest one, and a notebook cell may reference any of these packages. A package built for one runtime alone
        // fails to load on the other with an error that names neither the package nor the cure. Measured: .NET 8
        // could not load a build for .NET 10 alone, "System.Runtime, Version=10.0.0.0" not found.
        var single = PackableProjects()
            .Where(project => !FrameworksOf(project).SequenceEqual(["net8.0", "net10.0"]))
            .ToArray();

        Assert.True(single.Length == 0, $"Built for one runtime only: {string.Join(", ", single)}");
    }

    /// <summary>The frameworks a project is built for, in the order it names them.</summary>
    private static IReadOnlyList<string> FrameworksOf(string project)
    {
        var document = XDocument.Load(Path.Join(Root, project));
        var named = document.Descendants("TargetFrameworks").Select(frameworks => frameworks.Value)
            .Concat(document.Descendants("TargetFramework").Select(framework => framework.Value))
            .First();

        return [.. named.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)];
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
