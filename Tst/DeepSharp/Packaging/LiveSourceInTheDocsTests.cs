// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using System.Xml.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace DeepSharp.Tests.Packaging;

/// <summary>
/// What the documents say of the live source, and what no suite may do with what they say. A program in a document that a
/// suite runs, and that names the live source, would send a request to Binance on every build of every machine that builds
/// this repository — a guest of a budget that is shared with whatever else runs on the same address — so the programs a suite
/// runs are held apart from those it only compiles, and a page that lands is compiled and never run. The rest is the
/// documents agreeing with the code: the count of packages, the layout, the pages that lead to the new one, and no page that
/// still says the source is a design.
/// </summary>
public sealed partial class LiveSourceInTheDocsTests
{
    private static readonly string[] LiveNames = ["BinanceCandles", "ReadBinanceAsync", "LandAsync"];

    private static readonly string[] Numbers =
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
        "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven", "twenty-eight", "twenty-nine", "thirty",
    ];

    [GeneratedRegex("```csharp\\n(?<code>.*?)```", RegexOptions.Singleline)]
    private static partial Regex CSharpBlock();

    private static string Read(params string[] path) => File.ReadAllText(Path.Join([Repository.Root, .. path])).ReplaceLineEndings("\n");

    private static string WikiText(string page) => File.ReadAllText(Path.Join(Wiki.Folder, page)).ReplaceLineEndings("\n");

    private readonly record struct Document(string Name, string Text);

    private static bool NamesTheLiveSource(string code) => LiveNames.Any(name => code.Contains(name, StringComparison.Ordinal));

    private static string[] BlocksOf(string text) => [.. CSharpBlock().Matches(text).Select(match => match.Groups["code"].Value)];

    private static string[] PackageIds() => [.. ReleaseContractTests.PackableProjects()
        .Select(project => XDocument.Load(Path.Join(Repository.Root, project)).Descendants("PackageId").SingleOrDefault()?.Value ?? Path.GetFileNameWithoutExtension(project))
        .Order(StringComparer.Ordinal)];

    [Fact]
    public void TheReadme_SaysHowManyPackagesThePackMakes()
    {
        var count = PackageIds().Length;

        Assert.Contains($"makes all {Numbers[count]}.", Read("README.md"), StringComparison.Ordinal);
    }

    [Fact]
    public void TheReadmesTable_HasARowForEveryPackage()
    {
        var readme = Read("README.md");
        var missing = PackageIds().Where(id => !readme.Contains($"| `{id}` |", StringComparison.Ordinal)).ToArray();

        Assert.True(missing.Length == 0, $"The README's table of packages has no row for: {string.Join(", ", missing)}");
    }

    [Fact]
    public void TheLayoutOfTheRepositorysArchitecture_NamesEveryProjectOfSrc() => AssertLayoutNamesEveryProject("ARCHITECTURE.md", Read("ARCHITECTURE.md"));

    [Fact]
    public void TheLayoutOfTheWikisArchitecturePage_NamesEveryProjectOfSrc() => AssertLayoutNamesEveryProject("the wiki's Architecture page", WikiText("Architecture.md"));

    private static void AssertLayoutNamesEveryProject(string document, string text)
    {
        var missing = Directory.GetDirectories(Path.Join(Repository.Root, "Src"))
            .Select(folder => Path.GetFileName(folder))
            .Where(name => !text.Contains($"Src/{name}/", StringComparison.Ordinal))
            .ToArray();

        Assert.True(missing.Length == 0, $"{document} lays out the source without: {string.Join(", ", missing)}");
    }

    [Fact]
    public void TheRunBlockOfTheProjectsInstructions_ListsEveryCheckOfThePackagesJustMade()
    {
        var instructions = Read("CLAUDE.md");
        var checks = Directory.GetFiles(Path.Join(Repository.Root, "tools"), "check.sh", SearchOption.AllDirectories)
            .Select(path => Path.GetFileName(Path.GetDirectoryName(path)!))
            .ToArray();

        Assert.True(checks.Length >= 5, $"Only {checks.Length} checks were found under tools: {string.Join(", ", checks)}");
        Assert.All(checks, check => Assert.Contains($"bash tools/{check}/check.sh nupkgs", instructions, StringComparison.Ordinal));
    }

    [Fact]
    public void TheReadmesLiveExample_Compiles_AndIsNeverRun()
    {
        var block = Assert.Single(BlocksOf(Read("README.md")), NamesTheLiveSource);

        // That literal is how the one example of the README that a suite runs is found, and a second would be run too.
        Assert.DoesNotContain("Pdd.Create()", block, StringComparison.Ordinal);

        var lines = block.Split('\n');
        var source = $$"""
            using System;
            using System.Threading.Tasks;
            {{string.Join('\n', lines.Where(line => line.StartsWith("using ", StringComparison.Ordinal)))}}

            public static class Example
            {
                public static async Task Land()
                {
            {{string.Join('\n', lines.Where(line => !line.StartsWith("using ", StringComparison.Ordinal)))}}
                }
            }
            """;
        var references = ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!)
            .Split(Path.PathSeparator)
            .Select(path => MetadataReference.CreateFromFile(path));
        var compilation = CSharpCompilation.Create(
            "ReadmeLiveExample",
            [CSharpSyntaxTree.ParseText(source, cancellationToken: TestContext.Current.CancellationToken)],
            references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: NullableContextOptions.Enable));

        using var image = new MemoryStream();
        var emitted = compilation.Emit(image, cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(emitted.Success, string.Join('\n', emitted.Diagnostics.Where(each => each.Severity == DiagnosticSeverity.Error)));
    }

    [Fact]
    public void NoProgramASuiteRuns_NamesTheLiveSource()
    {
        var readmeRun = BlocksOf(Read("README.md")).Single(code => code.Contains("Pdd.Create()", StringComparison.Ordinal));
        var runs = Wiki.Runnable().SelectMany(run => run.Program.Blocks.Select(block => block.Code)).Append(readmeRun).ToArray();

        Assert.True(runs.Length > 10, $"Only {runs.Length} programs were found that a suite runs: the selection is not what the suites run.");
        Assert.DoesNotContain(runs, NamesTheLiveSource);
    }

    [Fact]
    public void NoCellOfTheWiki_NamesTheLiveSource_ForTheNotebooksSuiteRunsTheCells()
    {
        var cells = Wiki.Pages().SelectMany(page => page.Cells).ToArray();

        Assert.NotEmpty(cells);
        Assert.DoesNotContain(cells, cell => NamesTheLiveSource(cell.Code));
    }

    [Fact]
    public void TheRuleThatFindsAProgramNamingTheLiveSource_FindsOneThatDoes_AndPassesOneThatDoesNot()
    {
        // The control for the guards above: a program that lands is found by the rule, one that reads a file is not.
        Assert.True(NamesTheLiveSource("var landed = await new BinanceCandles(\"BTCEUR\", \"1d\", from, to).LandAsync(folder);"));
        Assert.True(NamesTheLiveSource("var rows = await Pdd.Create().ReadBinanceAsync(candles);"));
        Assert.False(NamesTheLiveSource("var rows = Pdd.Create().ReadCsv(\"btceur-1d.csv\");"));
    }

    [Fact]
    public void TheWiki_HasAPageForLiveSources_ThatTheSidebarTheHomePageAndTheRoadmapLeadTo()
    {
        Assert.True(File.Exists(Path.Join(Wiki.Folder, "Live-sources.md")), "The wiki has no page for live sources.");
        Assert.Contains("[[Live sources]]", WikiText("_Sidebar.md"), StringComparison.Ordinal);
        Assert.Contains("| [[Live sources]] |", WikiText("Home.md"), StringComparison.Ordinal);
        Assert.Contains("[[Live sources]]", WikiText("Pipeline.md"), StringComparison.Ordinal);
        Assert.Contains("wiki/Live-sources", Read("README.md"), StringComparison.Ordinal);
    }

    [Fact]
    public void TheRoadmap_SaysLiveSourcesAreDone_AndHasNoRowLeftThatSaysLater()
    {
        var roadmap = WikiText("Roadmap.md");

        Assert.Contains("| **Done** | [[Live sources]] |", roadmap, StringComparison.Ordinal);
        Assert.DoesNotContain("**Later**", roadmap, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("None is built yet")]
    [InlineData("design, not yet built")]
    [InlineData("each will be a package of its own")]
    [InlineData("what comes next: live sources")]
    public void NoDocument_SaysALiveSourceIsAnIntentionStill(string stale)
    {
        var documents = new[] { "README.md", "ARCHITECTURE.md", "CONTRIBUTING.md", "CLAUDE.md" }.Select(file => new Document(file, Read(file)))
            .Concat(Wiki.Pages().Select(page => new Document(page.Name, WikiText(page.Name))));

        Assert.All(documents, document => Assert.DoesNotContain(stale, document.Text, StringComparison.Ordinal));
    }

    [Fact]
    public void TheArchitecture_NamesTheLandingThatIsBuilt()
    {
        var architecture = Read("ARCHITECTURE.md");

        Assert.Contains("DeepSharp.Pipelines.Binance", architecture, StringComparison.Ordinal);
        Assert.Contains("ReadBinanceAsync", architecture, StringComparison.Ordinal);
        Assert.Contains("BinanceCandles", architecture, StringComparison.Ordinal);
    }
}
