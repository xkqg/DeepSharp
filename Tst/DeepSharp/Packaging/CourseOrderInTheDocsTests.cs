// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using DeepSharp.Pipelines;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace DeepSharp.Tests.Packaging;

/// <summary>
/// A course teaches one order, so everything that walks through it is held to that order: the chains the README, the wiki
/// and the samples write, the ten pages of the tutorial, and the page that describes the answer. An order held by hand
/// drifts — a tutorial written first and a course written after it taught two orders for a while, and a reader who learned
/// the one met a refusal in the other — so the stages are held to the named courses, and every place that shows an order is
/// measured against the stages.
/// </summary>
public sealed partial class CourseOrderInTheDocsTests
{
    private static readonly Stage[] Stages =
    [
        new("the source", ["read.csv"], ["ReadCsv", "ReadParquet", "ReadExcel", "ReadJson", "ReadBinanceAsync"]),
        new("the columns", ["declare"], ["Declare"]),
        new("the order", ["order.by"], ["OrderBy", "Shuffle"]),
        new("the gaps", ["settle.gaps"], ["SettleGaps"]),
        new("the features", ["feature.add"], ["AddFeature", "Cyclical", "TimeParts", "TimePartsAsNumbers", "Reshape", "Add", "DropWarmUp"]),
        new("the scale from bounds", ["scale.given"], ["ScaleGiven"]),
        new("the split", ["split.stratified", "split.byTime"], ["SplitStratified", "SplitByTime", "SplitAtRandom"]),
        new("the answer", ["target", "target.ahead"], ["Target", "Ahead", "Distribution", "Labels"]),
        new("what is dropped", ["drop.columns"], ["Drop"]),
        new("what is learned", ["fill.missing"], ["FillMissing", "FillNaN", "EncodeCategories", "Encode"]),
        new("the scale", ["normalise"], ["Normalise"]),
        new("the report", ["evidence.report"], ["Report"]),
        new("the network", ["learn.network"], ["WithTorch", "WithTensorflow", "WithML"]),
    ];

    // The stage each page of the tutorial is about; the last page, on saving, is about none.
    private static readonly Dictionary<string, string> StageOfPage = new(StringComparer.Ordinal)
    {
        ["Reading a file"] = "the source",
        ["Declaring the columns"] = "the columns",
        ["Settling the gaps"] = "the gaps",
        ["Working out the features"] = "the features",
        ["Splitting the rows"] = "the split",
        ["Naming the answer"] = "the answer",
        ["Filling and scaling"] = "what is learned",
        ["Measuring and drawing"] = "the report",
        ["Naming the network"] = "the network",
    };

    [GeneratedRegex("^\\*\\*Next:\\*\\* \\[\\[(?<page>[^\\]]+)\\]\\]", RegexOptions.Multiline)]
    private static partial Regex NextPage();

    [GeneratedRegex("^  - \\[\\[(?<page>[^\\]]+)\\]\\]$", RegexOptions.Multiline)]
    private static partial Regex SidebarEntry();

    [GeneratedRegex("```csharp\\n(?<code>.*?)```", RegexOptions.Singleline)]
    private static partial Regex CSharpBlock();

    /// <summary>One stage of the course: what it is called, the verbs of the courses' steps that stand for it, and the methods of a chain that write it.</summary>
    /// <param name="Name">What it is called.</param>
    /// <param name="Verbs">The verbs of the steps of the named courses that stand for it.</param>
    /// <param name="Methods">The methods a chain writes it with.</param>
    private readonly record struct Stage(string Name, string[] Verbs, string[] Methods);

    /// <summary>One verb of a chain: the stage it belongs to, the method that wrote it, and the line it stands on in its text.</summary>
    /// <param name="Stage">The place of its stage in the order.</param>
    /// <param name="Method">The method that wrote it.</param>
    /// <param name="Line">The line it stands on, from nought.</param>
    private readonly record struct Written(int Stage, string Method, int Line);

    /// <summary>Text with code in it, and where it stands.</summary>
    /// <param name="Where">The file it is in.</param>
    /// <param name="Line">The line its code starts on, from one.</param>
    /// <param name="Code">The code.</param>
    private readonly record struct Source(string Where, int Line, string Code);

    private static int StageOfMethod(string method) => Array.FindIndex(Stages, stage => stage.Methods.Contains(method));

    private static int StageNamed(string name) => Array.FindIndex(Stages, stage => stage.Name == name);

    private static bool Rising(int[] places) => places.Zip(places.Skip(1), (before, after) => before < after).All(each => each);

    // Every chain of a text that starts at a Create, as the stages its verbs belong to, in the order they are written: the
    // chain's own calls, never what a lambda inside one of them says.
    private static IEnumerable<Written[]> ChainsIn(string code)
    {
        var outermost = CSharpSyntaxTree.ParseText(code).GetRoot().DescendantNodes().OfType<InvocationExpressionSyntax>()
            .Where(call => call.Parent is not MemberAccessExpressionSyntax);

        foreach (var call in outermost)
        {
            var written = new List<Written>();
            var last = string.Empty;
            ExpressionSyntax? receiver = call;

            while (true)
            {
                // A chain written after an awaited door stands on a parenthesis: the await is the same chain, read through.
                receiver = ReadThrough(receiver);

                if (receiver is not InvocationExpressionSyntax { Expression: MemberAccessExpressionSyntax access })
                {
                    break;
                }

                last = access.Name.Identifier.Text;

                if (StageOfMethod(last) is var stage and >= 0)
                {
                    written.Add(new Written(stage, last, access.Name.GetLocation().GetLineSpan().StartLinePosition.Line));
                }

                receiver = access.Expression;
            }

            if (last == "Create")
            {
                written.Reverse();

                yield return [.. Placed(written)];
            }
        }
    }

    // One door takes every step a person writes, so the door says nothing of where the step stands in the course: the split
    // does. Above it a step puts columns on without learning; below it, it learns from the training rows.
    private static IEnumerable<Written> Placed(List<Written> chain)
    {
        var split = StageNamed("the split");
        var learned = StageNamed("what is learned");
        var divided = false;

        foreach (var each in chain)
        {
            divided |= each.Stage == split;

            yield return divided && each.Method == "Add" ? each with { Stage = learned } : each;
        }
    }

    private static ExpressionSyntax? ReadThrough(ExpressionSyntax? expression) => expression switch
    {
        ParenthesizedExpressionSyntax parenthesized => ReadThrough(parenthesized.Expression),
        AwaitExpressionSyntax awaited => ReadThrough(awaited.Expression),
        _ => expression,
    };

    private static bool IsBuilt(string path) =>
        path.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
        || path.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal);

    // Everywhere the documents write a chain: the README's blocks, every block of the wiki, and the samples' programs.
    private static IEnumerable<Source> Sources()
    {
        var readme = File.ReadAllText(Path.Join(Repository.Root, "README.md")).ReplaceLineEndings("\n");

        foreach (var block in CSharpBlock().Matches(readme).Cast<Match>())
        {
            yield return new Source("README.md", readme[..block.Groups["code"].Index].Count(letter => letter == '\n') + 1, block.Groups["code"].Value);
        }

        foreach (var page in Wiki.Pages())
        {
            foreach (var block in page.Programs.SelectMany(program => program.Blocks).Distinct())
            {
                yield return new Source(page.Name, block.Line, block.Code);
            }
        }

        var files = Directory.GetFiles(Path.Join(Repository.Root, "Samples"), "*.cs", SearchOption.AllDirectories).Where(path => !IsBuilt(path)).Order(StringComparer.Ordinal);

        foreach (var file in files)
        {
            yield return new Source(Path.GetRelativePath(Repository.Root, file).Replace('\\', '/'), 1, File.ReadAllText(file));
        }
    }

    private static string Page(string name) => File.ReadAllText(Path.Join(Wiki.Folder, name)).ReplaceLineEndings("\n");

    [Fact]
    public void TheStages_StandInTheOrderOfEveryNamedCourse()
    {
        foreach (var course in PipelineCourse.Named)
        {
            var places = course.Steps.Select(step =>
            {
                var place = Array.FindIndex(Stages, stage => stage.Verbs.Contains(step.Verb));

                Assert.True(place >= 0, $"The step '{step.Verb}' of a named course stands for no stage here.");

                return place;
            }).ToArray();

            Assert.True(Rising(places), $"The stages stand in another order than a named course's steps: {string.Join(", ", course.Steps.Select(step => step.Verb))}.");
        }
    }

    [Fact]
    public void EveryChainTheDocumentsWrite_ListsItsVerbsInTheOrderOfTheCourse()
    {
        var faults = new List<string>();
        var chains = 0;

        foreach (var source in Sources())
        {
            foreach (var chain in ChainsIn(source.Code))
            {
                chains++;

                for (var at = 1; at < chain.Length; at++)
                {
                    if (chain[at].Stage < chain[at - 1].Stage)
                    {
                        faults.Add($"{source.Where}:{source.Line + chain[at].Line}: .{chain[at].Method} ({Stages[chain[at].Stage].Name}) stands after .{chain[at - 1].Method} ({Stages[chain[at - 1].Stage].Name})");
                    }
                }
            }
        }

        Assert.True(faults.Count == 0, $"{faults.Count} chains teach another order than the course:\n{string.Join('\n', faults)}");
        Assert.True(chains >= 30, $"The documents hold {chains} chains: what this reads is not what a person reads.");
    }

    [Fact]
    public void AChainThatStartsWithAnAwaitedDoor_IsReadVerbForVerb_AsAnyOtherChainIs()
    {
        // A door that lands a window is awaited, and a chain written after it stands on a parenthesis: a walker that stopped
        // there would skip every such chain and pass whatever order it was written in.
        var chains = ChainsIn("""
            var pipeline = (await Pdd.Create().ReadBinanceAsync(window))
                .Declare(schema => schema.Timestamp("timestamp"))
                .OrderBy("timestamp");
            """).Where(chain => chain.Length > 1).ToArray();

        Assert.Equal(["ReadBinanceAsync", "Declare", "OrderBy"], Assert.Single(chains).Select(written => written.Method));
        Assert.True(Rising([.. chains[0].Select(written => written.Stage)]));
    }

    [Fact]
    public void AChainThatStartsWithAnAwaitedDoor_IsHeldToTheOrderOfTheCourse()
    {
        var chains = ChainsIn("""
            var pipeline = (await Pdd.Create().ReadBinanceAsync(window))
                .OrderBy("timestamp")
                .Declare(schema => schema.Timestamp("timestamp"));
            """).Where(chain => chain.Length > 1).ToArray();

        Assert.False(Rising([.. Assert.Single(chains).Select(written => written.Stage)]));
    }

    [Fact]
    public void AStepAddedBelowTheSplit_IsSomethingLearned_AndOneAddedAboveIt_IsAFeature()
    {
        // One door takes every step a person writes, so the door says nothing of where in the course the step stands: the
        // split does. Above it the step puts columns on without learning; below it, it learns from the training rows.
        var below = Assert.Single(ChainsIn("""
            var pipeline = Pdd.Create()
                .ReadCsv("flocks.csv")
                .Declare(schema => schema.Number("Age", "Weight"))
                .SplitAtRandom(0.70, 0.15)
                .Target("Weight")
                .Add(new CentredStep("Age", "Age_centred"))
                .Normalise("Age");
            """), chain => chain.Length > 1);
        var above = Assert.Single(ChainsIn("""
            var pipeline = Pdd.Create()
                .ReadCsv("flocks.csv")
                .Declare(schema => schema.Number("Age", "Weight"))
                .Add(new WeeksStep("Age", "AgeInWeeks"))
                .SplitAtRandom(0.70, 0.15);
            """), chain => chain.Length > 1);

        Assert.Equal(StageNamed("what is learned"), below.Single(written => written.Method == "Add").Stage);
        Assert.True(Rising([.. below.Select(written => written.Stage)]));
        Assert.Equal(StageNamed("the features"), above.Single(written => written.Method == "Add").Stage);
        Assert.True(Rising([.. above.Select(written => written.Stage)]));
    }

    [Fact]
    public void TheTutorial_WalksItsStepsInTheOrderOfTheCourse_EachPageNumberedAndLinkedAsTheTableHasIt()
    {
        var steps = Wiki.Tutorial();
        var places = steps.Where(step => StageOfPage.ContainsKey(step.Title)).Select(step => StageNamed(StageOfPage[step.Title])).ToArray();

        Assert.Equal(Enumerable.Range(1, 10), steps.Select(step => step.At));
        Assert.Equal(steps.Count - 1, places.Length);
        Assert.True(Rising(places), $"The tutorial's steps are not in the order of the course: {string.Join(" > ", steps.Select(step => step.Title))}.");

        for (var at = 0; at < steps.Count; at++)
        {
            var page = Page(steps[at].Page.Name);
            var next = NextPage().Match(page);

            Assert.StartsWith($"# Step {steps[at].At} — ", page, StringComparison.Ordinal);
            Assert.Equal(at == steps.Count - 1 ? string.Empty : steps[at + 1].Title, next.Success ? next.Groups["page"].Value : string.Empty);
        }

        Assert.Equal(steps.Select(step => step.Title), SidebarEntry().Matches(Page("_Sidebar.md")).Select(match => match.Groups["page"].Value));
    }

    [Fact]
    public void ThePipelinePage_DescribesTheAnswerBeforeWhatLearns_AndNeitherPageCallsItTheLastStep()
    {
        var page = Page("Pipeline.md");
        var answer = page.IndexOf("\n## What a model is asked to predict\n", StringComparison.Ordinal);

        Assert.True(answer >= 0, "The page has no section on what a model is asked to predict.");

        foreach (var learns in new[] { "Gaps", "Categories", "Normalising" })
        {
            Assert.True(answer < page.IndexOf($"\n## {learns}\n", StringComparison.Ordinal), $"The page describes the answer after '{learns}', which stands below it in the course.");
        }

        Assert.DoesNotContain("The last step names the answer", page, StringComparison.Ordinal);
        Assert.DoesNotContain("The last step names the answer", Page("Architecture.md"), StringComparison.Ordinal);
    }
}
