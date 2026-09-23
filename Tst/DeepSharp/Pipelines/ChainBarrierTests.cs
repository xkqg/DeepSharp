// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The rule this library exists to keep — nothing learns from the data before it has been split — held at
/// the only place both doors pass through. Method placement alone is not enough and a council proved it:
/// the builder's own extension point took a learning step, and a hand-written file could put one anywhere
/// it liked. So the invariant lives on the declaration, and these tests come at it from every side.
/// </summary>
public class ChainBarrierTests
{
    private static IEnumerable<string> PublicMethodsOf<T>() =>
        typeof(T).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                 .Select(method => method.Name);

    [Fact]
    public void BeforeTheSplit_NothingThatLearnsIsEvenOffered()
    {
        var offered = PublicMethodsOf<PipelineBuilder>().ToList();

        Assert.DoesNotContain("FillMissing", offered);
        Assert.DoesNotContain("Normalise", offered);
        Assert.DoesNotContain("Encode", offered);
    }

    [Fact]
    public void AfterTheSplit_TheyAre()
    {
        Assert.Contains("FillMissing", PublicMethodsOf<FittingBuilder>());
    }

    [Fact]
    public void TheSplitIsWhatMovesYouAcross()
    {
        Assert.Contains("SplitByTime", PublicMethodsOf<PipelineBuilder>());
        Assert.NotEqual(typeof(PipelineBuilder), typeof(FittingBuilder));

        var split = typeof(PipelineBuilder).GetMethod(
            "SplitByTime", [typeof(string), typeof(double), typeof(double)]);

        Assert.NotNull(split);
        Assert.Equal(typeof(FittingBuilder), split.ReturnType);
    }

    [Fact]
    public void AndYouCannotGoBack()
    {
        var returns = typeof(FittingBuilder)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(method => method.ReturnType);

        Assert.DoesNotContain(typeof(PipelineBuilder), returns);
    }

    [Fact]
    public void TheExtensionPointWillNotTakeALearningStep()
    {
        // `Add` is the door every verb from every other package comes through, so it is also the door a
        // learning verb would come through. It takes the narrower kind, and a step that learns is refused
        // where it is written rather than in a number three weeks later.
        var before = Pdd.Create().ReadCsv("x.csv");

        Assert.Throws<InvalidOperationException>(() => before.Add(new FillMissingStep("age", With.Mean)));
    }

    [Fact]
    public void AFileThatFitsBeforeItSplits_RefusesToLoad()
    {
        // The same rule, from the other door. A hand-edited file that moves the fill above the split is
        // the most ordinary way this leak arrives, and it is the one the type system cannot see.
        const string json = """
            {"declaration":[{"step":"read.csv","path":"x.csv"},
                            {"step":"fill.missing","column":"age","with":"mean"},
                            {"step":"split.byTime","column":"t","train":0.7,"validation":0.15,"test":0.15}]}
            """;

        var refused = Assert.Throws<InvalidOperationException>(() => PipelineDeclaration.FromJson(json));

        Assert.Contains("fill.missing", refused.Message);
        Assert.Contains("split", refused.Message);
    }

    [Fact]
    public void AFileThatFitsWithoutSplittingAtAll_RefusesToLoad()
    {
        const string json = """{"declaration":[{"step":"fill.missing","column":"age","with":"mean"}]}""";

        Assert.Throws<InvalidOperationException>(() => PipelineDeclaration.FromJson(json));
    }

    [Fact]
    public void ADeclarationBuiltByHandFromStepsInTheWrongOrder_IsRefusedToo()
    {
        // The declaration's own constructor is the one place the builder, the extension point and the file
        // all pass through, which is why the rule lives there and not in three places that must agree.
        var wrongOrder = new IPipelineStep[]
        {
            new FillMissingStep("age", With.Mean),
            new SplitByTimeStep("t", new SplitShares(0.70, 0.15, 0.15)),
        };

        Assert.Throws<InvalidOperationException>(() => new PipelineDeclaration(wrongOrder));
    }

    [Fact]
    public void OnceSplit_TheBuilderYouStartedWithIsSpent()
    {
        // Holding on to the pre-split builder used to let a feature be declared after the split, into the
        // same list, which is the leak arriving from the side. The builder says so instead.
        var before = Pdd.Create().ReadCsv("a.csv");
        before.SplitByTime("t", 0.70, 0.15);

        Assert.Throws<InvalidOperationException>(() => before.Add(new ReadCsvStep("b.csv")));
        Assert.Throws<InvalidOperationException>(() => before.SplitByTime("t2", 0.50, 0.25));
    }

    [Fact]
    public void TwoPipelinesFromOneStart_AreNotQuietlyTheSamePipeline()
    {
        // A parameter sweep that reuses a common prefix used to produce one declaration containing every
        // arm, with each arm reporting the other's steps as its own.
        var first = Pdd.Create().ReadCsv("a.csv").SplitByTime("t", 0.70, 0.15);
        var second = Pdd.Create().ReadCsv("a.csv").SplitByTime("t", 0.50, 0.25);

        first.FillMissing("age", With.Mean);
        second.FillMissing("age", With.Median);

        Assert.NotEqual(first.Declaration, second.Declaration);
        Assert.Equal(3, first.Declaration.Steps.Count);
        Assert.Equal(3, second.Declaration.Steps.Count);
    }
}
