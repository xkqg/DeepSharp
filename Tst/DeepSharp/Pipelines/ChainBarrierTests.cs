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
        var before = Pdd.Create().ReadCsv("x.csv").Declare(schema => schema.Number("age"));

        Assert.Throws<DeclarationException>(() => before.Add(FillMissingStep.Of("age", With.Mean)));
    }

    public static TheoryData<ISplitStep> EverySplit => new()
    {
        new SplitByPartColumnStep("part"),
        new SplitAtRandomStep(new SplitShares(0.70, 0.15, 0.15), 1),
        new SplitStratifiedStep("part", new SplitShares(0.70, 0.15, 0.15), 1),
        new SplitByTimeStep("t", new SplitShares(0.70, 0.15, 0.15)),
    };

    [Theory]
    [MemberData(nameof(EverySplit))]
    public void TheExtensionPointWillNotTakeASplit_NamingTheSplitDoor(ISplitStep split)
    {
        // A split that came in through Add stayed on the builder that does not learn, so what learns was then taken by the
        // same door, never ranked by the declared id, and the shares held back were never given to it.
        var before = Pdd.Create().ReadCsv("x.csv").Declare(schema => schema.Timestamp("t").Number("age", "part"));

        var refused = Assert.Throws<InvalidOperationException>(() => before.Add(split));

        Assert.Contains("Split(", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACallersSplitEntersByTheSplitDoor_AndHandsBackTheBuilderThatLearns()
    {
        var rows = new InMemoryRowSource(
            ["age", "part"],
            [["1", "0"], ["2", "0"], ["3", "0"], ["4", "1"], ["5", "2"], ["6", "2"]]);

        var prepared = Pdd.Create()
            .Read(rows, "six rows, each naming its part")
            .Declare(schema => schema.Number("age", "part"))
            .Split(new SplitByPartColumnStep("part"))
            .Normalise("age")
            .Build()
            .Run();

        Assert.Equal([Part.Train, Part.Train, Part.Train, Part.Validation, Part.Test, Part.Test], prepared.Parts);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ASplitOfYourOwn_AfterAShareWasHeldBack_IsRefused(bool predict)
    {
        // A share held back to predict or to calibrate on is given to the split the library writes; a split of the caller's
        // deals its own parts, so the share would be a promise nothing keeps.
        var before = Pdd.Create().ReadCsv("x.csv").Declare(schema => schema.Number("age", "part"));
        var held = predict ? before.Predict(10) : before.Calibrate(10);

        var refused = Assert.Throws<InvalidOperationException>(() => held.Split(new SplitByPartColumnStep("part")));

        Assert.Contains(predict ? "predict on" : "calibrate on", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileThatFitsBeforeItSplits_RefusesToLoad()
    {
        // The same rule, from the other door. A hand-edited file that moves the fill above the split is
        // the most ordinary way this leak arrives, and it is the one the type system cannot see.
        const string json = """
            {"version":2,"declaration":[{"step":"read.csv","path":"x.csv"},
                            {"step":"declare","remainder":"drop","columns":[{"name":"t","kind":"timestamp","optional":false},{"name":"age","kind":"number","optional":false}]},
                            {"step":"fill.missing","column":"age","with":"mean"},
                            {"step":"split.byTime","column":"t","train":0.7,"validation":0.15,"test":0.15}]}
            """;

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Contains("fill.missing", refused.Message);
        Assert.Contains("split", refused.Message);
    }

    [Fact]
    public void AFileThatFitsWithoutSplittingAtAll_RefusesToLoad()
    {
        const string json = """
            {"declaration":[{"step":"declare","remainder":"drop","columns":[{"name":"age","kind":"number","optional":false}]},
                            {"step":"fill.missing","column":"age","with":"mean"}]}
            """;

        Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    [Fact]
    public void ADeclarationBuiltByHandFromStepsInTheWrongOrder_IsRefusedToo()
    {
        // The declaration's own constructor is the one place the builder, the extension point and the file
        // all pass through, which is why the rule lives there and not in three places that must agree.
        var wrongOrder = new IPipelineStep[]
        {
            new DeclareStep([new ColumnDeclaration("t", ColumnKind.Timestamp, false), new ColumnDeclaration("age", ColumnKind.Number, false)]),
            FillMissingStep.Of("age", With.Mean),
            new SplitByTimeStep("t", new SplitShares(0.70, 0.15, 0.15)),
        };

        Assert.Throws<DeclarationException>(() => new PipelineDeclaration(wrongOrder));
    }

    [Fact]
    public void OnceSplit_TheBuilderYouStartedWithIsSpent()
    {
        // Holding on to the pre-split builder used to let a feature be declared after the split, into the
        // same list, which is the leak arriving from the side. The builder says so instead.
        var before = Pdd.Create().ReadCsv("a.csv").Declare(schema => schema.Timestamp("t", "t2"));
        before.SplitByTime("t", 0.70, 0.15);

        Assert.Throws<InvalidOperationException>(() => before.Add(new ReadCsvStep("b.csv")));
        Assert.Throws<InvalidOperationException>(() => before.SplitByTime("t2", 0.50, 0.25));
    }

    [Fact]
    public void TwoPipelinesFromOneStart_AreNotQuietlyTheSamePipeline()
    {
        // A parameter sweep that reuses a common prefix used to produce one declaration containing every
        // arm, with each arm reporting the other's steps as its own.
        var first = Pdd.Create().ReadCsv("a.csv").Declare(schema => schema.Timestamp("t").Number("age")).SplitByTime("t", 0.70, 0.15);
        var second = Pdd.Create().ReadCsv("a.csv").Declare(schema => schema.Timestamp("t").Number("age")).SplitByTime("t", 0.50, 0.25);

        first.FillMissing("age", With.Mean);
        second.FillMissing("age", With.Median);

        Assert.NotEqual(first.Declaration, second.Declaration);
        Assert.Equal(4, first.Declaration.Steps.Count);
        Assert.Equal(4, second.Declaration.Steps.Count);
    }
}
