// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// An answer of several free numbers: a flock's weight at thirty-five days and its spread, each any finite number,
/// predicted together as one answer. Nothing makes them shares or classes, so they are handed over as they stand and
/// come back in their own units.
/// </summary>
public class NumbersTests
{
    private static readonly string[] Answers = ["mean", "spread"];

    // Eight flocks: their age, and the mean and the spread of their weight.
    private static InMemoryRowSource Flocks() =>
        new(
            ["age", "mean", "spread"],
            [
                .. Enumerable.Range(0, 8).Select<int, IReadOnlyList<string?>>(flock =>
                [
                    (30 + flock).ToString(CultureInfo.InvariantCulture),
                    (2000 + (37.5 * flock)).ToString(CultureInfo.InvariantCulture),
                    (-0.25 * flock).ToString(CultureInfo.InvariantCulture),
                ]),
            ]);

    private static FittingBuilder Split() =>
        Pdd.Create()
            .Read(Flocks(), "eight flocks")
            .Declare(schema => schema.Number("age", "mean", "spread"))
            .SplitAtRandom(0.50, seed: 3);

    [Fact]
    public void SeveralFreeNumbers_AreOneAnswer_HandedOverAsTheyStand()
    {
        var prepared = Split().Numbers(Answers).Build().Run();
        var batch = prepared.Batch(Part.Train);
        var flocks = prepared.Table.Identities.Where((_, row) => prepared.Parts[row] == Part.Train).Select(identity => identity.ReadAt).ToArray();

        Assert.Equal(Answers, batch.AnswerNames);
        Assert.Equal(["age"], batch.FeatureNames);
        Assert.Null(batch.Labels);

        for (var at = 0; at < flocks.Length; at++)
        {
            Assert.Equal([2000 + (37.5 * flocks[at]), -0.25 * flocks[at]], batch.Answers![at]);
        }
    }

    [Fact]
    public void FreeNumbersScaledOnTheWay_ComeBackInTheirOwnUnits()
    {
        var prepared = Split().Normalise("mean", Scale.Standard).Numbers(Answers).Build().Run();
        var batch = prepared.Batch(Part.Test);
        var flocks = prepared.Table.Identities.Where((_, row) => prepared.Parts[row] == Part.Test).Select(identity => identity.ReadAt).ToArray();

        var back = prepared.BackToOriginal(batch.Answers!, Part.Test);

        for (var at = 0; at < flocks.Length; at++)
        {
            Assert.Equal(2000 + (37.5 * flocks[at]), back[at][0], 9);
            Assert.Equal(-0.25 * flocks[at], back[at][1], 12);
        }
    }

    [Fact]
    public void FreeNumbersAreAmounts_SoAReportThatCountsClassesOfThemIsRefusedWhereItIsWritten()
    {
        var refused = Assert.Throws<DeclarationException>(() =>
            Split().Numbers(Answers).Report(report => report.Measure(Metric.Accuracy).On(Part.Test).As(Shown.Numbers)).Build());

        Assert.Contains("accuracy", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'target.numbers'", refused.Message, StringComparison.Ordinal);
        Assert.False(((INamesTheAnswer)new NumbersStep(Answers)).AnswersCanBeClasses);
        Assert.False(((INamesTheAnswer)new NumbersStep(Answers)).IsOrdered);
        Assert.Null(((INamesTheAnswer)new NumbersStep(Answers)).Refusal([-1e9, 0.5]));
    }

    [Fact]
    public void AnAnswerOfFreeNumbers_HoldsAtLeastTwo_InTheWordsEveryAnswerOfSeveralColumnsUses()
    {
        var one = Assert.Throws<ArgumentException>(() => new NumbersStep(["mean"]));
        var inAFile = Assert.Throws<PipelineFileException>(() => StepCatalog.BuiltIn().ReadStep("""{"step": "target.numbers", "columns": ["mean"]}"""));

        Assert.Contains("'columns' names at least two columns: an answer held in one column is a target.", one.Message, StringComparison.Ordinal);
        Assert.Contains("at least two columns", inAFile.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => new NumbersStep(null!));
        Assert.Throws<ArgumentException>(() => new NumbersStep(["mean", "mean"]));
    }

    [Fact]
    public void AnAnswerOfFreeNumbers_IsNewInTheEighthFile_AndIsWrittenDownAndReadBack()
    {
        var catalog = StepCatalog.BuiltIn();
        var step = new NumbersStep(Answers);
        var declaration = Split().Numbers(Answers).Build().Declaration;

        Assert.Equal(8, catalog.Describe("target.numbers").Since);
        Assert.Equal(step, PipelineDeclaration.FromJson(declaration.ToJson(), catalog).Output);
        Assert.Equal(step, declaration.Output);
        Assert.Equal(step.GetHashCode(), new NumbersStep(Answers).GetHashCode());
        Assert.NotEqual(step, new NumbersStep(["spread", "mean"]));
        Assert.False(step.Equals(null));
        Assert.Equal(Answers, step.Columns);
        Assert.Equal(Answers, step.Answers);
        Assert.Equal("target.numbers", step.Verb);
        Assert.Same(ColumnState.None, step.After(ColumnState.None));
    }
}
