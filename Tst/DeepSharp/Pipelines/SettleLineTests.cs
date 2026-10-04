// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// One line settles the gaps of many columns, each the way it is named, where the features are worked out.
/// </summary>
/// <remarks>
/// The same door every other per-column verb has, and the same fan-out behind it: what reaches the declaration is one
/// step a column. The kinds it offers are the ones no row decides — a nought, a number you choose, a refusal — and the
/// ways a fill learns from the training rows are not among them, because this line's verb would throw on them.
/// </remarks>
public class SettleLineTests
{
    private static PipelineBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Optional("age", ColumnKind.Number).Number("fare"));

    [Fact]
    public void OneLine_DeclaresOneStepAColumn_InTheOrderTheyWereNamed()
    {
        var declaration = Passengers()
            .SettleGaps(gaps => gaps.Zero("age").Constant(-1, "fare", "sibsp"))
            .SplitStratified("survived", 0.70, 0.15)
            .Declaration;

        var settled = declaration.Steps.OfType<SettleGapsStep>().ToArray();

        Assert.Equal(["age", "fare", "sibsp"], settled.Select(step => step.Column));
        Assert.Equal([With.Zero, With.Constant(-1), With.Constant(-1)], settled.Select(step => step.Strategy));
        Assert.Equal("settle.gaps", Assert.Single(settled.Select(step => step.Verb).Distinct()));
    }

    [Fact]
    public void ALineThatNamesNoColumn_IsRefusedWhereItStands()
    {
        var refused = Assert.Throws<ArgumentException>(() => Passengers().SettleGaps(gaps => { }));

        Assert.Contains("gaps", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SettlingStandsWhereTheFeaturesAre_AndFillingStillDoesNot()
    {
        // The whole point of the verb: settled first, the feature worked out from settled columns, and only then the
        // split and what learns from the training rows.
        var built = Passengers()
            .SettleGaps(gaps => gaps.Zero("age"))
            .AddFeature("older", "age", Arithmetic.Plus, "sibsp")
            .SplitStratified("survived", 0.70, 0.15)
            .Normalise("age", "older", "fare")
            .Target("survived")
            .Build();

        var prepared = built.Run();

        Assert.NotEmpty(prepared.Batch(Part.Train).Features);
        Assert.DoesNotContain(Enumerable.Range(0, prepared.Table.RowCount), prepared.Table["older"].IsMissing);
        Assert.Equal(891, prepared.Table.RowCount);
    }

    [Fact]
    public void AStepThatLearns_IsStillRefusedAboveTheSplit()
    {
        var refused = Assert.Throws<DeclarationException>(() => new PipelineDeclaration([
            new ReadCsvStep("titanic.csv"),
            new DeclareStep([new ColumnDeclaration("age", ColumnKind.Number, true)]),
            FillMissingStep.Of("age", With.Median),
            new SplitAtRandomStep(new SplitShares(0.70, 0.15, 0.15), 1),
        ]));

        Assert.Equal("fill.missing", Assert.Single(refused.Faults).Verb);
    }

    [Fact]
    public void SettlingBelowTheSplit_IsWrittenToo_SinceNothingAboutItNeedsTheSplit()
    {
        var declaration = Passengers()
            .SplitStratified("survived", 0.70, 0.15)
            .Add(new SettleGapsStep("age", With.Zero))
            .Declaration;

        Assert.Equal("settle.gaps", declaration.Steps[^1].Verb);
    }
}
