// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Saved column decisions taken over into a pipeline: the schema whole, the drops made equal, the output placed whole,
/// and — before anything is applied — a listing of every column whose decision it changes, the output before and after,
/// and the schema's order when that alone changes. A take-over whose result breaks a rule is refused with every fault
/// and applies nothing.
/// </summary>
public class TakeOverTests
{
    private static readonly string[] Header = ["survived", "pclass", "sex", "age", "fare", "embarked"];

    private static PipelineDeclaration Blocks(Action<SchemaBuilder>? schema = null) =>
        Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema ?? (columns => columns.Integer("survived", "pclass").Optional("age", ColumnKind.Number).Number("fare")))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Normalise("fare")
            .Declaration;

    private static PipelineDeclaration Then(IEnumerable<IPipelineStep> steps) => new(steps);

    private static PipelinePreset Saved(PipelineDeclaration decided, IReadOnlyList<string>? source = null) => PipelinePreset.Of(decided, source);

    [Fact]
    public void TheSameDecisions_ListNothing_AndGiveTheStepsBack()
    {
        var takenOver = Saved(Blocks()).TakeOver(Blocks(), Header);

        Assert.Equal(Blocks().Steps, takenOver.Steps);
        Assert.Empty(takenOver.Changes);
        Assert.Empty(takenOver.Faults);
        Assert.Null(takenOver.Output);
        Assert.Null(takenOver.DeclareOrder);
    }

    [Fact]
    public void EveryColumnWhoseDecisionChanges_IsListed_BeforeAndAfter_InTheSourcesOrder()
    {
        var decided = Then(Then(Blocks().Excluding("pclass")).WithKind("survived", ColumnKind.Category));

        var takenOver = Saved(decided).TakeOver(Blocks(), Header);
        var survived = takenOver.Changes[0];
        var pclass = takenOver.Changes[1];

        Assert.Equal(["survived", "pclass"], takenOver.Changes.Select(change => change.Column));
        Assert.Equal(ColumnKind.Integer, survived.Before.Kind);
        Assert.Equal(ColumnKind.Category, survived.After.Kind);
        Assert.Equal(ColumnKind.Integer, survived.After.Was);
        Assert.Equal(ColumnStanding.Taking, pclass.Before.Standing);
        Assert.Equal(ColumnStanding.Excluded, pclass.After.Standing);
        Assert.True(pclass.InSource);
        Assert.Equal(decided.Steps, takenOver.Steps);
    }

    [Fact]
    public void AnOutputDecided_IsPlacedWhole_AndListedOnce_AndNoneDecidedTakesTheOneStandingAway()
    {
        var target = new TargetStep("survived");
        var answered = Then([.. Blocks().Steps, target]);

        var placed = Saved(answered).TakeOver(Blocks(), Header);
        var removed = Saved(Blocks()).TakeOver(answered, Header);

        Assert.Equal(new OutputChange(null, target), placed.Output);
        Assert.Equal(answered.Steps, placed.Steps);
        Assert.Empty(placed.Changes);
        Assert.Equal(new OutputChange(target, null), removed.Output);
        Assert.Equal(Blocks().Steps, removed.Steps);
    }

    [Fact]
    public void AnOutputWhoseOnlyChangeIsWhichColumn_ListsTheOutput_AndNoColumnWhoseDecisionStayed()
    {
        // The answer column changes its role and what it offers; neither is a decision about the column.
        var byClass = Then([.. Blocks().Steps, new TargetStep("pclass")]);

        var takenOver = Saved(byClass).TakeOver(Then([.. Blocks().Steps, new TargetStep("survived")]), Header);

        Assert.Equal(new OutputChange(new TargetStep("survived"), new TargetStep("pclass")), takenOver.Output);
        Assert.Empty(takenOver.Changes);
    }

    [Fact]
    public void ASchemaThatDiffersOnlyInOrder_ListsTheOrder_AndANameOnlyOneDeclaresIsAChangeNotAnOrder()
    {
        var reordered = Blocks(columns => columns.Integer("survived").Number("fare").Integer("pclass").Optional("age", ColumnKind.Number));
        var widened = Blocks(columns => columns.Integer("survived", "pclass").Text("sex").Optional("age", ColumnKind.Number).Number("fare"));

        var order = Saved(reordered).TakeOver(Blocks(), Header);
        var sex = Saved(widened).TakeOver(Blocks(), Header);

        Assert.Equal(["survived", "pclass", "age", "fare"], order.DeclareOrder!.Value.Before);
        Assert.Equal(["survived", "fare", "pclass", "age"], order.DeclareOrder!.Value.After);
        Assert.Empty(order.Changes);
        Assert.Null(sex.DeclareOrder);
        Assert.Equal(["sex"], sex.Changes.Select(change => change.Column));
        Assert.Equal(ColumnStanding.NotDeclared, sex.Changes[0].Before.Standing);
    }

    [Fact]
    public void AResultThatBreaksARule_IsRefusedWithEveryFault_AndAppliesNothing()
    {
        // The saved schema has no age, and the fill below reads it.
        var withoutAge = Saved(Pdd.Create().ReadCsv("titanic.csv").Declare(columns => columns.Integer("survived", "pclass").Number("fare")).Declaration);

        var refused = withoutAge.TakeOver(Blocks(), Header);

        Assert.NotEmpty(refused.Faults);
        Assert.Contains(refused.Faults, fault => fault.Message.Contains("'age'", StringComparison.Ordinal));
        Assert.Empty(refused.Steps);
        Assert.Empty(refused.Changes);
        Assert.Null(refused.Output);
        Assert.Null(refused.DeclareOrder);
    }

    [Fact]
    public void AColumnAStepMade_IsListedWhenItsDropChanges_AndIsNeverSaidToBeMissingFromTheSource()
    {
        var dropped = Then(Blocks().Excluding("age_was_missing"));
        var embarked = Blocks(columns => columns.Integer("survived", "pclass").Optional("age", ColumnKind.Number).Number("fare").Text("embarked"));
        string[] noEmbarked = ["survived", "pclass", "sex", "age", "fare"];

        var made = Saved(Blocks()).TakeOver(dropped, Header).Changes.Single(change => change.Column == "age_was_missing");
        var missing = Saved(embarked).TakeOver(Blocks(), noEmbarked).Changes.Single(change => change.Column == "embarked");

        Assert.Equal(ColumnStanding.Dropped, made.Before.Standing);
        Assert.Equal(ColumnStanding.Made, made.After.Standing);
        Assert.Null(made.InSource);
        Assert.False(missing.InSource);
    }

    [Fact]
    public void TheColumnsNewToTheSource_AreThoseTheSavedFileNeverShowed()
    {
        string[] shown = ["survived", "pclass", "sex", "age", "fare"];

        var fromSource = Saved(Blocks(), shown).TakeOver(Blocks(), Header);
        var fromNames = Saved(Blocks()).TakeOver(Blocks(), Header);
        var unknown = Saved(Blocks()).TakeOver(Then(Blocks().Excluding("pclass")), header: null);

        Assert.Equal(["embarked"], fromSource.NewColumns);
        Assert.Equal(["sex", "embarked"], fromNames.NewColumns);
        Assert.Null(unknown.NewColumns);
        Assert.All(unknown.Changes, change => Assert.Null(change.InSource));
        Assert.Equal(["pclass"], unknown.Changes.Select(change => change.Column));
    }

    [Fact]
    public void BlocksWithoutASchema_TakeTheSavedOneDirectlyAfterTheSource()
    {
        var source = new PipelineDeclaration([new ReadCsvStep("titanic.csv")]);
        var schema = Blocks().Steps[1];

        var takenOver = Saved(Blocks()).TakeOver(source, Header);

        Assert.Equal([source.Steps[0], schema], takenOver.Steps);
        Assert.Equal(["survived", "pclass", "age", "fare"], takenOver.Changes.Select(change => change.Column));
    }
}
