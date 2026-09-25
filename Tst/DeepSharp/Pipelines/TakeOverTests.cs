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

    // ---- the chain: the schema where the chain declares it, the drops and the output where it names its answer

    // Saved from a pipeline that keeps the rest of the file and drops one column of it, and one a step makes.
    private static PipelinePreset SavedWithDrops(string path) =>
        PipelinePreset.Of(
            Pdd.Create()
                .ReadCsv(path)
                .Declare(columns => columns.Integer("survived").Category("pclass").Optional("age", ColumnKind.Number).Number("fare"), Remainder.Keep)
                .SplitStratified("survived", 0.70, 0.15)
                .FillMissing("age", With.Median)
                .Normalise("fare")
                .Drop("sex", "age_was_missing")
                .Target("survived")
                .Declaration,
            header: null);

    // Two listings are the same when every part says the same.
    private static void SameListing(PresetTakeOver expected, PresetTakeOver actual)
    {
        Assert.Equal(expected.Steps, actual.Steps);
        Assert.Equal(expected.Faults, actual.Faults);
        Assert.Equal(expected.NewColumns, actual.NewColumns);
        Assert.Equal(expected.Changes, actual.Changes);
        Assert.Equal(expected.Output, actual.Output);
        Assert.Equal(expected.DeclareOrder?.Before, actual.DeclareOrder?.Before);
        Assert.Equal(expected.DeclareOrder?.After, actual.DeclareOrder?.After);
    }

    [Fact]
    public void AChain_TakesTheSavedSchemaAtItsSource_AndTheDropsAndOutputWhereItNamesItsAnswer_ListingEachDecisionOnce()
    {
        var path = Repository.Data("titanic.csv");
        var header = CsvRowSource.HeaderOf(path);
        var saved = SavedWithDrops(path);

        var written = Pdd.Create()
            .ReadCsv(path)
            .Declare(saved, out var declared)
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Normalise("fare");
        var blocks = written.Declaration;

        written.Output(saved, out var taken);

        // The one door, over blocks equal to the chain where it names its answer; and over the source, the schema alone.
        SameListing(saved.TakeOver(blocks, header), taken);

        var atTheSource = new PipelinePreset(saved.Declare).TakeOver(new PipelineDeclaration([new ReadCsvStep(path)]), header);

        Assert.Equal(atTheSource.Steps, declared.Steps);
        Assert.Equal(atTheSource.Changes, declared.Changes);

        // What is new in the source is said of the whole of the saved decisions, the columns they drop too.
        Assert.Equal(taken.NewColumns, declared.NewColumns);
        Assert.DoesNotContain("sex", declared.NewColumns!);
        Assert.Contains("sex", atTheSource.NewColumns!);

        // Each decision once: the schema's — its columns, and the rest it keeps — where it is taken; the drops and the
        // output where they are placed.
        Assert.Equal(header, declared.Changes.Select(change => change.Column));
        Assert.Equal(ColumnStanding.Kept, declared.Changes.Single(change => change.Column == "sex").After.Standing);
        Assert.Equal(["sex", "age_was_missing"], taken.Changes.Select(change => change.Column));
        Assert.Null(declared.Output);
        Assert.Equal(new OutputChange(null, new TargetStep("survived")), taken.Output);

        // The chain goes on from the door's steps: its placement, not the pipeline the decisions were saved from.
        Assert.Equal(taken.Steps, written.Declaration.Steps);
    }

    [Fact]
    public void AChainWhoseSourceCannotBeReadYet_TakesTheSchemaOver_WithoutSayingWhatTheSourceHas()
    {
        var chain = Pdd.Create().ReadCsv(Path.Join(Path.GetTempPath(), "nowhere-yet.csv")).Declare(Saved(Blocks()), out var declared);

        Assert.Null(declared.NewColumns);
        Assert.All(declared.Changes, change => Assert.Null(change.InSource));
        Assert.Equal(Blocks().Steps[1], chain.Declaration.Steps[1]);
    }

    [Fact]
    public void AChainThatNamesNoSourceYet_TakesTheSchemaOver_KnowingNothingOfTheSourcesColumns()
    {
        var chain = Pdd.Create().Declare(Saved(Blocks()), out var declared);

        Assert.Null(declared.NewColumns);
        Assert.Equal([Blocks().Steps[1]], chain.Declaration.Steps);
        Assert.Equal(["survived", "pclass", "age", "fare"], declared.Changes.Select(change => change.Column));
    }

    [Fact]
    public void RowsHandedIn_AreTheSourcesColumns_ForTheChainsListing()
    {
        var rows = CsvRowSource.FromText("survived,pclass,sex,age,fare\n1,3,male,22,7.25\n", "passengers");

        Pdd.Create().Read(rows, "passengers").Declare(Saved(Blocks()), out var declared);

        Assert.Equal(["sex"], declared.NewColumns);
        Assert.All(declared.Changes, change => Assert.True(change.InSource));
    }

    [Fact]
    public void ADecisionTheChainCannotKeep_IsRefusedWithEveryFault_WhereItIsTakenOver_AndTheChainStaysAsItWas()
    {
        // The saved schema has neither age nor fare; one chain orders its rows by fare, the other fills age.
        var narrow = Saved(Pdd.Create().ReadCsv("titanic.csv").Declare(columns => columns.Integer("survived", "pclass")).Declaration);
        var ordered = Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(columns => columns.Integer("survived", "pclass").Optional("age", ColumnKind.Number).Number("fare"))
            .OrderBy("fare");
        var filled = Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(columns => columns.Integer("survived", "pclass").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median);

        var atTheSchema = Assert.Throws<DeclarationException>(() => ordered.Declare(narrow, out _));
        var atTheOutput = Assert.Throws<DeclarationException>(() => filled.Output(narrow, out _));

        Assert.Contains(atTheSchema.Faults, fault => fault.Message.Contains("'fare'", StringComparison.Ordinal));
        Assert.Contains(atTheOutput.Faults, fault => fault.Message.Contains("'age'", StringComparison.Ordinal));
        Assert.Equal(3, ordered.Declaration.Steps.Count);
        Assert.Equal(4, filled.Declaration.Steps.Count);
    }
}
