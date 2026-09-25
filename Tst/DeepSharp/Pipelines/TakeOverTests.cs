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
    public void TheColumnsNewToASource_AreThoseTheDecisionsLastShowedWithout_OrElseThoseTheyDoNotName()
    {
        string[] shown = ["survived", "pclass", "sex", "age", "fare"];
        var dropped = Saved(Then(Blocks().Excluding("fare")));

        Assert.Equal(["embarked"], Saved(Blocks(), shown).NewColumns(Header));
        Assert.Equal(["sex", "embarked"], Saved(Blocks()).NewColumns(Header));

        // A column the decisions drop is one they name.
        Assert.Equal(["sex", "embarked"], dropped.NewColumns(Header));
        Assert.Empty(Saved(Blocks(), Header).NewColumns(Header));
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
        Assert.Equal(expected.DeclareRemainder, actual.DeclareRemainder);
        Assert.Equal(expected.DropsNotMade, actual.DropsNotMade);
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

    [Fact]
    public void AChainWithoutTheStepThatMakesASavedDrop_ListsItAsNotMadeWhereItNamesItsAnswer_AndOnlyThere()
    {
        var path = Repository.Data("titanic.csv");
        var saved = PipelinePreset.Of(FillsAndDrops(path).Declaration, header: null);

        var written = Pdd.Create()
            .ReadCsv(path)
            .Declare(saved, out var declared)
            .SplitStratified("survived", 0.70, 0.15)
            .Normalise("fare");

        written.Output(saved, out var taken);

        Assert.Empty(declared.DropsNotMade);
        Assert.Equal(["age_was_missing"], taken.DropsNotMade.Select(drop => drop.Column));
        Assert.Equal(ColumnStanding.NotDeclared, taken.DropsNotMade[0].After.Standing);
        Assert.False(taken.DropsNotMade[0].InSource);
        Assert.DoesNotContain(taken.Changes, change => change.Column == "age_was_missing");
        Assert.DoesNotContain(written.Declaration.Steps, step => step is DropColumnsStep);
    }

    // ---- a saved drop the take-over cannot make: listed, never made, never refused

    private static readonly string[] Seaborn =
        ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "class", "who", "adult_male", "deck", "embark_town", "alive", "alone"];

    private static PipelineBuilder Head(Action<SchemaBuilder>? schema = null, Remainder remainder = Remainder.Drop) =>
        Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema ?? (columns => columns.Integer("survived", "pclass").Optional("age", ColumnKind.Number).Number("fare")), remainder);

    // The pipeline the drops were saved from: the fill makes age_was_missing, and a drop leaves it out.
    private static FittingBuilder FillsAndDrops(string path = "titanic.csv", Remainder remainder = Remainder.Drop) =>
        Pdd.Create()
            .ReadCsv(path)
            .Declare(columns => columns.Integer("survived", "pclass").Optional("age", ColumnKind.Number).Number("fare"), remainder)
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Normalise("fare")
            .Drop("age_was_missing")
            .Target("survived");

    private static PipelineDeclaration WithMaker(Action<SchemaBuilder>? schema = null, Remainder remainder = Remainder.Drop) =>
        Head(schema, remainder).SplitStratified("survived", 0.70, 0.15).FillMissing("age", With.Median).Normalise("fare").Target("survived").Declaration;

    private static PipelineDeclaration NoMaker(Remainder remainder = Remainder.Drop) =>
        Head(remainder: remainder).SplitStratified("survived", 0.70, 0.15).Normalise("fare").Target("survived").Declaration;

    // Blocks that declare embarked and write it down as numbers, one column per port, which takes embarked away.
    private static PipelineDeclaration Encoded(As how = As.OneHot) =>
        Head(columns => columns.Integer("survived", "pclass").Optional("age", ColumnKind.Number).Number("fare").Text("embarked"))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Encode("embarked", how)
            .Normalise("fare")
            .Target("survived")
            .Declaration;

    private static PipelinePreset DropsEmbarked(DeclareStep declare) => new(declare, ["embarked"], new TargetStep("survived"), Seaborn);

    private static DeclareStep ExcludesSex() => new(
    [
        new ColumnDeclaration("survived", ColumnKind.Integer, false), new ColumnDeclaration("pclass", ColumnKind.Integer, false),
        new ColumnDeclaration("age", ColumnKind.Number, true), new ColumnDeclaration("fare", ColumnKind.Number, false),
        new ColumnDeclaration("sex", ColumnKind.Text, false) { Excluded = true },
    ]);

    [Fact]
    public void ASavedDropOfAColumnNothingMakesOrTheSchemaLeavesOut_IsListedAsNotMade_AndTheStepsStayAsTheyAre()
    {
        var fromTheFill = PipelinePreset.Of(FillsAndDrops().Declaration, Seaborn);

        var known = fromTheFill.TakeOver(NoMaker(), Seaborn);
        var unknown = fromTheFill.TakeOver(NoMaker(), header: null);
        var embarked = DropsEmbarked(fromTheFill.Declare).TakeOver(WithMaker(), Seaborn);

        Assert.Equal(NoMaker().Steps, known.Steps);
        Assert.Empty(known.Changes);
        Assert.Equal(["age_was_missing"], known.DropsNotMade.Select(drop => drop.Column));
        Assert.Equal(ColumnStanding.NotDeclared, known.DropsNotMade[0].After.Standing);
        Assert.False(known.DropsNotMade[0].InSource);
        Assert.Equal(["age_was_missing"], unknown.DropsNotMade.Select(drop => drop.Column));
        Assert.Null(unknown.DropsNotMade[0].InSource);
        Assert.Equal(WithMaker().Steps, embarked.Steps);
        Assert.Equal(["embarked"], embarked.DropsNotMade.Select(drop => drop.Column));
        Assert.True(embarked.DropsNotMade[0].InSource);
    }

    [Fact]
    public void ASavedDropOfAColumnAStepBelowTakesAway_IsListedAsNotMade_WithHowItStands()
    {
        var consumed = DropsEmbarked(Encoded().Steps.OfType<DeclareStep>().Single());
        var months = Pdd.Create()
            .ReadCsv("series.csv")
            .Declare(schema => schema.Timestamp("Date").Number("close"))
            .OrderBy("Date")
            .TimeParts("Date", TimePart.Month)
            .SplitByTime("Date", 0.70, 0.15, 5)
            .EncodeCategories()
            .Target("close")
            .Declaration;
        var savedMonths = new PipelinePreset(months.Steps.OfType<DeclareStep>().Single(), ["Date_month"], new TargetStep("close"));

        var encoded = consumed.TakeOver(Encoded(), Seaborn);
        var madeThenTaken = savedMonths.TakeOver(months, ["Date", "close"]);

        Assert.Equal(Encoded().Steps, encoded.Steps);
        Assert.Equal(["embarked"], encoded.DropsNotMade.Select(drop => drop.Column));
        Assert.Equal(ColumnStanding.Taking, encoded.DropsNotMade[0].After.Standing);
        Assert.True(encoded.DropsNotMade[0].InSource);
        Assert.Null(consumed.TakeOver(Encoded(), header: null).DropsNotMade[0].InSource);
        Assert.Equal(["Date_month"], madeThenTaken.DropsNotMade.Select(drop => drop.Column));
        Assert.Equal(ColumnStanding.Made, madeThenTaken.DropsNotMade[0].After.Standing);
        Assert.Null(madeThenTaken.DropsNotMade[0].InSource);
    }

    [Fact]
    public void ASavedDropThatIsPlaced_AlreadyHeld_OrCoveredByTheSavedSchema_IsNeverListedAsNotMade_AndEachIsListedOnce()
    {
        var fromTheFill = PipelinePreset.Of(FillsAndDrops().Declaration, Seaborn);
        var kept = PipelinePreset.Of(FillsAndDrops(remainder: Remainder.Keep).Declaration, Seaborn);
        var fareDropped = PipelinePreset.Of(
            Head().SplitStratified("survived", 0.70, 0.15).FillMissing("age", With.Median).Normalise("fare").Drop("fare").Target("survived").Declaration, Seaborn);
        var noReaderOfFare = Head().SplitStratified("survived", 0.70, 0.15).FillMissing("age", With.Median).Target("survived").Declaration;
        var sexDropped = PipelinePreset.Of(new PipelineDeclaration(WithMaker(remainder: Remainder.Keep).Excluding("sex")), Seaborn);

        // A source that carries a column of the name the fill would make, kept with the rest of the file: its drop is placed.
        string[] carries = [.. Seaborn, "age_was_missing"];

        TakeOverCase[] cases =
        [
            new(fromTheFill, WithMaker(), Seaborn),
            new(fromTheFill, FillsAndDrops().Declaration, Seaborn),
            new(new PipelinePreset(ExcludesSex(), ["sex"], new TargetStep("survived"), Seaborn), WithMaker(), Seaborn),
            new(fareDropped, noReaderOfFare, Seaborn),
            new(kept, NoMaker(Remainder.Keep), Seaborn),
            new(kept, NoMaker(Remainder.Keep), null),
            new(kept, NoMaker(Remainder.Keep), carries),
            new(sexDropped, WithMaker(remainder: Remainder.Keep), Seaborn),
            new(DropsEmbarked(Encoded().Steps.OfType<DeclareStep>().Single()), Encoded(As.Ordinal), Seaborn),
        ];

        Assert.All(cases, each =>
        {
            var takenOver = each.Saved.TakeOver(each.Into, each.Header);

            Assert.Empty(takenOver.Faults);
            Assert.Empty(takenOver.DropsNotMade);
            Assert.All(each.Saved.Drop, column =>
            {
                var listed = takenOver.Changes.Count(change => change.Column == column);
                var held = each.Into.ChoicesFor([column]).Rows[0].Standing is ColumnStanding.Dropped or ColumnStanding.Excluded;

                Assert.Equal(1, listed + (listed == 0 && held ? 1 : 0));
            });
        });
    }

    [Fact]
    public void ARefusedTakeOver_ListsNoDropAsNotMade()
    {
        var withoutAge = new PipelinePreset(
            Pdd.Create().ReadCsv("titanic.csv").Declare(columns => columns.Integer("survived", "pclass").Number("fare")).Declaration.Steps.OfType<DeclareStep>().Single(),
            ["colour"]);

        var refused = withoutAge.TakeOver(Blocks(), Header);

        Assert.NotEmpty(refused.Faults);
        Assert.Empty(refused.DropsNotMade);
        Assert.Null(refused.DeclareRemainder);
    }

    [Fact]
    public void AColumnWhoseOnlyChangeIsWhetherTheSourceMayLackIt_IsListed()
    {
        var ageRequired = WithMaker(columns => columns.Integer("survived", "pclass").Number("age").Number("fare"));

        var required = PipelinePreset.Of(ageRequired, Seaborn).TakeOver(WithMaker(), Seaborn);
        var optional = PipelinePreset.Of(WithMaker(), Seaborn).TakeOver(ageRequired, Seaborn);

        var age = Assert.Single(required.Changes);

        Assert.Equal("age", age.Column);
        Assert.Equal(true, age.Before.Optional);
        Assert.Equal(false, age.After.Optional);
        Assert.Equal(false, Assert.Single(optional.Changes).Before.Optional);
        Assert.Null(WithMaker().ChoicesFor(["sex"]).Rows[0].Optional);
    }

    [Fact]
    public void WhatBecomesOfTheColumnsTheSchemaDoesNotName_IsListedWhenItChanges_WhetherTheSourcesColumnsAreKnownOrNot()
    {
        var keep = PipelinePreset.Of(WithMaker(remainder: Remainder.Keep), Seaborn);
        var refuse = PipelinePreset.Of(WithMaker(remainder: Remainder.Refuse), Seaborn);

        Assert.Equal(new RemainderChange(Remainder.Drop, Remainder.Keep), keep.TakeOver(WithMaker(), Seaborn).DeclareRemainder);
        Assert.Equal(new RemainderChange(Remainder.Drop, Remainder.Keep), keep.TakeOver(WithMaker(), header: null).DeclareRemainder);
        Assert.Equal(new RemainderChange(Remainder.Drop, Remainder.Refuse), refuse.TakeOver(WithMaker(), Seaborn).DeclareRemainder);
        Assert.Equal(new RemainderChange(Remainder.Refuse, Remainder.Drop), PipelinePreset.Of(WithMaker(), Seaborn).TakeOver(WithMaker(remainder: Remainder.Refuse), Seaborn).DeclareRemainder);
        Assert.Null(keep.TakeOver(WithMaker(remainder: Remainder.Keep), Seaborn).DeclareRemainder);

        // Blocks without a schema said nothing about the rest before.
        Assert.Equal(
            new RemainderChange(null, Remainder.Drop),
            Saved(Blocks()).TakeOver(new PipelineDeclaration([new ReadCsvStep("titanic.csv")]), Header).DeclareRemainder);
    }

    [Fact]
    public void EveryPartOfASavedColumn_TheRemainder_AndEveryParameterOfAnOutput_IsListedWhenItAloneChanges()
    {
        var catalog = StepCatalog.BuiltIn();
        var parts = catalog.Describe("declare").Parameters.OfType<ColumnDeclarationsParameter>().Single().Parts;

        // A part the listing does not compare would be applied unseen: every part is here, the name being the row itself.
        Assert.Equal(["name", "kind", "optional", "excluded", "was"], parts.Select(part => part.Parameter.Key));

        var blocks = WithMaker();
        var category = new PipelineDeclaration(blocks.WithKind("pclass", ColumnKind.Category));
        var categoryWithoutWas = WithMaker(columns => columns.Integer("survived").Category("pclass").Optional("age", ColumnKind.Number).Number("fare"));
        PartCase[] flips =
        [
            new("kind", WithMaker(columns => columns.Integer("survived").Number("pclass").Optional("age", ColumnKind.Number).Number("fare")), blocks, "pclass"),
            new("optional", WithMaker(columns => columns.Integer("survived", "pclass").Number("age").Number("fare")), blocks, "age"),
            new("excluded", new PipelineDeclaration(blocks.Excluding("pclass")), blocks, "pclass"),
            new("was", category, categoryWithoutWas, "pclass"),
        ];

        Assert.Equal(parts.Skip(1).Select(part => part.Parameter.Key), flips.Select(flip => flip.Part));
        Assert.All(flips, flip =>
            Assert.Equal([flip.Column], PipelinePreset.Of(flip.Saved, Seaborn).TakeOver(flip.Into, Seaborn).Changes.Select(change => change.Column)));
        Assert.NotNull(PipelinePreset.Of(WithMaker(remainder: Remainder.Refuse), Seaborn).TakeOver(blocks, Seaborn).DeclareRemainder);

        // Every parameter of every kind of output, changed alone.
        var series = Pdd.Create()
            .ReadCsv("series.csv")
            .Declare(schema => schema.Timestamp("Date").Number("close", "open", "a", "b", "w"))
            .OrderBy("Date")
            .SplitByTime("Date", 0.70, 0.15, 5)
            .Declaration;
        OutputCase[] outputs =
        [
            new(new TargetStep("close"), new TargetStep("open"), "column"),
            new(new DistributionStep(["a", "b"]), new DistributionStep(["a", "w"]), "columns"),
            new(new DistributionStep(["a", "b"]), new DistributionStep(["a", "b"], "w"), "scaleBy"),
            new(new LabelsStep(["a", "b"]), new LabelsStep(["a", "w"]), "columns"),
            new(new LabelsStep(["a", "b"]), new LabelsStep(["a", "b"], 1), "ones"),
            new(new AheadStep("close", 5), new AheadStep("open", 5), "column"),
            new(new AheadStep("close", 5), new AheadStep("close", 3), "ahead"),
            new(new AheadStep("close", 5), new AheadStep("close", 5, AheadAs.Return), "as"),
        ];

        foreach (var verb in catalog.Descriptions.Where(description => catalog.ReadStep(description.Template) is INamesTheAnswer).Select(description => description.Verb))
        {
            Assert.Equal(
                catalog.Describe(verb).Parameters.Select(parameter => parameter.Key).Order(StringComparer.Ordinal),
                outputs.Where(output => output.Before.Verb == verb).Select(output => output.Parameter).Order(StringComparer.Ordinal));
        }

        Assert.All(outputs, output =>
        {
            var into = new PipelineDeclaration(series.WithOutput(output.Before));
            var takenOver = PipelinePreset.Of(new PipelineDeclaration(series.WithOutput(output.After)), header: null).TakeOver(into, header: null);

            Assert.Empty(takenOver.Faults);
            Assert.Equal(new OutputChange(output.Before, output.After), takenOver.Output);
        });
    }

    private readonly record struct TakeOverCase(PipelinePreset Saved, PipelineDeclaration Into, IReadOnlyList<string>? Header);

    private readonly record struct PartCase(string Part, PipelineDeclaration Saved, PipelineDeclaration Into, string Column);

    private readonly record struct OutputCase(INamesTheAnswer Before, INamesTheAnswer After, string Parameter);
}
