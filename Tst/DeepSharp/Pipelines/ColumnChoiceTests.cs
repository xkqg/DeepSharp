// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a column can be made to do, said once for every door that makes it: taking a column in, leaving it out,
/// changing its kind — on the schema alone, or on the whole pipeline, where leaving out a column a step reads is a
/// drop after that step. Each hands back the steps it would make and never judges them: the rules every
/// declaration keeps do that. And each, asked for what already is, hands back the same steps.
/// </summary>
public class ColumnChoiceTests
{
    private static readonly string[] Header =
        ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "class", "who", "adult_male", "deck", "embark_town", "alive", "alone"];

    private static PipelineDeclaration Titanic(Remainder remainder = Remainder.Drop) =>
        Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema.Integer("survived", "pclass").Optional("age", ColumnKind.Number).Number("fare"), remainder)
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Normalise("fare")
            .Declaration;

    private static PipelineDeclaration Then(IReadOnlyList<IPipelineStep> steps) => new(steps);

    private static DeclareStep Schema(IEnumerable<IPipelineStep> steps) => steps.OfType<DeclareStep>().Single();

    private static ColumnDeclaration Declared(IEnumerable<IPipelineStep> steps, string name) =>
        Schema(steps).Columns.Single(column => column.Name == name);

    private static ColumnChoice Row(PipelineDeclaration declaration, string column) =>
        Assert.Single(declaration.ChoicesFor([column]).Rows);

    // ---- the schema's own operations

    [Fact]
    public void TakingInAnExcludedColumn_BringsItBackWithTheKindItHad()
    {
        var schema = new DeclareStep([
            new ColumnDeclaration("a", ColumnKind.Number, false),
            new ColumnDeclaration("b", ColumnKind.Integer, false) { Excluded = true },
        ]);

        var back = schema.WithColumn("b", ColumnKind.Text, ["a", "b"]);

        Assert.False(back.Columns[1].Excluded);
        Assert.Equal(ColumnKind.Integer, back.Columns[1].Kind);
        Assert.Equal(["a", "b"], back.Columns.Select(column => column.Name));
    }

    [Fact]
    public void TakingInAColumnThatTakesPart_ChangesNothing()
    {
        var schema = new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]);

        Assert.Equal(schema, schema.WithColumn("a", ColumnKind.Text, ["a"]));
    }

    [Fact]
    public void ANewColumn_StandsWhereTheSourceHasIt_AndOneTheSourceLacksStandsLast()
    {
        var schema = new DeclareStep([
            new ColumnDeclaration("survived", ColumnKind.Integer, false),
            new ColumnDeclaration("fare", ColumnKind.Number, false),
        ]);

        var withSex = schema.WithColumn("sex", ColumnKind.Category, Header);
        var withExtra = schema.WithColumn("extra", ColumnKind.Number, Header);

        Assert.Equal(["survived", "sex", "fare"], withSex.Columns.Select(column => column.Name));
        Assert.Equal(ColumnKind.Category, withSex.Columns[1].Kind);
        Assert.Equal(["survived", "fare", "extra"], withExtra.Columns.Select(column => column.Name));
    }

    [Fact]
    public void ANewColumnAfterEveryColumnTheSourceHas_StandsLast()
    {
        var schema = new DeclareStep([new ColumnDeclaration("survived", ColumnKind.Integer, false)]);

        Assert.Equal(["survived", "alone"], schema.WithColumn("alone", ColumnKind.Boolean, Header).Columns.Select(column => column.Name));
    }

    [Fact]
    public void ExcludingInTheSchema_KeepsTheKindAndWhatACategoryWas()
    {
        var schema = new DeclareStep([
            new ColumnDeclaration("a", ColumnKind.Number, false),
            new ColumnDeclaration("b", ColumnKind.Category, false) { Was = ColumnKind.Integer },
        ]);

        var excluded = schema.WithColumnExcluded("b");

        Assert.Equal(new ColumnDeclaration("b", ColumnKind.Category, false) { Was = ColumnKind.Integer, Excluded = true }, excluded.Columns[1]);
        Assert.Equal(excluded, excluded.WithColumnExcluded("b"));
        Assert.Equal(schema, schema.WithColumnExcluded("colour"));
    }

    [Fact]
    public void ExcludingTheOnlyColumnASchemaTakes_IsRefused_InTheWordsAFileShows()
    {
        var schema = new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]);

        var refused = Assert.Throws<ArgumentException>(() => schema.WithColumnExcluded("a"));

        Assert.Equal("The schema excludes every column it names and keeps none of the rest, so no column would take part.", refused.Message);
    }

    [Fact]
    public void AKindChange_RemembersTheKindACategoryCameFrom_AndGoesBackToIt()
    {
        var schema = new DeclareStep([new ColumnDeclaration("pclass", ColumnKind.Integer, false)]);

        var category = schema.WithColumnKind("pclass", ColumnKind.Category);
        var back = category.WithColumnKind("pclass", ColumnKind.Integer);

        Assert.Equal(ColumnKind.Category, category.Columns[0].Kind);
        Assert.Equal(ColumnKind.Integer, category.Columns[0].Was);
        Assert.Equal(schema, back);
    }

    [Fact]
    public void TheSameKindAgain_ChangesNothing_AndACategoryNeverWasACategory()
    {
        var category = new DeclareStep([new ColumnDeclaration("pclass", ColumnKind.Category, false) { Was = ColumnKind.Integer }]);

        Assert.Equal(category, category.WithColumnKind("pclass", ColumnKind.Category));
    }

    [Fact]
    public void AnyOtherKind_ForgetsWhatACategoryWas()
    {
        var category = new DeclareStep([new ColumnDeclaration("pclass", ColumnKind.Category, false) { Was = ColumnKind.Integer }]);

        var number = category.WithColumnKind("pclass", ColumnKind.Number);

        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Number, false), number.Columns[0]);
        Assert.Null(new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]).WithColumnKind("a", ColumnKind.Text).Columns[0].Was);
    }

    [Fact]
    public void AKindForAColumnTheSchemaDoesNotName_IsRefused_SinceTakingItInIsAnotherOperation()
    {
        var schema = new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]);

        var refused = Assert.Throws<ArgumentException>(() => schema.WithColumnKind("sex", ColumnKind.Category));

        Assert.Contains("'sex'", refused.Message, StringComparison.Ordinal);
    }

    // ---- the pipeline's operations

    [Theory]
    [InlineData(Remainder.Drop)]
    [InlineData(Remainder.Keep)]
    public void ExcludingAColumnNoStepReads_ExcludesItInTheSchema_WithItsKind(Remainder remainder)
    {
        var steps = Titanic(remainder).Excluding("pclass");

        Assert.True(Declared(steps, "pclass").Excluded);
        Assert.Equal(ColumnKind.Integer, Declared(steps, "pclass").Kind);
        Assert.Equal(5, steps.Count);
        Assert.Empty(PipelineDeclaration.FaultsIn(steps));
    }

    [Fact]
    public void ExcludingAColumnAStepReads_IsADropAfterTheLastStepThatReadsIt()
    {
        var steps = Titanic().Excluding("age");

        Assert.Equal(["read.csv", "declare", "split.stratified", "fill.missing", "drop.columns", "normalise"], steps.Select(step => step.Verb));
        Assert.Equal(["age"], ((DropColumnsStep)steps[4]).Columns);
        Assert.Empty(PipelineDeclaration.FaultsIn(steps));
    }

    [Fact]
    public void ExcludingAColumnAStepMade_IsADropAfterIt_AndAnotherNameInADropStandingThere()
    {
        var steps = Then(Then(Titanic().Excluding("age_was_missing")).Excluding("age")).Steps;

        Assert.Equal(["read.csv", "declare", "split.stratified", "fill.missing", "drop.columns", "normalise"], steps.Select(step => step.Verb));
        Assert.Equal(["age_was_missing", "age"], ((DropColumnsStep)steps[4]).Columns);
    }

    [Fact]
    public void ExcludingAColumnTheRestKeeps_IsADropAfterTheSchema()
    {
        var steps = Titanic(Remainder.Keep).Excluding("sex");

        Assert.Equal(["read.csv", "declare", "drop.columns", "split.stratified", "fill.missing", "normalise"], steps.Select(step => step.Verb));
        Assert.Equal(["sex"], ((DropColumnsStep)steps[2]).Columns);
        Assert.Empty(PipelineDeclaration.FaultsIn(steps));
    }

    [Fact]
    public void ExcludingAColumnThatNoLongerReachesTheEnd_ChangesNothing()
    {
        var dropped = Then(Titanic().Excluding("age"));

        Assert.Equal(dropped.Steps, dropped.Excluding("age"));
        Assert.Equal(Titanic().Steps, Titanic().Excluding("sex"));
    }

    [Fact]
    public void LeavingOutTheOnlyColumnTheSchemaTakes_IsRefusedAtTheSchema_AndNotOffered()
    {
        // A schema takes a column unless it keeps the rest, and refuses to be made into one that takes none.
        var one = Pdd.Create().ReadCsv("titanic.csv").Declare(schema => schema.Integer("survived")).Declaration;

        var refused = Assert.Throws<DeclarationException>(() => one.Excluding("survived"));

        Assert.Equal(
            [new DeclarationFault(1, "declare", "The schema excludes every column it names and keeps none of the rest, so no column would take part.")],
            refused.Faults);
        Assert.False(Row(one, "survived").Offers.HasFlag(ColumnOffers.Exclude));
    }

    [Fact]
    public void ExcludingTheAnswer_GivesStepsTheRulesRefuse()
    {
        var answered = Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema.Integer("survived", "pclass"))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Declaration;

        Assert.NotEmpty(PipelineDeclaration.FaultsIn(answered.Excluding("survived")));
    }

    [Fact]
    public void TakingInAnExcludedColumn_BringsItBackAsItWas_WhateverKindIsAsked()
    {
        var excluded = Then(Titanic().Excluding("pclass"));

        var steps = excluded.Including("pclass", ColumnKind.Text, Header);

        Assert.Equal(Titanic().Steps, steps);
    }

    [Fact]
    public void TakingInADroppedColumn_TakesItsNameOutOfTheDrop_AndADropLeftEmptyGoes()
    {
        var both = Then(Then(Titanic().Excluding("age_was_missing")).Excluding("age"));

        var ageBack = Then(both.Including("age", ColumnKind.Text, Header));
        var allBack = ageBack.Including("age_was_missing", ColumnKind.Text, Header);

        Assert.Equal(["age_was_missing"], ageBack.Steps.OfType<DropColumnsStep>().Single().Columns);
        Assert.Equal(Titanic().Steps, allBack);
    }

    [Fact]
    public void TakingInAColumnTheSchemaDoesNotName_TakesItWithTheKindAsked_WhereTheSourceHasIt()
    {
        var steps = Titanic().Including("sex", ColumnKind.Category, Header);

        Assert.Equal(["survived", "pclass", "sex", "age", "fare"], Schema(steps).Columns.Select(column => column.Name));
        Assert.Equal(ColumnKind.Category, Declared(steps, "sex").Kind);
        Assert.Empty(PipelineDeclaration.FaultsIn(steps));
    }

    [Fact]
    public void TakingInAColumnThatTakesPart_OrOneAStepReplaced_ChangesNothing()
    {
        Assert.Equal(Titanic().Steps, Titanic().Including("fare", ColumnKind.Text, Header));

        var encoded = Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema.Integer("survived").Category("sex"))
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories()
            .Declaration;

        Assert.Equal(encoded.Steps, encoded.Including("sex", ColumnKind.Text, Header));
    }

    [Fact]
    public void AKindChange_IsTheSchemasOwn_AndTheSameKindAgainChangesNothing()
    {
        var category = Then(Titanic().WithKind("pclass", ColumnKind.Category));

        Assert.Equal(ColumnKind.Category, Declared(category.Steps, "pclass").Kind);
        Assert.Equal(ColumnKind.Integer, Declared(category.Steps, "pclass").Was);
        Assert.Equal(category.Steps, category.WithKind("pclass", ColumnKind.Category));
    }

    [Fact]
    public void APipelineWithoutASchema_ChangesNothing_AndOffersNothing()
    {
        var unread = new PipelineDeclaration([new ReadCsvStep("titanic.csv")]);

        Assert.Equal(unread.Steps, unread.Including("sex", ColumnKind.Text, Header));
        Assert.Equal(unread.Steps, unread.Excluding("sex"));
        Assert.Equal(unread.Steps, unread.WithKind("sex", ColumnKind.Category));
        Assert.Equal(new ColumnChoice("sex", ColumnStanding.NotDeclared, null, null, ColumnOffers.None), Row(unread, "sex"));
    }

    // ---- how each column stands, and what it offers

    [Fact]
    public void EveryColumnAsked_SaysHowItStands_InTheOrderItWasAsked()
    {
        var pipeline = Then(Then(Titanic().Excluding("pclass")).Excluding("age"));

        var rows = pipeline.ChoicesFor(["fare", "pclass", "sex", "age", "age_was_missing", "survived"]).Rows;

        Assert.Equal(["fare", "pclass", "sex", "age", "age_was_missing", "survived"], rows.Select(row => row.Name));
        Assert.Equal(
            [ColumnStanding.Taking, ColumnStanding.Excluded, ColumnStanding.NotDeclared, ColumnStanding.Dropped, ColumnStanding.Made, ColumnStanding.Taking],
            rows.Select(row => row.Standing));
        Assert.Equal([ColumnKind.Number, ColumnKind.Integer, null, ColumnKind.Number], rows.Take(4).Select(row => row.Kind));
        Assert.NotNull(rows[4].Kind);
        Assert.Equal(ColumnStanding.Kept, Row(Titanic(Remainder.Keep), "sex").Standing);
    }

    [Fact]
    public void AColumnThatTakesPart_OffersToBeExcluded_AndMadeACategory()
    {
        var pclass = Row(Titanic(), "pclass");

        Assert.Equal(ColumnOffers.Exclude | ColumnOffers.MakeCategory, pclass.Offers);
    }

    [Fact]
    public void AColumnAStepScalesAsANumber_OffersNoCategory()
    {
        Assert.Equal(ColumnOffers.Exclude, Row(Titanic(), "fare").Offers);
    }

    [Fact]
    public void AColumnThatIsOut_OffersToBeTakenIn()
    {
        var pipeline = Then(Then(Titanic().Excluding("pclass")).Excluding("age"));

        Assert.Equal(ColumnOffers.Include | ColumnOffers.MakeCategory, Row(pipeline, "pclass").Offers);
        Assert.True(Row(pipeline, "age").Offers.HasFlag(ColumnOffers.Include));
        Assert.Equal(ColumnOffers.Include, Row(Titanic(), "sex").Offers);
    }

    [Fact]
    public void TheAnswer_OffersNoWayToBeExcluded()
    {
        var answered = Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema.Integer("survived", "pclass"))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Declaration;

        Assert.False(Row(answered, "survived").Offers.HasFlag(ColumnOffers.Exclude));
    }

    [Fact]
    public void ACategory_OffersTheWayBack_OnlyWhenItSaysWhatItWas()
    {
        var madeOne = Then(Titanic().WithKind("pclass", ColumnKind.Category));
        var bornOne = Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema.Integer("survived").Category("sex"))
            .SplitStratified("survived", 0.70, 0.15)
            .Declaration;

        Assert.Equal(ColumnOffers.Exclude | ColumnOffers.BackToWas, Row(madeOne, "pclass").Offers);
        Assert.Equal(ColumnKind.Integer, Row(madeOne, "pclass").Was);
        Assert.False(Row(bornOne, "sex").Offers.HasFlag(ColumnOffers.BackToWas));
    }
}
