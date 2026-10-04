// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What should not be there, found on the training rows and never in the file: a column that hands a model the answer,
/// one that says again what another says, one that only names its row, and a value that is how the file writes that
/// nothing is known. Each is said with the columns it is about and how it is answered — a step the column rules keep
/// for the column's kind, the column left out, or a value the schema says stands for a gap.
/// </summary>
public class AlertTests
{
    [Fact]
    public void TheGapsOfANumberColumn_AreAnsweredByBothVerbs_EachWhereItStands()
    {
        // Two verbs answer a gap and they stand in different halves of the pipeline, so the one sentence says which is
        // which: settling puts in a value no row decided and may stand where the features are worked out, filling learns
        // its value from the training rows and stands below the split.
        var alert = Assert.Single(Passengers().Alerts, each => each.Column == "age" && each.Says.Contains("rows are gaps", StringComparison.Ordinal));

        Assert.Contains("settle.gaps", alert.Says, StringComparison.Ordinal);
        Assert.Contains("fill.missing", alert.Says, StringComparison.Ordinal);
        Assert.Contains("above", alert.Says, StringComparison.Ordinal);
        Assert.Contains("below", alert.Says, StringComparison.Ordinal);
    }

    private static string Titanic => Repository.Data("titanic.csv");

    // Every column of the published file, as the proposal of its kinds proposes it, profiled above a split that trains on
    // 623 rows, with survived as the answer.
    private static DataProfile Passengers()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema
                .Integer("survived", "pclass", "sibsp", "parch")
                .Category("sex", "embarked", "class", "who", "deck", "embark_town")
                .Optional("age", ColumnKind.Number)
                .Number("fare")
                .Boolean("adult_male", "alone", "alive"))
            .Profile()
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Build()
            .Run();

        return (DataProfile)prepared.Evidence[2];
    }

    private static DataProfile Profiled(string csv, Action<SchemaBuilder> schema, Func<PipelineBuilder, PipelineBuilder>? more = null)
    {
        var chain = Pdd.Create().Read(CsvRowSource.FromText(csv), "rows").Declare(schema);

        return (DataProfile)(more?.Invoke(chain) ?? chain).Profile().Build().Run().Evidence[(more is null ? 2 : 3)];
    }

    private static string Rows(params object[] values) => string.Join('\n', values) + '\n';

    [Fact]
    public void AColumnThatRestatesTheAnswer_IsFound_WithEveryValueAndTheAnswerItGoesWith_AndAnsweredByLeavingItOut()
    {
        var alive = Assert.Single(Passengers().Alerts, alert => alert.Column == "alive" && alert.Answer.Action == AlertAction.LeaveOut);

        Assert.Equal(["alive", "survived"], alive.Columns);
        Assert.Equal(AlertAnswer.LeaveOut("alive"), alive.Answer);
        Assert.Contains("False with 0 in 384, True with 1 in 239", alive.Says, StringComparison.Ordinal);
        Assert.Contains("the answer", alive.Says, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnThatSaysAgainWhatAnotherSays_IsFound_OnceAgainstTheFirstColumnItRepeats()
    {
        var alerts = Passengers().Alerts;
        var @class = Assert.Single(alerts, alert => alert.Column == "class" && alert.Answer.Action == AlertAction.LeaveOut);
        var town = Assert.Single(alerts, alert => alert.Column == "embark_town" && alert.Answer.Action == AlertAction.LeaveOut);

        Assert.Equal(["class", "pclass"], @class.Columns);
        Assert.Contains("First with 1 in 146, Second with 2 in 123, Third with 3 in 354", @class.Says, StringComparison.Ordinal);
        Assert.Equal(["embark_town", "embarked"], town.Columns);

        // Nothing else says again what another says: who goes with adult_male but not the other way round.
        Assert.Equal(
            ["alive", "class", "embark_town"],
            alerts.Where(alert => alert.Columns.Count == 2).Select(alert => alert.Column).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void LeavingOutACategoryTheStepBelowEncodes_LeavesItOutInTheSchema_WithItsKind()
    {
        // The step that encodes every category names none, and the category never reaches the end itself, so the
        // answer leaves it out where the schema takes it in.
        var @class = Assert.Single(Passengers().Alerts, alert => alert.Column == "class" && alert.Answer.Action == AlertAction.LeaveOut);
        var encoded = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Integer("survived", "pclass").Category("sex", "class").Number("fare"))
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories()
            .Target("survived")
            .Declaration;

        var answered = @class.Answer.AppliedTo(encoded);

        Assert.Equal(
            new ColumnDeclaration("class", ColumnKind.Category, Optional: false) { Excluded = true },
            answered.OfType<DeclareStep>().Single().Columns.Single(column => column.Name == "class"));
        Assert.Empty(PipelineDeclaration.FaultsIn(answered));
    }

    [Fact]
    public void AValueThatIsHowTheFileWritesNothingIsKnown_IsFound_AndAnsweredBySayingSoInTheSchema()
    {
        var fare = Assert.Single(Passengers().Alerts, alert => alert.Answer.Action == AlertAction.SayMissing);

        Assert.Equal(["fare"], fare.Columns);
        Assert.Equal(AlertAnswer.SayMissing("fare", "0"), fare.Answer);
        Assert.Contains("12 of 623 rows hold 0", fare.Says, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryStepAnAlertNames_IsOneTheColumnRulesKeepForTheColumnsKind()
    {
        var profile = Passengers();
        var catalog = StepCatalog.BuiltIn();

        foreach (var alert in profile.Alerts.Where(alert => alert.Answer.Action == AlertAction.Step))
        {
            var kind = profile.Columns.Single(column => column.Name == alert.Answer.Column).Kind;
            var column = catalog.Describe(alert.Answer.Verb!).Parameters.OfType<ColumnParameter>().SingleOrDefault()
                         ?? (StepParameter?)catalog.Describe(alert.Answer.Verb!).Parameters.OfType<ColumnsParameter>().SingleOrDefault();

            Assert.True(
                column switch
                {
                    ColumnParameter one => one.Accepts.Contains(kind),
                    ColumnsParameter many => many.Accepts.Contains(kind),
                    _ => kind == ColumnKind.Category,
                },
                $"{alert.Answer.Verb} on {alert.Column}, which holds {kind}");
        }

        // A gap among words is answered by the encoder, which makes it no category and marks it; among numbers by a fill.
        Assert.Contains(profile.Alerts, alert => alert.Answer == AlertAnswer.Step("encode.categories", "deck") && alert.Says.Contains("are gaps.", StringComparison.Ordinal));
        Assert.Contains(profile.Alerts, alert => alert.Answer == AlertAnswer.Step("fill.missing", "age"));
    }

    [Theory]
    [InlineData(ColumnKind.Boolean, "true")]
    [InlineData(ColumnKind.Timestamp, "2015-02-18")]
    public void AGapInAColumnNoFillTakes_IsAnsweredByDroppingTheRowsItIsIn(ColumnKind kind, string value)
    {
        var profile = Profiled(Rows("x", value, "", value), schema => schema.Column(new ColumnDeclaration("x", kind, Optional: false)));

        Assert.Contains(profile.Alerts, alert => alert.Answer == AlertAnswer.Step("drop.gaps", "x"));
    }

    [Fact]
    public void ThePriceSeries_RaisesNoIdentifierAlert_ForMomentsOrForAVolumeThatIsNoRunningNumber()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close").Integer("AAPL.Volume").Category("direction"))
            .OrderBy("Date")
            .Profile()
            .SplitByTime("Date", 0.70, 0.15)
            .Build()
            .Run();

        var profile = (DataProfile)prepared.Evidence[3];

        Assert.Equal(354, profile.Columns.Single(column => column.Name == "AAPL.Volume").Distinct);
        Assert.DoesNotContain(profile.Alerts, alert => alert.Answer.Action == AlertAction.LeaveOut);
    }

    [Fact]
    public void AColumnThatOnlyNamesItsRow_IsFound_ARunningNumberOrWordsNoTwoRowsShare()
    {
        var profile = Profiled(
            Rows("id,name,n", "1,Ann,5", "2,Bob,5", "3,Cyd,6", "4,Dee,6"),
            schema => schema.Integer("id", "n").Text("name"));

        Assert.Contains(profile.Alerts, alert => alert.Answer == AlertAnswer.LeaveOut("id"));
        Assert.Contains(profile.Alerts, alert => alert.Answer == AlertAnswer.LeaveOut("name"));
        Assert.DoesNotContain(profile.Alerts, alert => alert.Answer == AlertAnswer.LeaveOut("n"));
    }

    [Fact]
    public void AColumnTheRowsArePutInOrderBy_IsNeverAnIdentifier()
    {
        var profile = Profiled(
            Rows("id,v", "1,5", "2,5", "3,6", "4,6"),
            schema => schema.Integer("id", "v"),
            chain => chain.OrderBy("id"));

        Assert.DoesNotContain(profile.Alerts, alert => alert.Answer.Action == AlertAction.LeaveOut && alert.Column == "id");
    }

    [Theory]
    [InlineData("-999", "smallest")]
    [InlineData("9999", "largest")]
    [InlineData("-1", "smallest")]
    public void ANumberFarFromTheRestWrittenAsFilesWriteNothing_IsFoundAtEitherEnd(string value, string end)
    {
        var profile = Profiled(Rows("x", value, value, "10.5", "11", "11.25", "12", "12.5", "13"), schema => schema.Number("x"));

        var alert = Assert.Single(profile.Alerts, alert => alert.Answer.Action == AlertAction.SayMissing);

        Assert.Equal(AlertAnswer.SayMissing("x", value), alert.Answer);
        Assert.Contains($"the {end} value", alert.Says, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("x\n0\n0\n1\n1\n2\n3\n")]
    [InlineData("x\n512.3292\n512.3292\n10\n11\n12\n13\n")]
    [InlineData("x\n0\n10\n11\n12\n13\n")]
    [InlineData("x\n0\n0\n5\n")]
    public void ACountStartingAtNought_ARepeatedExtreme_AndAValueHeldOnce_AreNotHowAFileWritesNothing(string csv)
    {
        var profile = Profiled(csv, schema => schema.Number("x"));

        Assert.DoesNotContain(profile.Alerts, alert => alert.Answer.Action == AlertAction.SayMissing);
    }

    [Fact]
    public void AValueTheSchemaAlreadySaysStandsForAGap_IsNotFoundAgain()
    {
        var profile = Profiled(Rows("x", "-999", "-999", "10.5", "11", "11.25", "12"), schema => schema.Column(new ColumnDeclaration("x", ColumnKind.Number, Optional: false) { Missing = "-999" }));

        Assert.DoesNotContain(profile.Alerts, alert => alert.Answer.Action == AlertAction.SayMissing);
    }

    [Fact]
    public void AnAnswerTheRowsDoNotHoldWhereTheProfileStands_IsNotComparedWith()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .OrderBy("Date")
            .Profile()
            .SplitByTime("Date", 0.70, 0.15, 5)
            .Ahead("AAPL.Close", 5)
            .Build()
            .Run();

        Assert.DoesNotContain(((DataProfile)prepared.Evidence[3]).Alerts, alert => alert.Columns.Count == 2);
    }

    [Fact]
    public void AnAnswerWhoseValuesNeverRepeat_IsComparedWithNothing()
    {
        // Two columns whose values never repeat go with each other value for value by chance alone.
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Open", "AAPL.Close").Category("direction"))
            .Profile()
            .SplitAtRandom(0.70, 0.15)
            .Target("AAPL.Close")
            .Build()
            .Run();

        Assert.DoesNotContain(((DataProfile)prepared.Evidence[2]).Alerts, alert => alert.Columns.Count == 2);
    }

    [Fact]
    public void AColumnGoingWithAnotherInManyValues_NamesTheFirstFew_AndCountsTheRest()
    {
        var letters = "abcdefgh";
        var rows = Enumerable.Range(0, 16).Select(row => $"{row % 8},{letters[row % 8]}");

        var profile = Profiled(Rows(["n,w", .. rows]), schema => schema.Integer("n").Text("w"));
        var repeats = Assert.Single(profile.Alerts, alert => alert.Columns.Count == 2);

        Assert.Equal(["w", "n"], repeats.Columns);
        Assert.Contains("a with 0 in 2, b with 1 in 2, c with 2 in 2, d with 3 in 2, e with 4 in 2, and 3 more", repeats.Says, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAlertsAnswer_AppliedToThePipeline_LeavesTheColumnOut_OrSaysTheValueOnTheSchema_OrLeavesAStepToBeWritten()
    {
        var declaration = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Integer("survived").Number("fare").Boolean("alive"))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Declaration;

        var left = new PipelineDeclaration(AlertAnswer.LeaveOut("alive").AppliedTo(declaration));
        var said = new PipelineDeclaration(AlertAnswer.SayMissing("fare", "0").AppliedTo(declaration));

        Assert.True(((DeclareStep)left.Steps[1]).Columns.Single(column => column.Name == "alive").Excluded);
        Assert.Equal("0", ((DeclareStep)said.Steps[1]).Columns.Single(column => column.Name == "fare").Missing);
        Assert.Same(declaration.Steps, AlertAnswer.Step("fill.missing", "fare").AppliedTo(declaration));
        Assert.Throws<ArgumentNullException>(() => AlertAnswer.LeaveOut("alive").AppliedTo(null!));

        // Without a schema there is nothing to say a value on.
        var unread = new PipelineDeclaration([new ReadCsvStep("fares.csv")]);

        Assert.Same(unread.Steps, AlertAnswer.SayMissing("fare", "0").AppliedTo(unread));
        Assert.Throws<ArgumentNullException>(() => ((PipelineDeclaration)null!).WithMissing("fare", "0"));

        // A column the schema does not name has no value of its own to say, and the refusal is in the words a file shows.
        var nowhere = Assert.Single(Assert.Throws<DeclarationException>(() => AlertAnswer.SayMissing("nowhere", "0").AppliedTo(declaration)).Faults);

        Assert.Equal(1, nowhere.At);
        Assert.Contains("'nowhere'", nowhere.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("(Parameter", nowhere.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoAlertsThatSayTheSameThing_AreEqual()
    {
        var one = new ProfileAlert(["a", "b"], "says", AlertAnswer.LeaveOut("a"));

        Assert.Equal(one, new ProfileAlert(["a", "b"], "says", AlertAnswer.LeaveOut("a")));
        Assert.Equal(one.GetHashCode(), new ProfileAlert(["a", "b"], "says", AlertAnswer.LeaveOut("a")).GetHashCode());
        Assert.NotEqual(one, one with { Columns = ["a"] });
        Assert.NotEqual(one, one with { Says = "other" });
        Assert.Equal("a", one.Column);
    }
}
