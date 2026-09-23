// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The edges: a file that is not shaped the way a file should be, a table asked to hold columns of
/// different lengths, a fit with nothing to learn from, and every new verb written out and read back. None
/// of these is exotic — each is what a real file does on the day it changes.
/// </summary>
public class RefusalsAndEdgesTests
{
    // ---- reading -------------------------------------------------------------------------------------

    [Fact]
    public void RowsAlreadyInHand_AreASourceLikeAnyOther()
    {
        var source = new InMemoryRowSource(["a", "b"], [["1", "2"], ["3", null]]);

        Assert.Equal(["a", "b"], source.ColumnNames);
        Assert.Equal(2, source.Rows.Count());
    }

    [Fact]
    public void AQuotedFieldMayHoldCommasLineBreaksAndQuotes()
    {
        var source = CsvRowSource.FromText("name,note\n\"Smith, John\",\"he said \"\"yes\"\"\"\n");

        var row = source.Rows.Single();

        Assert.Equal("Smith, John", row[0]);
        Assert.Equal("he said \"yes\"", row[1]);
    }

    [Fact]
    public void AFileWithoutATrailingLineBreak_StillHasThatLastRow()
    {
        Assert.Equal(2, CsvRowSource.FromText("a\n1\n2").Rows.Count());
    }

    [Fact]
    public void AFileEndingInALineBreak_HasNoEmptyRowAfterIt()
    {
        Assert.Equal(2, CsvRowSource.FromText("a\r\n1\r\n2\r\n").Rows.Count());
    }

    [Fact]
    public void AFileWithNoHeaderAtAll_IsRefused()
    {
        Assert.Throws<FormatException>(() => CsvRowSource.FromText(string.Empty));
    }

    [Fact]
    public void ARowWithTheWrongNumberOfCells_IsRefusedWithItsLineNumber()
    {
        var refused = Assert.Throws<FormatException>(() => CsvRowSource.FromText("a,b\n1,2\n3\n"));

        Assert.Contains("line 3", refused.Message);
    }

    [Fact]
    public void AFileThatIsNotThere_SaysSo()
    {
        Assert.Throws<FileNotFoundException>(() => new CsvRowSource("no-such-file-anywhere.csv"));
    }

    [Fact]
    public void ACellMissingBecauseTheRowIsShort_IsAGap()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("b", ColumnKind.Number, true)]),
            new InMemoryRowSource(["a", "b"], [["1"]]));

        Assert.True(table["b"].IsMissing(0));
    }

    // ---- the table -----------------------------------------------------------------------------------

    [Fact]
    public void ColumnsOfDifferentLengths_AreNotATable()
    {
        var refused = Assert.Throws<ArgumentException>(() => new Table([
            new Column<double>("a", ColumnKind.Number, [1.0, 2.0]),
            new Column<double>("b", ColumnKind.Number, [1.0]),
        ]));

        Assert.Contains("different lengths", refused.Message);
    }

    [Fact]
    public void TwoColumnsWithOneName_AreNotATableEither()
    {
        Assert.Throws<ArgumentException>(() => new Table([
            new Column<double>("a", ColumnKind.Number, [1.0]),
            new Column<double>("a", ColumnKind.Number, [2.0]),
        ]));
    }

    [Fact]
    public void AColumnThatIsNotThere_SaysWhatIs()
    {
        var table = new Table([new Column<double>("a", ColumnKind.Number, [1.0])]);

        var refused = Assert.Throws<KeyNotFoundException>(() => table["b"]);

        Assert.Contains("'b'", refused.Message);
        Assert.Contains("a", refused.Message);
    }

    [Fact]
    public void AColumnOfTheWrongLength_CannotBePutOnATable()
    {
        var table = new Table([new Column<double>("a", ColumnKind.Number, [1.0, 2.0])]);

        Assert.Throws<ArgumentException>(
            () => table.Put(new Column<double>("b", ColumnKind.Number, [1.0])));
    }

    [Fact]
    public void PuttingAColumnThatIsAlreadyThere_ReplacesIt()
    {
        var table = new Table([new Column<double>("a", ColumnKind.Number, [1.0])]);

        table.Put(new Column<double>("a", ColumnKind.Number, [9.0]));

        Assert.Single(table.Columns);
        Assert.Equal(9, ((Column<double>)table["a"])[0]);
        Assert.True(table.Remove("a"));
        Assert.False(table.Remove("a"));
        Assert.Empty(table.Columns);
    }

    [Fact]
    public void AnEmptyTable_HasNoRows()
    {
        Assert.Equal(0, new Table([]).RowCount);
    }

    [Fact]
    public void AColumnWithoutAName_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new Column<double>(" ", ColumnKind.Number, [1.0]));
        Assert.Throws<ArgumentException>(() => new TextColumn(" ", ["a"]));
    }

    [Fact]
    public void AColumnSaysWhatItHolds_AsTextAndAsAGap()
    {
        var numbers = new Column<double>("a", ColumnKind.Number, [1.5, null]);
        var words = new TextColumn("b", ["yes", null]);

        Assert.Equal("1.5", numbers.TextAt(0));
        Assert.Null(numbers.TextAt(1));
        Assert.Equal("yes", words.TextAt(0));
        Assert.Null(words.TextAt(1));
    }

    // ---- the shares ----------------------------------------------------------------------------------

    [Theory]
    [InlineData(0.0, 0.5, 0.5)]
    [InlineData(0.5, 0.5, 0.0)]
    [InlineData(double.NaN, 0.5, 0.5)]
    public void ASharesThatIsNotAShare_IsRefused(double train, double validation, double test)
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new SplitAtRandomStep(new SplitShares(train, validation, test), 1));
    }

    [Fact]
    public void ButValidationMayBeNothingAtAll()
    {
        // A two-way division into training and test is an ordinary way to work; a part that is simply not
        // there is different from one that was meant to exist and came out empty.
        var shares = new SplitShares(0.5, 0, 0.5);

        shares.Validate();

        Assert.Equal(0, shares.Validation);
    }

    [Fact]
    public void SharesThatDoNotMakeAWhole_AreRefused()
    {
        Assert.Throws<ArgumentException>(
            () => new SplitStratifiedStep("a", new SplitShares(0.70, 0.15, 0.10), 1));
    }

    [Fact]
    public void AStratifiedSplitWithoutAColumn_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new SplitStratifiedStep(" ", new SplitShares(0.8, 0.1, 0.1), 1));
    }

    [Fact]
    public void SharesOverVeryFewRows_StillUseEveryRowExactlyOnce()
    {
        var over = new SplitShares(0.70, 0.15, 0.15).Over(3);

        Assert.Equal(3, over.Length);
        Assert.Equal(2, over.Count(split => split == Part.Train));
    }

    [Fact]
    public void SharesOverOneRow_PutItInTraining()
    {
        Assert.Equal([Part.Train], new SplitShares(0.70, 0.15, 0.15).Over(1));
    }

    // ---- splitting -----------------------------------------------------------------------------------

    [Fact]
    public void ASplitInTimeReadsAWholeNumberOrAPlainNumberAsTime()
    {
        foreach (var kind in new[] { ColumnKind.Integer, ColumnKind.Number })
        {
            var table = SchemaBinding.Bind(
                new DeclareStep([new ColumnDeclaration("t", kind, false)]),
                CsvRowSource.FromText("t\n3\n1\n2\n"));

            var parts = new SplitByTimeStep("t", new SplitShares(0.34, 0.33, 0.33)).Assign(table);

            Assert.Equal(Part.Train, parts[1]);
            Assert.Equal(Part.Test, parts[0]);
        }
    }

    [Fact]
    public void ASplitInTimeOnWordsIsRefused()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("t", ColumnKind.Text, false)]),
            CsvRowSource.FromText("t\na\nb\nc\n"));

        var refused = Assert.Throws<InvalidOperationException>(
            () => new SplitByTimeStep("t", new SplitShares(0.34, 0.33, 0.33)).Assign(table));

        Assert.Contains("no order to put rows in", refused.Message);
    }

    [Fact]
    public void AStratifiedSplitCountsAGapAsAGroupOfItsOwn()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("g", ColumnKind.Text, true)]),
            CsvRowSource.FromText("g\na\na\n\n\n"));

        var parts = new SplitStratifiedStep("g", new SplitShares(0.5, 0.25, 0.25), 1).Assign(table);

        Assert.Equal(4, parts.Length);
    }

    // ---- fitting -------------------------------------------------------------------------------------

    [Fact]
    public void AFitWithNothingToLearnFrom_SaysSoRatherThanInventingANumber()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, true)]),
            CsvRowSource.FromText("a\n\n\n"));

        var parts = new[] { Part.Train, Part.Train };

        Assert.Throws<InvalidOperationException>(() => FillMissingStep.Of("a", With.Mean).Fit(table, parts));
        Assert.Throws<InvalidOperationException>(() => new NormaliseStep("a").Fit(table, parts));
        Assert.Throws<InvalidOperationException>(() => new EncodeStep("a").Fit(table, parts));
    }

    [Fact]
    public void AGapWithNothingBeforeIt_CannotCarryAnythingForward()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, true)]),
            CsvRowSource.FromText("a\n\n2\n"));

        var step = FillMissingStep.Of("a", With.Previous);
        var learned = step.Fit(table, [Part.Train, Part.Train]);

        Assert.Throws<InvalidOperationException>(() => step.ApplyTo(table, learned));
    }

    [Fact]
    public void AGapInWordsIsNotFilledWithANumber()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Text, true)]),
            CsvRowSource.FromText("a\nx\n\n"));

        var step = FillMissingStep.Of("a", With.Mean);

        Assert.Throws<InvalidOperationException>(() => step.Fit(table, [Part.Train, Part.Train]));
    }

    [Fact]
    public void AWholeNumberColumn_IsFilledWithAWholeNumber()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Integer, true)]),
            CsvRowSource.FromText("a\n1\n2\n\n"));

        var step = FillMissingStep.Of("a", With.Mean);
        step.ApplyTo(table, step.Fit(table, [Part.Train, Part.Train, Part.Test]));

        Assert.Equal(2, ((Column<long>)table["a"])[2]);
    }

    [Fact]
    public void WhatAFitNeverLearned_CannotBeReplayed()
    {
        var values = new FittedStepValues();

        Assert.Throws<InvalidOperationException>(() => values.Number("value"));
        Assert.Throws<InvalidOperationException>(() => values.List("categories"));
    }

    [Fact]
    public void WhatAFitLearned_WritesItselfOut()
    {
        var values = new FittedStepValues();
        values.Learned("centre", 1.5);
        values.Learned("categories", ["a", "b"]);

        var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            values.WriteTo(writer);
        }

        var written = System.Text.Encoding.UTF8.GetString(buffer.ToArray());

        Assert.Contains("\"centre\":1.5", written, StringComparison.Ordinal);
        Assert.Contains("\"categories\":[\"a\",\"b\"]", written, StringComparison.Ordinal);
    }

    // ---- features and transforms ---------------------------------------------------------------------

    [Fact]
    public void ACycleNeedsAMomentInTime()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]),
            CsvRowSource.FromText("a\n1\n"));

        Assert.Throws<InvalidOperationException>(() => new CyclicalStep("a", Period.HourOfDay).AddTo(table));
    }

    [Theory]
    [InlineData(Period.HourOfDay)]
    [InlineData(Period.DayOfWeek)]
    [InlineData(Period.DayOfMonth)]
    [InlineData(Period.MonthOfYear)]
    public void EveryCycleHasALengthAndAPlace(Period period)
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("t", ColumnKind.Timestamp, true)]),
            CsvRowSource.FromText("t\n2026-09-23T14:30:00\n\n"));

        new CyclicalStep("t", period).AddTo(table);

        var stem = $"t_{period.ToString().ToLowerInvariant()}";

        Assert.InRange(((Column<double>)table[$"{stem}_sin"])[0]!.Value, -1, 1);
        Assert.True(table[$"{stem}_cos"].IsMissing(1));
    }

    [Theory]
    [InlineData(Form.Signed)]
    [InlineData(Form.Unit)]
    [InlineData(Form.SplitSign)]
    public void EveryFormLeavesAGapAsAGap(Form form)
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("t", ColumnKind.Timestamp, true)]),
            CsvRowSource.FromText("t\n\n"));

        new CyclicalStep("t", Period.HourOfDay, form).AddTo(table);

        var name = form == Form.SplitSign ? "t_hourofday_sin_pos" : "t_hourofday_sin";

        Assert.True(table[name].IsMissing(0));
    }

    [Fact]
    public void AFeatureOnWordsIsRefused()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Text, false)]),
            CsvRowSource.FromText("a\nx\n"));

        Assert.Throws<InvalidOperationException>(
            () => new AddFeatureStep("c", "a", Arithmetic.Plus, "a").AddTo(table));
    }

    [Fact]
    public void AFeatureNeedsNamesOnBothSides()
    {
        Assert.Throws<ArgumentException>(() => new AddFeatureStep(" ", "a", Arithmetic.Plus, "b"));
        Assert.Throws<ArgumentException>(() => new AddFeatureStep("c", " ", Arithmetic.Plus, "b"));
        Assert.Throws<ArgumentException>(() => new AddFeatureStep("c", "a", Arithmetic.Plus, " "));
        Assert.Throws<ArgumentException>(() => new CyclicalStep(" ", Period.HourOfDay));
        Assert.Throws<ArgumentException>(() => new NormaliseStep(" "));
        Assert.Throws<ArgumentException>(() => new EncodeStep(" "));
        Assert.Throws<ArgumentException>(() => new NormaliseRowStep([]));
    }

    [Fact]
    public void AColumnWhereEveryValueIsTheSame_DoesNotDivideByNothing()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]),
            CsvRowSource.FromText("a\n5\n5\n"));

        var step = new NormaliseStep("a", Scale.MinMax);
        step.ApplyTo(table, step.Fit(table, [Part.Train, Part.Train]));

        Assert.Equal(0, ((Column<double>)table["a"])[0]);
    }

    [Fact]
    public void ARowOfNothingAtAll_IsLeftAlone()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([
                new ColumnDeclaration("a", ColumnKind.Number, true),
                new ColumnDeclaration("b", ColumnKind.Number, true)]),
            CsvRowSource.FromText("a,b\n,\n0,0\n"));

        new NormaliseRowStep(["a", "b"], Norm.Max).AddTo(table);

        Assert.True(table["a"].IsMissing(0));
        Assert.Equal(0, ((Column<double>)table["b"])[1]);
    }

    [Fact]
    public void ManyColumnsScaledInOneBreath()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Number("fare").Integer("pclass", "sibsp"))
            .SplitAtRandom(0.70, 0.15)
            .Normalise("fare", "pclass", "sibsp")
            .Build()
            .Run();

        Assert.Equal(3, prepared.Fitted.Keys.Count(at => prepared.Declaration.Steps[at] is IFittedStep));
    }

    // ---- the file ------------------------------------------------------------------------------------

    [Fact]
    public void EveryNewStepSurvivesTheFileOnItsOwn()
    {
        IPipelineStep[] steps =
        [
            new SplitAtRandomStep(new SplitShares(0.7, 0.15, 0.15), 42),
            new SplitStratifiedStep("g", new SplitShares(0.7, 0.15, 0.15), 42),
            new AddFeatureStep("c", "a", Arithmetic.Times, "b"),
            new CyclicalStep("t", Period.DayOfMonth, Form.Unit),
            new NormaliseStep("a", Scale.MaxAbs, OutOfRange.Refuse),
            new NormaliseRowStep(["a", "b"], Norm.Max),
            new EncodeStep("g", As.Ordinal, Unseen.Refuse),
        ];

        foreach (var step in steps)
        {
            // A step that learns needs a split before it; a split is the one thing that must not get a
            // second split in front of it, because a declaration divides its rows once.
            IPipelineStep schema = new DeclareStep([
                new ColumnDeclaration("t", ColumnKind.Timestamp, false),
                new ColumnDeclaration("g", ColumnKind.Category, false),
                new ColumnDeclaration("a", ColumnKind.Number, false),
                new ColumnDeclaration("b", ColumnKind.Number, false),
            ]);

            var one = new PipelineDeclaration(step is ISplitStep
                ? [schema, step]
                : [schema, new SplitByTimeStep("t", new SplitShares(0.7, 0.15, 0.15)), step]);

            Assert.Equal(one, PipelineDeclaration.FromJson(one.ToJson(), StepCatalog.BuiltIn()));
        }
    }

    [Fact]
    public void AScalingOfRowsWithoutItsColumnList_IsRefused()
    {
        const string json = """{"declaration":[{"step":"normalise.row","norm":"l2"}]}""";

        Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    [Fact]
    public void ADeclareStepWithoutItsColumnList_IsRefused()
    {
        const string json = """{"declaration":[{"step":"declare","remainder":"drop"}]}""";

        Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    [Fact]
    public void AStepMissingATrueOrFalse_IsRefused()
    {
        const string json = """
            {"declaration":[{"step":"declare","remainder":"drop",
                             "columns":[{"name":"a","kind":"number"}]}]}
            """;

        Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    [Fact]
    public void TwoStepsThatDifferOnlyInTheirLists_AreNotEqual()
    {
        Assert.NotEqual(
            (IPipelineStep)new NormaliseRowStep(["a"]),
            new NormaliseRowStep(["a", "b"]));

        Assert.NotEqual(
            new NormaliseRowStep(["a"]).GetHashCode(),
            new NormaliseRowStep(["a", "b"]).GetHashCode());

        Assert.NotEqual(
            (IPipelineStep)new DeclareStep([new ColumnDeclaration("a", ColumnKind.Text, false)]),
            new DeclareStep([new ColumnDeclaration("b", ColumnKind.Text, false)]));

        Assert.NotEqual(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Text, false)]).GetHashCode(),
            new DeclareStep([new ColumnDeclaration("b", ColumnKind.Text, false)]).GetHashCode());
    }

    [Fact]
    public void AGapInEveryKind_StaysAGap()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([
                new ColumnDeclaration("n", ColumnKind.Number, true),
                new ColumnDeclaration("i", ColumnKind.Integer, true),
                new ColumnDeclaration("b", ColumnKind.Boolean, true),
                new ColumnDeclaration("t", ColumnKind.Timestamp, true)]),
            CsvRowSource.FromText("n,i,b,t\n,,,\n"));

        Assert.All(table.Columns, column => Assert.True(column.IsMissing(0)));
    }

    [Fact]
    public void AFeatureCanBeWorkedOutFromTrueAndFalse()
    {
        // True counts as one and false as nothing, which is what a model would have to do with them anyway.
        var table = SchemaBinding.Bind(
            new DeclareStep([
                new ColumnDeclaration("a", ColumnKind.Boolean, true),
                new ColumnDeclaration("b", ColumnKind.Boolean, true)]),
            CsvRowSource.FromText("a,b\nTrue,False\n,True\n"));

        new AddFeatureStep("both", "a", Arithmetic.Plus, "b").AddTo(table);

        var both = (Column<double>)table["both"];

        Assert.Equal(1, both[0]);
        Assert.True(both.IsMissing(1));
    }

    [Fact]
    public void ReplayingAFillOntoWords_IsRefusedRatherThanGuessed()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Text, true)]),
            CsvRowSource.FromText("a\nx\n\n"));

        var learned = new FittedStepValues();
        learned.Learned("value", 1);

        Assert.Throws<InvalidOperationException>(
            () => FillMissingStep.Of("a", With.Mean).ApplyTo(table, learned));
    }

    [Fact]
    public void WhatAFitLearnedIsThereToRead()
    {
        var values = new FittedStepValues();
        values.Learned("value", 2.5);
        values.Learned("categories", ["a"]);

        Assert.Equal(2.5, values.Numbers["value"]);
        Assert.Equal(["a"], values.Lists["categories"]);
    }

    [Fact]
    public void APipelineThatLearnsNothing_NeedsNoSplitAtAll()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("sibsp", "parch"))
            .AddFeature("family", "sibsp", Arithmetic.Plus, "parch")
            .Build()
            .Run();

        Assert.Equal(891, prepared.CountIn(Part.Undivided));
        Assert.Empty(prepared.Fitted);
        Assert.True(prepared.Table.Has("family"));
    }

    [Fact]
    public void ATrueOrFalseColumnCountsAsOneAndNought_AndAGapStaysAGap()
    {
        // Yes and no are the one kind of word a model can be handed without an encoder, because there are
        // exactly two of them and nobody has to decide which comes first.
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("paid", ColumnKind.Boolean, true)]),
            CsvRowSource.FromText("paid\ntrue\nfalse\n\n"));

        var numbers = table.NumbersOf("paid");

        Assert.Equal(1, numbers[0]);
        Assert.Equal(0, numbers[1]);
        Assert.Null(numbers[2]);
    }

    [Fact]
    public void AWholeNumberColumnWithNothingInARow_IsAGapAndNotANought()
    {
        // A count of nothing and no count at all are different facts, and the difference survives the
        // widening to the kind of number the arithmetic works in.
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("trades", ColumnKind.Integer, true)]),
            CsvRowSource.FromText("trades\n8123\n\n"));

        var numbers = table.NumbersOf("trades");

        Assert.Equal(8123, numbers[0]);
        Assert.Null(numbers[1]);
    }
}
