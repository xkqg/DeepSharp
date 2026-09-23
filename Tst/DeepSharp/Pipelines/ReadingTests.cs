// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Opening real data and reading it into the declared columns. The file under test is a published dataset
/// rather than an invented one, because invented numbers demonstrate the happy path and nothing else: this
/// one has gaps of two very different sizes, a boolean dialect, and columns that restate the answer.
/// </summary>
public class ReadingTests
{
    private static string Titanic => Repository.Data("titanic.csv");

    [Fact]
    public void TextAlreadyInHand_IsReadAsItsFileWouldBe_AndARefusalNamesWhereItCameFrom()
    {
        // Bytes read once — to be fingerprinted, say — are parsed from memory rather than read a second time, and
        // a fault in them still names the file rather than "the text".
        var text = File.ReadAllText(Titanic);
        var fromTheFile = new CsvRowSource(Titanic);
        var inHand = CsvRowSource.FromText(text, Titanic);

        Assert.Equal(fromTheFile.ColumnNames, inHand.ColumnNames);
        Assert.Equal(fromTheFile.Rows, inHand.Rows);
        Assert.StartsWith("rows.csv line 2", Assert.Throws<FormatException>(() => CsvRowSource.FromText("a,b\n1\n", "rows.csv")).Message, StringComparison.Ordinal);
        Assert.StartsWith("The text line 2", Assert.Throws<FormatException>(() => CsvRowSource.FromText("a,b\n1\n")).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(() => CsvRowSource.FromText("a\n1\n", " "));
    }

    [Fact]
    public void ADeclaredPipeline_ReadsTheFileItWasPointedAt()
    {
        var table = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema
                .Integer("survived", "pclass", "sibsp", "parch")
                .Text("sex", "embarked")
                .Number("fare")
                .Optional("age", ColumnKind.Number)
                .Boolean("adult_male", "alone"))
            .Build()
            .Prepare();

        Assert.Equal(891, table.RowCount);
        Assert.Equal(10, table.Columns.Count);
        Assert.Equal("survived", table.Columns[0].Name);
    }

    [Fact]
    public void AColumnNobodyDeclared_DoesNotComeAlong()
    {
        // 'alive' is 'survived' written as a word, and 'class' is 'pclass' written as a word. A pipeline
        // that carried everything in the file would hand a model the answer it is being asked to predict.
        var table = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Integer("survived").Text("sex"))
            .Build()
            .Prepare();

        Assert.Equal(2, table.Columns.Count);
        Assert.False(table.Has("alive"));
        Assert.False(table.Has("class"));
    }

    [Fact]
    public void KeepingTheRest_BringsItAlongAsText()
    {
        var table = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Integer("survived"), Remainder.Keep)
            .Build()
            .Prepare();

        Assert.Equal(15, table.Columns.Count);
        Assert.Equal(ColumnKind.Text, table["fare"].Kind);
    }

    [Fact]
    public void RefusingTheRest_StopsAFileThatHasGrownAColumn()
    {
        var refused = Assert.Throws<InvalidOperationException>(
            () => Pdd.Create()
                .ReadCsv(Titanic)
                .Declare(schema => schema.Integer("survived"), Remainder.Refuse)
                .Build()
                .Prepare());

        Assert.Contains("pclass", refused.Message);
    }

    [Fact]
    public void AnEmptyCell_IsAGapAndNotAValue()
    {
        var table = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Optional("age", ColumnKind.Number).Text("deck"))
            .Build()
            .Prepare();

        var age = (Column<double>)table["age"];
        var deck = (TextColumn)table["deck"];

        // Measured against the published file: age is empty in 177 rows of 891, deck in 688.
        Assert.Equal(177, Enumerable.Range(0, table.RowCount).Count(age.IsMissing));
        Assert.Equal(688, Enumerable.Range(0, table.RowCount).Count(deck.IsMissing));

        // The first row has an age and no deck, which is the row the row-walk followed.
        Assert.Equal(22.0, age[0]);
        Assert.True(deck.IsMissing(0));
    }

    [Fact]
    public void ABooleanDialect_IsReadRatherThanTurnedIntoACategory()
    {
        // The file writes True and False with a capital, which is one tool's spelling and not the only one.
        var table = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Boolean("adult_male", "alone"))
            .Build()
            .Prepare();

        var adultMale = (Column<bool>)table["adult_male"];

        Assert.Equal(ColumnKind.Boolean, adultMale.Kind);
        Assert.True(adultMale[0]);
        Assert.False(((Column<bool>)table["alone"])[0]);
    }

    [Theory]
    [InlineData("true")]
    [InlineData("TRUE")]
    [InlineData("1")]
    [InlineData("yes")]
    [InlineData("y")]
    [InlineData("t")]
    public void EverySpellingOfTrue_IsTrue(string written)
    {
        var table = Read($"flag\n{written}\n", schema => schema.Boolean("flag"));

        Assert.True(((Column<bool>)table["flag"])[0]);
    }

    [Theory]
    [InlineData("false")]
    [InlineData("False")]
    [InlineData("0")]
    [InlineData("no")]
    [InlineData("n")]
    [InlineData("f")]
    public void EverySpellingOfFalse_IsFalse(string written)
    {
        var table = Read($"flag\n{written}\n", schema => schema.Boolean("flag"));

        Assert.False(((Column<bool>)table["flag"])[0]);
    }

    [Fact]
    public void AValueThatIsNotWhatItWasDeclaredToBe_IsRefusedWithItsRowAndColumn()
    {
        var refused = Assert.Throws<FormatException>(
            () => Read("age\n22\nmaybe\n", schema => schema.Number("age")));

        Assert.Contains("Row 2", refused.Message);
        Assert.Contains("age", refused.Message);
        Assert.Contains("maybe", refused.Message);
    }

    [Theory]
    [InlineData(ColumnKind.Integer, "half", "whole number")]
    [InlineData(ColumnKind.Boolean, "perhaps", "true or false")]
    [InlineData(ColumnKind.Timestamp, "last tuesday", "moment in time")]
    public void EveryKindSaysWhatItWanted(ColumnKind kind, string written, string wanted)
    {
        var refused = Assert.Throws<FormatException>(
            () => Read($"value\n{written}\n", schema => schema.Column("value", kind)));

        Assert.Contains(wanted, refused.Message);
    }

    [Fact]
    public void ADeclaredColumnTheSourceDoesNotHave_IsRefusedUnlessItWasOptional()
    {
        var refused = Assert.Throws<InvalidOperationException>(
            () => Read("age\n22\n", schema => schema.Number("age", "weight")));

        Assert.Contains("weight", refused.Message);

        var table = Read("age\n22\n", schema => schema.Number("age").Optional("weight", ColumnKind.Number));

        Assert.Single(table.Columns);
    }

    [Fact]
    public void APipelineThatNeverSaysWhereOrWhat_CannotBePrepared()
    {
        Assert.Throws<InvalidOperationException>(() => Pdd.Create().Build().Prepare());
        Assert.Throws<InvalidOperationException>(() => Pdd.Create().ReadCsv("x.csv").Build().Prepare());
    }

    [Fact]
    public void ATimeSeries_ReadsItsDatesTheSameOnEveryMachine()
    {
        var table = Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close").Text("direction"))
            .Build()
            .Prepare();

        var dates = (Column<DateTime>)table["Date"];

        Assert.Equal(506, table.RowCount);
        Assert.Equal(new DateTime(2015, 2, 17, 0, 0, 0, DateTimeKind.Utc), dates[0]);
        Assert.Equal("Increasing", ((TextColumn)table["direction"])[0]);
    }

    private static Table Read(string csv, Action<SchemaBuilder> schema)
    {
        var builder = new SchemaBuilder();
        schema(builder);

        return SchemaBinding.Bind(new DeclareStep(builder.Columns), CsvRowSource.FromText(csv));
    }
}
