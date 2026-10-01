// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.RegularExpressions;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A moment is read the way its column says its moments are written, and a column that says nothing is read as ISO 8601
/// writes a moment. Never by a reading that fills in from the clock what the text leaves out: that one read 7.25 as the
/// twenty-fifth of July of whichever year it ran in, and 02/03/2015 as the third of February on every machine.
/// </summary>
public class TimestampFormatTests
{
    private static Table Read(string csv, Action<SchemaBuilder> schema)
    {
        var builder = new SchemaBuilder();
        schema(builder);

        return SchemaBinding.Bind(new DeclareStep(builder.Columns), CsvRowSource.FromText(csv));
    }

    private static string Moment(Table table, string column, int row = 0) =>
        ((Column<DateTime>)table[column])[row]!.Value.ToString("o", CultureInfo.InvariantCulture);

    private static readonly string[] Header = ["when", "close"];

    private static PipelineDeclaration Prices(string? format) =>
        Pdd.Create()
            .ReadCsv("prices.csv")
            .Declare(schema => schema.Column(new ColumnDeclaration("when", ColumnKind.Timestamp, Optional: false) { Format = format }).Number("close"))
            .Declaration;

    [Fact]
    public void ADateWrittenDayFirst_IsReadByTheFormatItsColumnDeclares()
    {
        var table = Read("when\n27/11/2015\n02/03/2015\n", schema => schema.Column(new ColumnDeclaration("when", ColumnKind.Timestamp, Optional: false) { Format = "dd/MM/yyyy" }));

        Assert.Equal("2015-11-27T00:00:00.0000000Z", Moment(table, "when"));
        Assert.Equal("2015-03-02T00:00:00.0000000Z", Moment(table, "when", 1));
    }

    [Fact]
    public void ACellWrittenAnotherWayThanItsColumnsFormat_IsRefusedSayingHowItShouldBeWritten()
    {
        var refused = Assert.Throws<FormatException>(
            () => Read("when\n27/11/2015\n7.25\n", schema => schema.Column(new ColumnDeclaration("when", ColumnKind.Timestamp, Optional: false) { Format = "dd/MM/yyyy" })));

        Assert.Equal("Row 2, column 'when': '7.25' is not a moment in time written as dd/MM/yyyy.", refused.Message);
    }

    [Theory]
    [InlineData("2015-02-18", "2015-02-18T00:00:00.0000000Z")]
    [InlineData("2015-02-18 09:30", "2015-02-18T09:30:00.0000000Z")]
    [InlineData("2015-02-18 09:30:15", "2015-02-18T09:30:15.0000000Z")]
    [InlineData("2015-02-18T09:30", "2015-02-18T09:30:00.0000000Z")]
    [InlineData("2015-02-18T09:30:15", "2015-02-18T09:30:15.0000000Z")]
    [InlineData("2015-02-18T09:30:15Z", "2015-02-18T09:30:15.0000000Z")]
    [InlineData("2015-02-18T09:30:15+01:00", "2015-02-18T08:30:15.0000000Z")]
    [InlineData("2015-02-18T09:30:15.123", "2015-02-18T09:30:15.1230000Z")]
    [InlineData("2015-02-18T09:30:15.1230000Z", "2015-02-18T09:30:15.1230000Z")]
    [InlineData("2015-02-18T09:30:15.1230000+01:00", "2015-02-18T08:30:15.1230000Z")]
    public void AColumnThatNamesNoFormat_ReadsEveryShapeOfIso8601AsTheMomentItWrites(string written, string moment)
    {
        var table = Read($"when\n{written}\n", schema => schema.Timestamp("when"));

        Assert.Equal(moment, Moment(table, "when"));
    }

    [Theory]
    [InlineData("7.25")]
    [InlineData("1.5")]
    [InlineData("02/03/2015")]
    [InlineData("27/11/2015")]
    [InlineData("12:30")]
    [InlineData("Feb 18 2015")]
    public void ACellThatIsNotIso8601_IsRefusedWhereTheColumnNamesNoFormat(string written)
    {
        var refused = Assert.Throws<FormatException>(() => Read($"when\n{written}\n", schema => schema.Timestamp("when")));

        Assert.Equal(
            $"Row 1, column 'when': '{written}' is not a moment in time as ISO 8601 writes one, such as 2015-02-18 or "
            + "2015-02-18T09:30:15; a column of moments written another way declares its format.",
            refused.Message);
    }

    [Fact]
    public void AFormatIsWrittenIntoTheFileOnlyWhereItIsSaid_AndReadBackAsItWas()
    {
        var declaration = Pdd.Create()
            .ReadCsv("prices.csv")
            .Declare(schema => schema.Column(new ColumnDeclaration("when", ColumnKind.Timestamp, Optional: false) { Format = "dd/MM/yyyy" }).Timestamp("logged").Number("close"))
            .Declaration;

        var json = declaration.ToJson();

        Assert.Single(Regex.Matches(json, "\"format\""));
        Assert.Contains("\"format\": \"dd/MM/yyyy\"", json, StringComparison.Ordinal);
        Assert.Equal(declaration, PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    [Fact]
    public void AFileIsWrittenAgainstTheFourthVersion_AndOneWrittenAgainstTheSecondIsReadStill()
    {
        Assert.Equal(4, PipelineDeclaration.Version);

        var second = """{"version":2,"declaration":[{"step":"read.csv","path":"x.csv"},{"step":"declare","remainder":"drop","columns":[{"name":"when","kind":"timestamp","optional":false}]}]}""";

        Assert.Equal(2, PipelineDeclaration.FromJson(second, StepCatalog.BuiltIn()).Steps.Count);
    }

    [Fact]
    public void AFormatOnAColumnThatHoldsNoMoments_IsRefused()
    {
        var refused = Assert.Throws<ArgumentException>(
            () => new DeclareStep([new ColumnDeclaration("close", ColumnKind.Number, false) { Format = "dd/MM/yyyy" }]));

        Assert.Contains("only a timestamp", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFormatOfNothingButSpaces_IsRefused()
    {
        Assert.Throws<ArgumentException>(
            () => new DeclareStep([new ColumnDeclaration("when", ColumnKind.Timestamp, false) { Format = " " }]));
    }

    [Fact]
    public void AColumnOfMomentsMadeAnotherKind_ForgetsItsFormat()
    {
        var declare = (DeclareStep)Prices("dd/MM/yyyy").Steps[1];

        var text = declare.WithColumnKind("when", ColumnKind.Text);

        Assert.Null(text.Columns.Single(column => column.Name == "when").Format);
    }

    [Fact]
    public void TheSchemasOwnOperation_GivesAColumnItsFormat_AndTakesItAway()
    {
        var declare = (DeclareStep)Prices(null).Steps[1];

        var written = declare.WithColumnFormat("when", "dd/MM/yyyy");
        var again = written.WithColumnFormat("when", null);

        Assert.Equal("dd/MM/yyyy", written.Columns.Single(column => column.Name == "when").Format);
        Assert.Equal(declare, again);
        Assert.Same(written, written.WithColumnFormat("when", "dd/MM/yyyy"));
        Assert.Throws<ArgumentException>(() => declare.WithColumnFormat("close", "dd/MM/yyyy"));
        Assert.Throws<ArgumentException>(() => declare.WithColumnFormat("elsewhere", "dd/MM/yyyy"));
    }

    [Fact]
    public void ThePipelinesOwnOperation_ReadsAColumnsMomentsByAFormat_WhereTheSchemaNamesIt()
    {
        var written = new PipelineDeclaration(Prices(null).WithFormat("when", "dd/MM/yyyy"));
        var unread = new PipelineDeclaration([new ReadCsvStep("prices.csv")]);

        Assert.Equal("dd/MM/yyyy", ((DeclareStep)written.Steps[1]).Columns[0].Format);
        Assert.Same(unread.Steps, unread.WithFormat("when", "dd/MM/yyyy"));
        Assert.Throws<ArgumentNullException>(() => ((PipelineDeclaration)null!).WithFormat("when", null));
        Assert.DoesNotContain(
            "(Parameter",
            Assert.Single(Assert.Throws<DeclarationException>(() => Prices(null).WithFormat("close", "dd/MM/yyyy")).Faults).Message,
            StringComparison.Ordinal);
    }

    [Fact]
    public void ATakeOver_ListsAChangeOfFormatAlone()
    {
        var takenOver = PipelinePreset.Of(Prices("dd/MM/yyyy"), Header).TakeOver(Prices(null), Header);

        var when = Assert.Single(takenOver.Changes);

        Assert.Equal("when", when.Column);
        Assert.Null(when.Before.Format);
        Assert.Equal("dd/MM/yyyy", when.After.Format);
    }
}
