// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A file that writes 0 for a fare nobody knows, or a question mark for a class nobody wrote down, says so in its own
/// dialect. The schema says it once, on the column, and every cell holding that value is read as the gap it is — the gap
/// a fill then counts and marks — while the row is still known by what the file wrote.
/// </summary>
public class MissingValueTests
{
    private static Table Read(string csv, ColumnDeclaration column) =>
        SchemaBinding.Bind(new DeclareStep([column]), CsvRowSource.FromText(csv));

    private static ColumnDeclaration Column(string name, ColumnKind kind, string missing, string? format = null) =>
        new(name, kind, Optional: false) { Missing = missing, Format = format };

    private static bool[] Gaps(Table table, string column) => [.. Enumerable.Range(0, table.RowCount).Select(table[column].IsMissing)];

    [Fact]
    public void ACellHoldingTheValueAColumnSaysMeansAGap_IsAGap_AsItsKindReadsTheValue()
    {
        // 0 and 0.0 are one number: a frame hands 0 over where the file writes 0.0.
        var table = Read("fare\n0\n0.0\n7.25\n\n 0 \n", Column("fare", ColumnKind.Number, "0"));

        Assert.Equal([true, true, false, true, true], Gaps(table, "fare"));
        Assert.Equal(7.25, ((Column<double>)table["fare"])[2]);
    }

    [Fact]
    public void AValueThatIsNotOfTheColumnsKind_IsComparedAsItIsWritten_AndAnythingElseUnreadableIsStillRefused()
    {
        var table = Read("age\n?\n3\n", Column("age", ColumnKind.Number, "?"));

        Assert.Equal([true, false], Gaps(table, "age"));

        var refused = Assert.Throws<FormatException>(() => Read("age\n?\nx\n", Column("age", ColumnKind.Number, "?")));

        Assert.Equal("Row 2, column 'age': 'x' is not a number.", refused.Message);
    }

    [Theory]
    [InlineData(ColumnKind.Integer, "-1", "-01", null)]
    [InlineData(ColumnKind.Boolean, "?", "?", null)]
    [InlineData(ColumnKind.Boolean, "no", "N", null)]
    [InlineData(ColumnKind.Timestamp, "1900-01-01", "1900-01-01T00:00:00", null)]
    [InlineData(ColumnKind.Timestamp, "01/01/1900", "1/1/1900", "d/M/yyyy")]
    [InlineData(ColumnKind.Text, "unknown", "unknown", null)]
    [InlineData(ColumnKind.Category, "?", "?", null)]
    public void EveryKindReadsItsValueThatMeansAGap(ColumnKind kind, string missing, string written, string? format)
    {
        var table = Read($"x\n{written}\n", Column("x", kind, missing, format));

        Assert.True(table["x"].IsMissing(0));
    }

    [Fact]
    public void AmongWords_TheValueIsTheWordExactlyAsWritten()
    {
        var table = Read("port\nunknown\nUnknown\n unknown\n", Column("port", ColumnKind.Category, "unknown"));

        Assert.Equal([true, false, false], Gaps(table, "port"));
    }

    [Fact]
    public void ARowIsStillKnownByWhatTheFileWrote()
    {
        const string csv = "fare\n0\n7.25\n";

        var said = Read(csv, Column("fare", ColumnKind.Number, "0"));
        var unsaid = Read(csv, new ColumnDeclaration("fare", ColumnKind.Number, false));

        Assert.Equal(unsaid.Identities, said.Identities);
    }

    [Fact]
    public void ThroughAPipeline_AFillCountsTheValueAsTheGapItIs()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Column("fare", ColumnKind.Number, missing: "0"))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("fare", With.Median)
            .Build()
            .Run();

        // The published file writes 0 for fifteen fares.
        Assert.Equal(15, Enumerable.Range(0, prepared.Table.RowCount).Count(row => ((Column<double>)prepared.Table["fare_was_missing"])[row] == 1));
    }

    [Fact]
    public void TheValueIsWrittenIntoTheFileOnlyWhereItIsSaid_AndReadBackAsItWas()
    {
        var declaration = Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema.Column("fare", ColumnKind.Number, missing: "0").Number("age"))
            .Declaration;

        var json = declaration.ToJson();

        Assert.Single(Regex.Matches(json, "\"missing\""));
        Assert.Contains("\"missing\": \"0\"", json, StringComparison.Ordinal);
        Assert.Equal(declaration, PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    [Fact]
    public void AValueOfNothingButSpaces_IsRefused()
    {
        var refused = Assert.Throws<ArgumentException>(() => new DeclareStep([Column("fare", ColumnKind.Number, " ")]));

        Assert.Contains("'fare'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheSchemasOwnOperation_SaysWhichValueMeansAGap_AndTakesItBack_AndAKindChangeKeepsIt()
    {
        var declare = new DeclareStep([new ColumnDeclaration("fare", ColumnKind.Number, false)]);

        var said = declare.WithColumnMissing("fare", "0");

        Assert.Equal("0", said.Columns[0].Missing);
        Assert.Equal(declare, said.WithColumnMissing("fare", null));
        Assert.Same(said, said.WithColumnMissing("fare", "0"));
        Assert.Equal("0", said.WithColumnKind("fare", ColumnKind.Integer).Columns[0].Missing);
        Assert.Throws<ArgumentException>(() => declare.WithColumnMissing("elsewhere", "0"));
    }

    [Fact]
    public void ATakeOver_ListsAChangeOfTheValueAlone()
    {
        string[] header = ["fare"];

        PipelineDeclaration Declared(string? missing) =>
            Pdd.Create().ReadCsv("fares.csv").Declare(schema => schema.Column("fare", ColumnKind.Number, missing: missing)).Declaration;

        var fare = Assert.Single(PipelinePreset.Of(Declared("0"), header).TakeOver(Declared(null), header).Changes);

        Assert.Null(fare.Before.Missing);
        Assert.Equal("0", fare.After.Missing);
    }
}
