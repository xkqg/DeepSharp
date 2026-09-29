// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a source's cells say each column holds, proposed for a person to accept or change. Every cell of every row is
/// read, by the same reading the schema binds with, so a kind proposed is a kind every row binds as; and nothing is
/// taken in by it: a proposal is a value to decide from, never a step, and a schema is still written by a person.
/// </summary>
public class KindProposalTests
{
    private static string Titanic => Repository.Data("titanic.csv");

    private static string Apple => Repository.Data("apple.csv");

    private static KindProposal Proposed(string csv) => KindProposal.Of(CsvRowSource.FromText(csv));

    private static ColumnProposal Only(string csv) => Assert.Single(Proposed(csv).Columns);

    // A schema that takes every column as it was proposed.
    private static DeclareStep Accepted(KindProposal proposal) =>
        new([.. proposal.Columns.Select(column => new ColumnDeclaration(column.Name, column.Kind, Optional: false) { Format = column.Format })]);

    [Fact]
    public void TitanicsColumns_AreProposedFromEveryRow()
    {
        var proposal = KindProposal.Of(new CsvRowSource(Titanic));

        Assert.Equal(new CsvRowSource(Titanic).ColumnNames, proposal.Columns.Select(column => column.Name));

        Assert.All(["survived", "pclass", "sibsp", "parch"], name => Assert.Equal(ColumnKind.Integer, proposal[name].Kind));
        Assert.All(["age", "fare"], name => Assert.Equal(ColumnKind.Number, proposal[name].Kind));
        Assert.All(["sex", "embarked", "class", "who", "deck", "embark_town"], name => Assert.Equal(ColumnKind.Category, proposal[name].Kind));
        Assert.All(["adult_male", "alone", "alive"], name => Assert.Equal(ColumnKind.Boolean, proposal[name].Kind));

        // Whole numbers few enough to be groups are offered as a category, and never proposed as one: a class is a group
        // and a count of siblings is not, and nothing in the cells tells the two apart.
        Assert.All(["survived", "pclass", "sibsp", "parch"], name => Assert.Equal(ColumnKind.Category, proposal[name].Offered));
        Assert.Null(proposal["fare"].Offered);
        Assert.Null(proposal["sex"].Offered);

        // Counted over all 891 rows, as the published file holds them.
        Assert.Equal(new ColumnProposal("age", ColumnKind.Number, null, [], null, Values: 714, Gaps: 177, Distinct: null), proposal["age"]);
        Assert.Equal(new ColumnProposal("deck", ColumnKind.Category, null, [], null, Values: 203, Gaps: 688, Distinct: 7), proposal["deck"]);
        Assert.Equal(new ColumnProposal("embarked", ColumnKind.Category, null, [], null, Values: 889, Gaps: 2, Distinct: 3), proposal["embarked"]);
        Assert.Equal(new ColumnProposal("pclass", ColumnKind.Integer, ColumnKind.Category, [], null, Values: 891, Gaps: 0, Distinct: 3), proposal["pclass"]);
        Assert.Equal(2, proposal["alive"].Distinct);
    }

    [Fact]
    public void ThePriceSeries_ProposesItsDatesAsMoments_ItsVolumeAsWholeNumbers_AndItsDirectionAsACategory()
    {
        var proposal = KindProposal.Of(new CsvRowSource(Apple));

        Assert.Equal(ColumnKind.Timestamp, proposal["Date"].Kind);
        Assert.Null(proposal["Date"].Format);
        Assert.Equal(506, proposal["Date"].Values);
        Assert.Equal(ColumnKind.Integer, proposal["AAPL.Volume"].Kind);
        Assert.Null(proposal["AAPL.Volume"].Offered);
        Assert.Equal(ColumnKind.Category, proposal["direction"].Kind);
        Assert.Equal(2, proposal["direction"].Distinct);
        Assert.All(
            ["AAPL.Open", "AAPL.High", "AAPL.Low", "AAPL.Close", "AAPL.Adjusted", "dn", "mavg", "up"],
            name => Assert.Equal(ColumnKind.Number, proposal[name].Kind));
    }

    [Theory]
    [InlineData("titanic.csv")]
    [InlineData("apple.csv")]
    public void EveryKindProposed_BindsEveryRowOfTheFileItWasProposedFrom(string file)
    {
        var source = new CsvRowSource(Repository.Data(file));

        var table = SchemaBinding.Bind(Accepted(KindProposal.Of(source)), source);

        Assert.Equal(source.Rows.Count(), table.RowCount);
        Assert.Equal(source.ColumnNames.Count, table.Columns.Count);
    }

    [Theory]
    [InlineData("27/11/2015", "d/M/yyyy", "2015-11-27T00:00:00")]
    [InlineData("11/27/2015", "M/d/yyyy", "2015-11-27T00:00:00")]
    [InlineData("27-11-2015", "d-M-yyyy", "2015-11-27T00:00:00")]
    [InlineData("11-27-2015", "M-d-yyyy", "2015-11-27T00:00:00")]
    [InlineData("27.11.2015", "d.M.yyyy", "2015-11-27T00:00:00")]
    [InlineData("2015/11/27", "yyyy/M/d", "2015-11-27T00:00:00")]
    [InlineData("2015-11-7", "yyyy-M-d", "2015-11-07T00:00:00")]
    [InlineData("27/11/2015 09:30", "d/M/yyyy H:mm", "2015-11-27T09:30:00")]
    [InlineData("27/11/2015 21:30:15", "d/M/yyyy H:mm:ss", "2015-11-27T21:30:15")]
    [InlineData("11/27/2015 9:30 PM", "M/d/yyyy h:mm tt", "2015-11-27T21:30:00")]
    [InlineData("11/27/2015 9:30:15 PM", "M/d/yyyy h:mm:ss tt", "2015-11-27T21:30:15")]
    [InlineData("27 Nov 2015", "d MMM yyyy", "2015-11-27T00:00:00")]
    [InlineData("27 November 2015", "d MMMM yyyy", "2015-11-27T00:00:00")]
    [InlineData("Nov 27, 2015", "MMM d, yyyy", "2015-11-27T00:00:00")]
    [InlineData("November 27, 2015", "MMMM d, yyyy", "2015-11-27T00:00:00")]
    public void MomentsWrittenAnotherWayThanIso8601_AreProposedWithTheOneFormatThatReadsThemAll(string written, string format, string moment)
    {
        var column = Only($"when\n\"{written}\"\n\"{written}\"\n");

        Assert.Equal(ColumnKind.Timestamp, column.Kind);
        Assert.Equal(format, column.Format);
        Assert.Equal([format], column.Formats);

        var table = SchemaBinding.Bind(new DeclareStep([new ColumnDeclaration("when", ColumnKind.Timestamp, false) { Format = column.Format }]), CsvRowSource.FromText($"when\n\"{written}\"\n"));

        Assert.Equal(moment, ((Column<DateTime>)table["when"])[0]!.Value.ToString("s", CultureInfo.InvariantCulture));
    }

    [Fact]
    public void MomentsWhoseOrderOfDayAndMonthTheCellsCannotSettle_AreOfferedAsMoments_WithEveryFormatThatReadsThem_AndProposedAsNone()
    {
        // The first day of every month for two years: read day first or month first, every cell is a date, and the two
        // readings disagree on all but two of them. The file cannot say which it meant, so a person does.
        var months = string.Join('\n', Enumerable.Range(0, 24).Select(month => $"01/{(month % 12) + 1:00}/{2015 + (month / 12)}"));

        var column = Only($"month\n{months}\n");

        Assert.Equal(ColumnKind.Text, column.Kind);
        Assert.Equal(ColumnKind.Timestamp, column.Offered);
        Assert.Equal(["d/M/yyyy", "M/d/yyyy"], column.Formats);
        Assert.Null(column.Format);
    }

    [Theory]
    [InlineData("12:30\n13:45")]
    [InlineData("18/02\n19/02")]
    [InlineData("Feb 18\nFeb 19")]
    public void AnythingThatDoesNotNameAYearAMonthAndADay_IsNeverAMoment(string cells)
    {
        // A reading of any of these fills in what it leaves out from the day it runs.
        var column = Only($"when\n{cells}\n");

        Assert.Equal(ColumnKind.Text, column.Kind);
        Assert.Null(column.Offered);
        Assert.Empty(column.Formats);
    }

    [Theory]
    [InlineData("yes\nno\nyes", ColumnKind.Boolean, null)]
    [InlineData("True\nFalse\nTrue", ColumnKind.Boolean, null)]
    [InlineData("t\nf\nt", ColumnKind.Boolean, null)]
    [InlineData("1\nyes\n0", ColumnKind.Boolean, null)]
    [InlineData("0\n1\n1", ColumnKind.Integer, null)]
    [InlineData("1\n1\n1", ColumnKind.Integer, null)]
    [InlineData("0.5\n1\n2", ColumnKind.Number, null)]
    [InlineData("1e3\n2\nNaN", ColumnKind.Number, null)]
    [InlineData("2015-02-18\n2015-02-19T09:30:15Z\n2015-02-20 10:00", ColumnKind.Timestamp, null)]
    public void EachCellsForm_DecidesTheKind_TrueAndFalseBeforeNumbers_ExceptNoughtsAndOnes(string cells, ColumnKind kind, ColumnKind? offered)
    {
        var column = Only($"value\n{cells}\n");

        Assert.Equal(kind, column.Kind);
        Assert.Equal(offered, column.Offered);
    }

    [Fact]
    public void WordsAreACategory_WhenTheyAreFewAndEachStandsForManyRows_AndTextOtherwise()
    {
        static string Words(int distinct, int rows) =>
            string.Join('\n', Enumerable.Range(0, rows).Select(row => $"w{row % distinct}"));

        Assert.Equal(ColumnKind.Category, Only($"w\n{Words(64, 1280)}\n").Kind);
        Assert.Equal(ColumnKind.Text, Only($"w\n{Words(65, 1300)}\n").Kind);
        Assert.Null(Only($"w\n{Words(65, 1300)}\n").Distinct);
        Assert.Equal(ColumnKind.Text, Only($"w\n{Words(64, 1279)}\n").Kind);
        Assert.Equal(ColumnKind.Category, Only($"w\n{Words(1, 20)}\n").Kind);
        Assert.Equal(ColumnKind.Text, Only($"w\n{Words(2, 39)}\n").Kind);

        // Whole numbers that few are offered as a category beside them, and never proposed as one.
        var groups = Only($"n\n{string.Join('\n', Enumerable.Range(0, 80).Select(row => row % 4))}\n");

        Assert.Equal(ColumnKind.Integer, groups.Kind);
        Assert.Equal(ColumnKind.Category, groups.Offered);
        Assert.Null(Only($"n\n{string.Join('\n', Enumerable.Range(0, 64))}\n").Offered);
    }

    [Fact]
    public void AGap_IsCountedAsTheKindProposedCountsOne()
    {
        // Spaces are a gap among numbers and a word among words; an empty cell is a gap for both.
        var numbers = Only("n\n1\n  \n\n2\n");
        var words = Only("w\na\n  \n\nb\n");
        var nothing = Only("x\n\n\n");

        Assert.Equal(new ColumnProposal("n", ColumnKind.Integer, null, [], null, Values: 2, Gaps: 2, Distinct: 2), numbers);
        Assert.Equal(new ColumnProposal("w", ColumnKind.Text, null, [], null, Values: 3, Gaps: 1, Distinct: 3), words);
        Assert.Equal(new ColumnProposal("x", ColumnKind.Text, null, [], null, Values: 0, Gaps: 2, Distinct: 0), nothing);
    }

    [Fact]
    public void ACellMissingFromAShortRow_IsAGap()
    {
        var column = KindProposal.Of(new InMemoryRowSource(["a", "b"], [["1", "2"], ["3"]]))["b"];

        Assert.Equal(new ColumnProposal("b", ColumnKind.Integer, null, [], null, Values: 1, Gaps: 1, Distinct: 1), column);
    }

    [Fact]
    public void TheChainDoor_HandsBackTheProposal_ForAPersonToWriteTheSchemaFrom()
    {
        var chained = Pdd.Create().ReadCsv(Titanic).ProposedKinds();
        var handed = Pdd.Create().Read(new CsvRowSource(Titanic), "the titanic").ProposedKinds();

        Assert.Equal(KindProposal.Of(new CsvRowSource(Titanic)).Columns, chained.Columns);
        Assert.Equal(chained.Columns, handed.Columns);
        Assert.Throws<InvalidOperationException>(() => Pdd.Create().ProposedKinds());
    }

    [Fact]
    public void AColumnTheSourceDoesNotHave_IsNotProposed()
    {
        var proposal = Proposed("a\n1\n");

        Assert.Contains("'b'", Assert.Throws<ArgumentException>(() => proposal["b"]).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => KindProposal.Of(null!));
    }

    [Fact]
    public void TwoProposalsOfTheSameCells_AreEqual_ColumnByColumn()
    {
        var one = Only("month\n01/02/2015\n01/03/2015\n");
        var other = Only("month\n01/02/2015\n01/03/2015\n");

        Assert.Equal(one, other);
        Assert.Equal(one.GetHashCode(), other.GetHashCode());
        Assert.NotEqual(one, other with { Formats = ["d/M/yyyy"] });
        Assert.NotEqual(one, other with { Values = 3 });
    }
}
