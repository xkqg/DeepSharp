// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Two leaks one rule closes: a row that carries the value before it forward, or a history that reads a key's earlier
/// rows, read whichever part happened to stand there under a split at random or a stratified split — only a split in
/// time keeps every part a contiguous run; and a history reading a value an output already made from rows ahead reads
/// a value from later than the row it stands on, through every earlier row that holds it.
/// </summary>
public class EarlierAnswersFollowTimeTests
{
    [Fact]
    public void ARowCarriedForward_UnderASplitAtRandom_IsRefused_BecauseAPartIsNotAContiguousRun()
    {
        var fault = Assert.Single(Assert.Throws<DeclarationException>(() =>
            Pdd.Create()
                .Read(CsvRowSource.FromText("a,b\n1,1\n2,\n3,3\n4,4\n5,5\n6,6\n7,7\n8,8\n9,9\n10,10\n"), "ten rows")
                .Declare(schema => schema.Number("a", "b"))
                .OrderBy("b")
                .SplitAtRandom(0.70, 0.15)
                .FillMissing("a", With.Previous)
                .Build()).Faults);

        Assert.Equal("fill.missing", fault.Verb);
        Assert.Contains("contiguous run", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARowCarriedForward_UnderASplitByTime_OrderedByTheSplitsOwnColumn_IsNotRefused()
    {
        // The safe shape HistoryTests/FittingTests already rely on: unchanged by this rule.
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Optional("AAPL.Close", ColumnKind.Number))
            .OrderBy("Date")
            .SplitByTime("Date", 0.70, 0.15)
            .FillMissing("AAPL.Close", With.Previous)
            .Build()
            .Run();

        Assert.DoesNotContain(
            Enumerable.Range(0, prepared.Table.RowCount), ((Column<double>)prepared.Table["AAPL.Close"]).IsMissing);
    }

    [Fact]
    public void ARowCarriedForward_UnderASplitByTime_OrderedByAnotherColumn_IsRefused()
    {
        var fault = Assert.Single(Assert.Throws<DeclarationException>(() =>
            Pdd.Create()
                .Read(CsvRowSource.FromText("a,b,c\n1,1,9\n2,,8\n3,3,7\n4,4,6\n5,5,5\n6,6,4\n7,7,3\n8,8,2\n9,9,1\n10,10,0\n"), "ten rows")
                .Declare(schema => schema.Number("a", "b", "c"))
                .OrderBy("c")
                .SplitByTime("b", 0.70, 0.15)
                .FillMissing("a", With.Previous)
                .Build()).Faults);

        Assert.Equal("fill.missing", fault.Verb);
        Assert.Contains("order them by 'b'", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AHistoryThatReadsAValueAnOutputMadeFromRowsAhead_IsRefused()
    {
        var fault = Assert.Single(Assert.Throws<DeclarationException>(() =>
            Pdd.Create()
                .Read(CsvRowSource.FromText(
                    "t,date,key,v\n"
                    + "1,2025-01-01,A,10\n"
                    + "2,2025-01-02,A,20\n"
                    + "3,2025-01-03,A,30\n"
                    + "4,2025-01-04,A,40\n"
                    + "5,2025-01-05,A,50\n"
                    + "6,2025-01-06,A,60\n"), "rows")
                .Declare(schema => schema.Integer("t").Timestamp("date").Category("key").Number("v"))
                .OrderBy("t")
                .SplitByTime("t", 0.6, 0.2, 2)
                .Ahead("v", 2)
                .History("key", "v.ahead2", 2, "date")
                .Build()).Faults);

        Assert.Equal("history", fault.Verb);
        Assert.Contains("later than the row", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AHistoryThatReadsAPlainValue_IsNotRefused()
    {
        // The existing, already-tested shape (HistoryTests): reading the column itself, not an Ahead-made answer.
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText(
                "t,date,key,v,y\n"
                + "1,2025-01-01,A,10,0\n"
                + "2,2025-01-02,A,20,1\n"
                + "3,2025-01-03,A,30,0\n"), "rows")
            .Declare(schema => schema.Integer("t").Timestamp("date").Category("key").Number("v", "y"))
            .SplitByTime("t", train: 0.6, validation: 0.2)
            .Target("y")
            .History("key", "v", 2, "date")
            .Drop("t", "date", "key")
            .Build()
            .Run();

        Assert.NotNull(prepared);
    }
}
