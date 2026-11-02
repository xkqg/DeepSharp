// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What the earlier rows of a key already held, as one number: the training rows' mean <see cref="HistoryStep"/>
/// already reads, plus every other row of the key answered since the pipeline was fitted — a validation or a test
/// row read strictly before it, never one strictly after, the way a producer's own earlier rows are known by the
/// time a later one of it is served.
/// </summary>
/// <remarks>
/// Ten rows of one key, "A", t = 1..10, v = 10t. A split by time of 0.6/0.2 makes rows 1-6 training (v 10..60,
/// prior 35), 7-8 validation, 9-10 test. With a weight of one and every earlier row read: row 7, reading the
/// training floor alone (sum 210, count 6), reads (210 + 35) / 7 = 35 — the training rows' own mean, since the
/// prior already equals it. Row 8, with row 7's answer handed in beside it, reads the floor plus row 7's 70:
/// (210 + 70 + 35) / 8 = 39.375.
/// </remarks>
public class KnownHistoryTests
{
    private static string Rows(double row8 = 80) =>
        "t,key,v\n"
        + "1,A,10\n"
        + "2,A,20\n"
        + "3,A,30\n"
        + "4,A,40\n"
        + "5,A,50\n"
        + "6,A,60\n"
        + "7,A,70\n"
        + $"8,A,{row8.ToString(System.Globalization.CultureInfo.InvariantCulture)}\n"
        + "9,A,90\n"
        + "10,A,100\n";

    private static PreparedData Prepared(string rows) =>
        Pdd.Create()
            .Read(CsvRowSource.FromText(rows), "rows")
            .Declare(schema => schema.Integer("t").Category("key").Number("v"))
            .OrderBy("t")
            .SplitByTime("t", 0.6, 0.2)
            .KnownHistory("key", "v", new HistoryReach(1, 0), "t")
            .Build()
            .Run();

    private static double[] Known(PreparedData prepared) => [.. prepared.Table.NumbersOf("v_knownHistory_key").Select(value => value!.Value)];

    [Fact]
    public void ATrainingRow_ReadsOnlyTheTrainingFloor_AsHistoryAlreadyDoes()
    {
        var known = Known(Prepared(Rows()));

        // Rows 1..6: the training rows, in order, each reading the training rows strictly before it alone.
        Assert.Equal((0 + 35) / 1.0, known[0], 9);
        Assert.Equal((10 + 35) / 2.0, known[1], 9);
        Assert.Equal((30 + 35) / 3.0, known[2], 9);
        Assert.Equal((60 + 35) / 4.0, known[3], 9);
        Assert.Equal((100 + 35) / 5.0, known[4], 9);
        Assert.Equal((150 + 35) / 6.0, known[5], 9);
    }

    [Fact]
    public void ARowAnsweredSinceTheFit_IsReadByTheKeysLaterRows()
    {
        var known = Known(Prepared(Rows()));

        // Row 7 (validation): the training floor alone, which already equals the prior.
        Assert.Equal(35, known[6], 9);

        // Row 8 (validation), with row 7's answer in the same table: the floor plus row 7's 70.
        Assert.Equal((210 + 70 + 35) / 8.0, known[7], 9);
    }

    [Fact]
    public void AMutationOfAValidationRow_NeverMovesATrainingRowsColumn()
    {
        var before = Known(Prepared(Rows()));
        var after = Known(Prepared(Rows(row8: 999)));

        // Row 8 (changed) reads only rows strictly before it, so its own column does not move; rows 1..6
        // (training) read even less than that. Row 9, strictly after row 8, is the one that reads it.
        Assert.Equal(before[..8], after[..8]);
        Assert.NotEqual(before[8], after[8]);
    }

    [Fact]
    public void FitTimeAndReplayOfTheSameRows_GiveTheSameColumn()
    {
        var prepared = Prepared(Rows());
        var replayed = prepared.Replay(CsvRowSource.FromText(Rows()));

        Assert.Equal(
            prepared.Table.NumbersOf("v_knownHistory_key").Select(value => value!.Value),
            replayed.NumbersOf("v_knownHistory_key").Select(value => value!.Value));
    }

    [Fact]
    public void ARowServedAlone_ReadsExactlyWhatHistoryGivesForIt()
    {
        var prepared = Prepared(Rows());
        var alone = prepared.Replay(CsvRowSource.FromText("t,key,v\n8,A,\n"));

        var plain = Pdd.Create()
            .Read(CsvRowSource.FromText(Rows()), "rows")
            .Declare(schema => schema.Integer("t").Category("key").Number("v"))
            .OrderBy("t")
            .SplitByTime("t", 0.6, 0.2)
            .History("key", "v", new HistoryReach(1, 0), "t")
            .Build()
            .Run();

        Assert.Equal(plain.Table.NumbersOf("v_history_key")[7]!.Value, alone.NumbersOf("v_knownHistory_key")[0]!.Value, 9);
    }

    [Fact]
    public void ARowServedWithTheProducersLaterAnsweredRows_ReadsThem()
    {
        var prepared = Prepared(Rows());
        var withContext = prepared.Replay(CsvRowSource.FromText("t,key,v\n7,A,70\n8,A,\n"));

        Assert.Equal((210 + 70 + 35) / 8.0, withContext.NumbersOf("v_knownHistory_key")[1]!.Value, 9);
    }

    [Fact]
    public void UnderASplitAtRandom_KnownHistoryIsRefused_LikeHistoryAndFillMissingAre()
    {
        var fault = Assert.Single(Assert.Throws<DeclarationException>(() =>
            Pdd.Create()
                .Read(CsvRowSource.FromText(Rows()), "rows")
                .Declare(schema => schema.Integer("t").Category("key").Number("v"))
                .OrderBy("t")
                .SplitAtRandom(0.6, 0.2)
                .KnownHistory("key", "v", new HistoryReach(1, 0), "t")
                .Build()).Faults);

        Assert.Equal("history.known", fault.Verb);
        Assert.Contains("contiguous run", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AKnownHistoryThatReadsAValueAnOutputMadeFromRowsAhead_IsRefused()
    {
        var fault = Assert.Single(Assert.Throws<DeclarationException>(() =>
            Pdd.Create()
                .Read(CsvRowSource.FromText(Rows()), "rows")
                .Declare(schema => schema.Integer("t").Category("key").Number("v"))
                .OrderBy("t")
                .SplitByTime("t", 0.5, 0.2, 2)
                .Ahead("v", 2)
                .KnownHistory("key", "v.ahead2", new HistoryReach(1, 0), "t")
                .Build()).Faults);

        Assert.Equal("history.known", fault.Verb);
        Assert.Contains("later than the row", fault.Message, StringComparison.Ordinal);
    }
}
