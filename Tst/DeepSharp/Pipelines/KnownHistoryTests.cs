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

    // Takes every "members" key out of a file, wherever it stands, and says how many it took.
    private static int WithoutMembers(System.Text.Json.Nodes.JsonNode? node) => node switch
    {
        System.Text.Json.Nodes.JsonObject entry => (entry.Remove("members") ? 1 : 0) + entry.Select(property => WithoutMembers(property.Value)).ToArray().Sum(),
        System.Text.Json.Nodes.JsonArray list => list.Select(WithoutMembers).ToArray().Sum(),
        _ => 0,
    };

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

    // Twelve rows of one key, t = 1..12, v = 10t, dealt at random with seed 7 into t1, t6..t10 training (prior 68.33), t2,
    // t11, t12 validation and t3..t5 test.
    private static string Twelve(double row4 = 40) =>
        "t,key,v\n" + string.Concat(Enumerable.Range(1, 12).Select(t =>
            $"{t},A,{(t == 4 ? row4 : 10.0 * t).ToString(System.Globalization.CultureInfo.InvariantCulture)}\n"));

    private static PreparedData AtRandom(string rows) =>
        Pdd.Create()
            .Read(CsvRowSource.FromText(rows), "rows")
            .Declare(schema => schema.Integer("t").Category("key").Number("v"))
            .SplitAtRandom(0.5, 0.25, seed: 7)
            .KnownHistory("key", "v", new HistoryReach(1, 0), "t")
            .Build()
            .Run();

    [Fact]
    public void UnderASplitAtRandom_ATrainingRowReadsTheFloor_AndEveryOtherRowEveryEarlierKnownRowOfItsKey_AnyPart()
    {
        // The owner's answer to the seventh letter: a row whose outcome is asked for later reads every earlier row of its
        // key already known, whatever part it is in, as it would in production. A training row still reads training rows
        // alone, so nothing a model is chosen or measured on reaches what it learns from.
        var prepared = AtRandom(Twelve());
        var known = Known(prepared);
        Part[] parts = [Part.Train, Part.Validation, Part.Test, Part.Test, Part.Test, Part.Train, Part.Train, Part.Train, Part.Train, Part.Train, Part.Validation, Part.Validation];
        var prior = (10 + 60 + 70 + 80 + 90 + 100) / 6.0;

        Assert.Equal(parts, prepared.Parts);
        Assert.Equal(prior, known[0], 9);                                   // t1, training: nothing before it
        Assert.Equal((10 + 20 + prior) / 3, known[2], 9);                   // t3, test: t1 of the floor, t2 a validation row
        Assert.Equal((10 + 20 + 30 + prior) / 4, known[3], 9);              // t4, test: and t3, a test row
        Assert.Equal((10 + 20 + 30 + 40 + prior) / 5, known[4], 9);         // t5
        Assert.Equal((10 + prior) / 2, known[5], 9);                        // t6, training: t1 alone, never t2..t5
        Assert.Equal((410 + 140 + prior) / 11, known[10], 9);               // t11: the floor and t2..t5
        Assert.Equal((410 + 140 + 110 + prior) / 12, known[11], 9);         // t12: and t11
    }

    [Fact]
    public void UnderASplitAtRandom_AMutationOfARowAModelIsMeasuredOn_NeverMovesATrainingRowsColumn()
    {
        var before = AtRandom(Twelve());
        var after = AtRandom(Twelve(row4: 999));
        var training = Enumerable.Range(0, before.Table.RowCount).Where(row => before.Parts[row] == Part.Train).ToArray();

        Assert.Equal(before.Parts, after.Parts);

        Assert.Equal(training.Select(row => Known(before)[row]), training.Select(row => Known(after)[row]));
        Assert.NotEqual(Known(before)[4], Known(after)[4]);
    }

    [Fact]
    public void UnderASplitAtRandom_FitTimeAndReplayOfTheSameRows_GiveTheSameColumn()
    {
        // A row of the fit is known again on replay by what it says, so a training row reads the floor alone on both.
        var prepared = AtRandom(Twelve());
        var replayed = prepared.Replay(CsvRowSource.FromText(Twelve()));

        Assert.Equal(Known(prepared), replayed.NumbersOf("v_knownHistory_key").Select(value => value!.Value));
    }

    [Fact]
    public void ATrainingRowWithAGapAfterItsKeysLastValue_ReadsTheFloorAlone()
    {
        // A training row whose value is a gap is not in the floor, which keeps values; it is still a training row, and
        // reads no row a model is chosen on: t5 reads t1 alone, never t3.
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText("t,key,v,part\n1,A,10,0\n3,A,30,1\n5,A,,0\n7,A,70,2\n"), "four rows")
            .Declare(schema => schema.Integer("t").Category("key").Optional("v", ColumnKind.Number).Number("part"))
            .Split(new SplitByPartColumnStep("part"))
            .KnownHistory("key", "v", new HistoryReach(1, 0), "t")
            .Build()
            .Run();

        Assert.Equal((10 + 10) / 2.0, Known(prepared)[2], 9);
    }

    [Fact]
    public void KnownHistory_StandsWithoutAnOrderAbove_AndBelowASplitInTimeOrderedByAnotherColumn()
    {
        // It orders by its own columns, so it asks for no order of the rows and no particular one.
        Pdd.Create()
            .Read(CsvRowSource.FromText(Rows()), "rows")
            .Declare(schema => schema.Integer("t").Category("key").Number("v"))
            .SplitAtRandom(0.6, 0.2)
            .KnownHistory("key", "v", new HistoryReach(1, 0), "t")
            .Build();

        Pdd.Create()
            .Read(CsvRowSource.FromText(Rows()), "rows")
            .Declare(schema => schema.Integer("t").Category("key").Number("v"))
            .OrderBy("v")
            .SplitByTime("t", 0.6, 0.2)
            .KnownHistory("key", "v", new HistoryReach(1, 0), "t")
            .Build();
    }

    [Fact]
    public void AFitThatKeptNoMembers_AsOneThatAnEarlierReleaseWrote_ReplaysTheColumnItWrote()
    {
        // 0.18.0 kept the floor alone and read the rows after a key's last training row; a pipeline it fitted, read now,
        // serves what it served then.
        var prepared = Prepared(Rows());
        var json = System.Text.Json.Nodes.JsonNode.Parse(prepared.ToJson())!;

        Assert.Equal(1, WithoutMembers(json));

        var earlier = PreparedData.FromJson(json.ToJsonString(), StepCatalog.BuiltIn());

        Assert.Equal(
            prepared.Replay(CsvRowSource.FromText("t,key,v\n7,A,70\n8,A,\n")).NumbersOf("v_knownHistory_key"),
            earlier.Replay(CsvRowSource.FromText("t,key,v\n7,A,70\n8,A,\n")).NumbersOf("v_knownHistory_key"));
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(-1, 1)]
    [InlineData(double.NaN, 1)]
    [InlineData(1, -1)]
    public void ANoWindowAndNoWeight_OrANumberThatMeansNothing_IsRefusedWhereTheStepIsMade(double shrink, int window)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new KnownHistoryStep("key", "v", ["t"], new HistoryReach(shrink, window)));
    }

    [Fact]
    public void TwoKnownHistorySteps_AreEqual_WhenTheySayTheSameThing()
    {
        var one = new KnownHistoryStep("key", "v", ["t"], new HistoryReach(1, 2));

        Assert.Equal(one, new KnownHistoryStep("key", "v", ["t"], new HistoryReach(1, 2)));
        Assert.Equal(one.GetHashCode(), new KnownHistoryStep("key", "v", ["t"], new HistoryReach(1, 2)).GetHashCode());
        Assert.NotEqual(one, new KnownHistoryStep("other", "v", ["t"], new HistoryReach(1, 2)));
        Assert.NotEqual(one, new KnownHistoryStep("key", "other", ["t"], new HistoryReach(1, 2)));
        Assert.NotEqual(one, new KnownHistoryStep("key", "v", ["other"], new HistoryReach(1, 2)));
        Assert.NotEqual(one, new KnownHistoryStep("key", "v", ["t"], new HistoryReach(2, 2)));
        Assert.NotEqual(one, new KnownHistoryStep("key", "v", ["t"], new HistoryReach(1, 3)));
        Assert.NotEqual(one, one with { Written = "named" });
        Assert.False(one.Equals(null));
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
