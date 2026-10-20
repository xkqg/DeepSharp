// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A history over the last few places of the order instead of every earlier one: the mean of a farm's last three flocks. The
/// window counts distinct places of the order, so rows of one date are one place and a window never cuts between them; a row reads
/// only training rows strictly before it, as every history does, so nothing a model is chosen or measured on reaches a feature,
/// and a row served alone reads the last places of the training rows. Without a weight to pull it towards the training rows' mean,
/// a window that holds nothing reads that mean.
/// </summary>
/// <remarks>
/// The rows are the ones of <see cref="HistoryTests"/>: of key A, the training rows 1 (day 1, 10), 3 and 4 (day 2, 30 and 50 — rows
/// of one place), 6 (day 4, 60). The mean of every training row is 35.
/// </remarks>
public class HistoryWindowTests
{
    private const string Rows =
        "t,date,round,key,v,y\n"
        + "1,2025-01-01,1,A,10,0\n"
        + "2,2025-01-01,1,B,20,1\n"
        + "3,2025-01-02,1,A,30,0\n"
        + "4,2025-01-02,2,A,50,1\n"
        + "5,2025-01-03,1,B,40,0\n"
        + "6,2025-01-04,1,A,60,1\n"
        + "7,2025-01-05,1,A,70,0\n"
        + "8,2025-01-06,1,B,80,1\n"
        + "9,2025-01-07,1,A,90,0\n"
        + "10,2025-01-08,1,C,100,1\n";

    private static PreparedData Prepared(string rows, HistoryReach reach) =>
        Pdd.Create()
            .Read(CsvRowSource.FromText(rows), "rows")
            .Declare(schema => schema.Integer("t", "round").Timestamp("date").Category("key").Number("v", "y"))
            .SplitByTime("t", train: 0.6, validation: 0.2)
            .Target("y")
            .History("key", "v", reach, "date")
            .Drop("t", "date", "round", "key")
            .Build()
            .Run();

    private static double[] Read(PreparedData prepared, Part part)
    {
        var batch = prepared.Batch(part);
        var at = batch.FeatureNames.ToList().IndexOf("v_history_key");

        return [.. batch.Features.Select(row => row[at])];
    }

    [Fact]
    public void AWindowOfOnePlace_ReadsTheMeanOfTheLastPlaceBeforeARow_WholeEvenWhereTwoRowsStandOnIt()
    {
        var prepared = Prepared(Rows, new HistoryReach(0, 1));

        // Rows 1 and 2 have nothing before them, so they read the mean of every training row; 3 and 4 read day 1; 5 reads B's day 1;
        // 6 reads the last place before day 4, day 2 — both its rows, 30 and 50.
        Assert.Equal([35, 35, 10, 10, 20, 40], Read(prepared, Part.Train), new Close());
        Assert.Equal([60, 40], Read(prepared, Part.Validation), new Close());
        Assert.Equal([60, 35], Read(prepared, Part.Test), new Close());
    }

    [Fact]
    public void AWindowOfTwoPlaces_ReadsTheLastTwo_AndOneLongerThanTheHistory_ReadsAllOfIt()
    {
        Assert.Equal([35, 35, 10, 10, 20, 30], Read(Prepared(Rows, new HistoryReach(0, 2)), Part.Train), new Close());
        Assert.Equal([35, 35, 10, 10, 20, 30], Read(Prepared(Rows, new HistoryReach(0, 50)), Part.Train), new Close());
    }

    [Fact]
    public void AWeightPullsAWindowTowardsTheTrainingRowsMean_OverTheRowsItHolds()
    {
        // The window of one place before row 6 holds 30 and 50: (80 + 2 x 35) / (2 + 2).
        Assert.Equal([35, 35, (10 + 70) / 3.0, (10 + 70) / 3.0, (20 + 70) / 3.0, (80 + 70) / 4.0], Read(Prepared(Rows, new HistoryReach(2, 1)), Part.Train), new Close());
    }

    [Fact]
    public void WithNoWindow_TheHistoryIsEveryEarlierRow_AsItWas()
    {
        var plain = Pdd.Create()
            .Read(CsvRowSource.FromText(Rows), "rows")
            .Declare(schema => schema.Integer("t", "round").Timestamp("date").Category("key").Number("v", "y"))
            .SplitByTime("t", train: 0.6, validation: 0.2)
            .Target("y")
            .History("key", "v", 2, "date")
            .Drop("t", "date", "round", "key")
            .Build()
            .Run();

        Assert.Equal(Read(plain, Part.Train), Read(Prepared(Rows, new HistoryReach(2, 0)), Part.Train));
        Assert.Equal(plain.ToJson(), Prepared(Rows, new HistoryReach(2, 0)).ToJson());
    }

    [Fact]
    public void ARowServedAlone_ReadsTheLastPlacesOfTheTrainingRowsBeforeIt()
    {
        var prepared = Prepared(Rows, new HistoryReach(0, 1));
        var served = prepared.Served(CsvRowSource.FromText("t,date,round,key,v,y\n11,2025-01-09,1,A,0,\n12,2025-01-09,1,Z,0,\n"));

        Assert.Equal([60, 35], served.Features.Select(row => row[served.FeatureNames.ToList().IndexOf("v_history_key")]), new Close());
    }

    [Fact]
    public void TheRowsOfOnePlace_AreOnePlaceWhateverOrderTheyAreHandedIn()
    {
        var swapped = Rows.Replace("3,2025-01-02,1,A,30,0\n4,2025-01-02,2,A,50,1\n", "4,2025-01-02,2,A,50,1\n3,2025-01-02,1,A,30,0\n", StringComparison.Ordinal);

        Assert.NotEqual(Rows, swapped);
        Assert.Equal(
            Read(Prepared(Rows, new HistoryReach(0, 1)), Part.Train).Order(),
            Read(Prepared(swapped, new HistoryReach(0, 1)), Part.Train).Order());
    }

    [Fact]
    public void ThePlaceIsSummedRowByRow_NotByTheDifferenceOfTwoRunningTotals()
    {
        // 1e16 + 1 is 1e16 in a double, so the running total of the rows up to the second is no more than the first, and the
        // difference of the two says the second row held nothing. Added row by row the place holds its 1.
        const string Big =
            "t,date,round,key,v,y\n"
            + "1,2025-01-01,1,A,1e16,0\n"
            + "2,2025-01-02,1,A,1,1\n"
            + "3,2025-01-03,1,A,-1e16,0\n"
            + "4,2025-01-04,1,A,3,1\n"
            + "5,2025-01-05,1,A,5,0\n"
            + "6,2025-01-06,1,A,7,1\n"
            + "7,2025-01-07,1,A,9,0\n"
            + "8,2025-01-08,1,A,11,1\n"
            + "9,2025-01-09,1,A,13,0\n"
            + "10,2025-01-10,1,A,15,1\n";
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText(Big), "rows")
            .Declare(schema => schema.Integer("t", "round").Timestamp("date").Category("key").Number("v", "y"))
            .SplitByTime("t", train: 0.6, validation: 0.2)
            .Target("y")
            .History("key", "v", new HistoryReach(0, 1), "date")
            .Drop("t", "date", "round", "key")
            .Build()
            .Run();

        // Row 3 reads day 2 alone: 1. Row 4 reads day 3: -1e16. Row 5 reads day 4: 3.
        var read = Read(prepared, Part.Train);

        Assert.Equal(1.0, read[2]);
        Assert.Equal(-1e16, read[3]);
        Assert.Equal(3.0, read[4]);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(-1, 1)]
    [InlineData(double.NaN, 1)]
    [InlineData(1, -1)]
    public void ANoWindowAndNoWeight_OrANumberThatMeansNothing_IsRefusedWhereTheStepIsMade(double shrink, int window)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HistoryStep("key", "v", ["date"], new HistoryReach(shrink, window)));
    }

    [Fact]
    public void AWindow_IsWrittenAndReadBack_AndLeftOutWhenThereIsNone()
    {
        var windowed = Declared(new HistoryReach(0, 3));
        var plain = Declared(new HistoryReach(2, 0));

        Assert.Contains("\"window\": 3", windowed.ToJson(), StringComparison.Ordinal);
        Assert.DoesNotContain("window", plain.ToJson(), StringComparison.Ordinal);
        Assert.Equal(windowed.ToJson(), PipelineDeclaration.FromJson(windowed.ToJson(), StepCatalog.BuiltIn()).ToJson());
        Assert.Equal(3, Assert.IsType<HistoryStep>(PipelineDeclaration.FromJson(windowed.ToJson(), StepCatalog.BuiltIn()).Steps.Single(step => step is HistoryStep)).Window);
        Assert.Equal(0, Assert.IsType<HistoryStep>(PipelineDeclaration.FromJson(plain.ToJson(), StepCatalog.BuiltIn()).Steps.Single(step => step is HistoryStep)).Window);
    }

    [Fact]
    public void TwoStepsAreTheSame_OnlyWhenTheirWindowsAre()
    {
        Assert.Equal(new HistoryStep("key", "v", ["date"], new HistoryReach(0, 3)), new HistoryStep("key", "v", ["date"], new HistoryReach(0, 3)));
        Assert.NotEqual(new HistoryStep("key", "v", ["date"], new HistoryReach(0, 3)), new HistoryStep("key", "v", ["date"], new HistoryReach(0, 4)));
    }

    private static PipelineDeclaration Declared(HistoryReach reach) =>
        Pdd.Create()
            .Read(CsvRowSource.FromText(Rows), "rows")
            .Declare(schema => schema.Integer("t", "round").Timestamp("date").Category("key").Number("v", "y"))
            .SplitByTime("t", train: 0.6, validation: 0.2)
            .Target("y")
            .History("key", "v", reach, "date")
            .Drop("t", "date", "round", "key")
            .Build()
            .Declaration;

    private sealed class Close : IEqualityComparer<double>
    {
        public bool Equals(double left, double right) => Math.Abs(left - right) < 1e-9;

        public int GetHashCode(double value) => 0;
    }
}
