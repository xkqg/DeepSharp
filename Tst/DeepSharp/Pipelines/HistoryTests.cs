// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What the earlier rows of the same kind held, as one number: the mean of a column over the training rows of a row's key that
/// came strictly before it, pulled towards the mean of every training row by a weight the person says — the m-estimate a type
/// history of flocks is. Learned from the training rows alone, it is replayed on every other row and on one served row as the
/// same number, and no row ever reads its own value or a later one.
/// </summary>
/// <remarks>
/// The rows, in the order of their dates, and what each reads with a weight of two, the training rows' mean being 35:
/// training — 1 A 10: nothing earlier, 35; 2 B 20: 35; 3 A 30: row 1, (10 + 70) / 3; 4 A 50, the same date as row 3, which is
/// not earlier: row 1 alone, (10 + 70) / 3; 5 B 40: row 2, (20 + 70) / 3; 6 A 60: rows 1, 3 and 4, (90 + 70) / 5. Validation
/// and test read the training rows before them — 7 A: (150 + 70) / 6; 8 B: (60 + 70) / 4; 9 A: (150 + 70) / 6 — and 10, of a
/// key no training row holds, the mean, 35.
/// </remarks>
public class HistoryTests
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

    private static PreparedData Prepared(string rows = Rows, params string[] orderBy) =>
        Pdd.Create()
            .Read(CsvRowSource.FromText(rows), "rows")
            .Declare(schema => schema.Integer("t", "round").Timestamp("date").Category("key").Number("v", "y"))
            .SplitByTime("t", train: 0.6, validation: 0.2)
            .Target("y")
            .History("key", "v", 2, orderBy.Length > 0 ? orderBy : ["date"])
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
    public void ATrainingRow_ReadsTheTrainingRowsOfItsKeyStrictlyBeforeIt_PulledTowardsTheirMean()
    {
        Assert.Equal([35, 35, 80 / 3.0, 80 / 3.0, 30, 32], Read(Prepared(), Part.Train), new Close());
    }

    [Fact]
    public void ARowOutsideTheTrainingRows_ReadsTheTrainingRowsBeforeIt_AndAKeyTheyNeverHeld_ReadsTheirMean()
    {
        var prepared = Prepared();

        Assert.Equal([220 / 6.0, 130 / 4.0], Read(prepared, Part.Validation), new Close());
        Assert.Equal([220 / 6.0, 35], Read(prepared, Part.Test), new Close());
        Assert.Equal(35, prepared.Fitted.Single(each => each.Value.Numbers.ContainsKey("prior")).Value.Number("prior"));
    }

    [Fact]
    public void ALaterRowOfAKey_ChangesNoEarlierRowOfIt_ButThroughTheMeanOfEveryTrainingRow()
    {
        // Row 6, the last training row of A, made a hundred times larger: rows 1 to 5 read what they read, but for the one
        // mean every training row shares, which is learned from them all as a scale's mean is.
        var changed = Rows.Replace("6,2025-01-04,1,A,60,1", "6,2025-01-04,1,A,6000,1", StringComparison.Ordinal);
        const double Prior = (10 + 20 + 30 + 50 + 40 + 6000) / 6.0;

        Assert.Equal([Prior, Prior, (10 + (2 * Prior)) / 3, (10 + (2 * Prior)) / 3, (20 + (2 * Prior)) / 3], Read(Prepared(changed), Part.Train).Take(5), new Close());
    }

    [Fact]
    public void ARowTheModelIsChosenOrMeasuredOn_ChangesNoTrainingRow()
    {
        var changed = Rows.Replace("7,2025-01-05,1,A,70,0", "7,2025-01-05,1,A,7000,0", StringComparison.Ordinal);

        Assert.Equal(Read(Prepared(), Part.Train), Read(Prepared(changed), Part.Train), new Close());
    }

    [Fact]
    public void SeveralColumnsOrderTheRows_SoARowOfTheSameDateButALaterRound_ComesAfter()
    {
        // Ordered by the date and then the round, row 4 comes after row 3, and reads rows 1 and 3: (10 + 30 + 70) / 4.
        Assert.Equal([35, 35, 80 / 3.0, 110 / 4.0], Read(Prepared(Rows, "date", "round"), Part.Train).Take(4), new Close());
    }

    [Fact]
    public void OneRowServedAlone_ReadsWhatTheSameRowReadsInTheSet()
    {
        var again = PreparedData.FromJson(Prepared().ToJson(), StepCatalog.BuiltIn());
        var served = again.Served(CsvRowSource.FromText("t,date,round,key,v,y\n11,2025-01-09,1,A,0,\n"));

        Assert.Equal(220 / 6.0, served.Features[0][served.FeatureNames.ToList().IndexOf("v_history_key")], 9);
    }

    [Fact]
    public void ARowWithAGapWhereItsOrderIsSaid_IsRefused_NamingTheRow()
    {
        var gap = Rows.Replace("5,2025-01-03,1,B,40,0", "5,,1,B,40,0", StringComparison.Ordinal);

        var refused = Assert.Throws<InvalidOperationException>(() => Pdd.Create()
            .Read(CsvRowSource.FromText(gap), "rows")
            .Declare(schema => schema.Integer("t", "round").Optional("date", ColumnKind.Timestamp).Category("key").Number("v", "y"))
            .SplitByTime("t", train: 0.6, validation: 0.2)
            .Target("y")
            .History("key", "v", 2, "date")
            .Drop("t", "date", "round", "key")
            .Build()
            .Run());

        Assert.Contains("Row 5", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void AWeightOfNothingOrBelow_IsRefusedWhereTheStepIsMade(double shrink) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new HistoryStep("key", "v", ["date"], shrink));

    [Fact]
    public void TheWeight_HasNoDefault_AndTheStepIsWrittenAndReadBackAsItWas()
    {
        var description = StepCatalog.BuiltIn().Describe(HistoryStep.Name);
        var prepared = Prepared();

        Assert.Contains("shrink", description.Parameters.SelectMany(parameter => parameter.RequiredKeys));
        Assert.Equal(prepared.ToJson(), PreparedData.FromJson(prepared.ToJson(), StepCatalog.BuiltIn()).ToJson());
    }

    [Fact]
    public void AWholeNumberOrANumber_OrdersTheRows_AsADateDoes()
    {
        // t numbers the rows and v grows with them on every key, so both put row 4 after row 3, as the date and the round do.
        double[] expected = [35, 35, 80 / 3.0, 110 / 4.0, 30, 32];

        Assert.Equal(expected, Read(Prepared(Rows, "t"), Part.Train), new Close());
        Assert.Equal(expected, Read(Prepared(Rows, "v"), Part.Train), new Close());
    }

    [Fact]
    public void AColumnOfWords_HasNoOrderToPutRowsIn_AndIsRefusedWhereThePipelineIsDeclared()
    {
        var refused = Assert.Throws<DeclarationException>(() => Prepared(Rows, "key"));

        Assert.Contains("'key'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoStepsAreTheSame_WhenTheySayTheSame()
    {
        var step = new HistoryStep("key", "v", ["date"], 2);

        Assert.Equal(step, new HistoryStep("key", "v", ["date"], 2));
        Assert.Equal(step.GetHashCode(), new HistoryStep("key", "v", ["date"], 2).GetHashCode());
        Assert.NotEqual(step, new HistoryStep("key", "v", ["date", "round"], 2));
        Assert.NotEqual(step, new HistoryStep("key", "v", ["date"], 3));
        Assert.NotEqual(step, new HistoryStep("other", "v", ["date"], 2));
        Assert.False(step.Equals(null));
    }

    private sealed class Close : IEqualityComparer<double>
    {
        public bool Equals(double left, double right) => Math.Abs(left - right) < 1e-9;

        public int GetHashCode(double value) => 0;
    }
}
