// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A number written down as how far it fills each of a few bins, Gorishniy, Rubachev and Babenko's piecewise linear encoding:
/// the edges are the quantiles of the training rows alone, a value fills every bin below it and the share of its own bin it
/// reaches, and the bins together are the value again. Learned from the training rows, it is replayed on every other row and
/// on a served one, and a value beyond the edges is what the step says of it: let through the end bins, held at them, or refused.
/// </summary>
public class PiecewiseLinearTests
{
    // A hundred rows in time: the first seventy, the training rows, hold the numbers one to seventy; the rows after them ten
    // times their place, far beyond anything the training rows hold.
    private static readonly string Rows = MadeRows(t => t <= 70 ? t : t * 10);

    private static PreparedData Prepared(string rows = "", int bins = 4, OutOfRange outOfRange = OutOfRange.Pass) =>
        Pdd.Create()
            .Read(CsvRowSource.FromText(rows.Length > 0 ? rows : Rows), "rows")
            .Declare(schema => schema.Integer("t").Number("x", "y"))
            .SplitByTime("t", train: 0.70, validation: 0.15)
            .Target("y")
            .Drop("t")
            .PiecewiseLinear("x", bins, outOfRange)
            .Build()
            .Run();

    [Fact]
    public void TheEdges_AreTheQuantilesOfTheTrainingRowsAlone()
    {
        var prepared = Prepared();

        Assert.Equal([1, 18.25, 35.5, 52.75, 70], prepared.Fitted.Last().Value.Curve("edges"));
    }

    [Fact]
    public void AValue_FillsEveryBinBelowIt_AndItsOwnBinAsFarAsItReaches_AndTheBinsTogetherAreTheValueAgain()
    {
        var batch = Prepared().Batch(Part.Train);
        double[] edges = [1, 18.25, 35.5, 52.75, 70];

        Assert.Equal(["x_bin1", "x_bin2", "x_bin3", "x_bin4"], batch.FeatureNames);

        for (var row = 0; row < batch.Features.Count; row++)
        {
            var bins = batch.Features[row];

            Assert.All(bins, share => Assert.InRange(share, 0, 1));
            Assert.Equal(row + 1, Rebuilt(bins, edges), 9);
        }

        // Thirty sits in the second bin: the first full, the second three fifths of the way, the rest empty.
        Assert.Equal([1, (30 - 18.25) / 17.25, 0, 0], batch.Features[29], new Close());
    }

    [Fact]
    public void AValueBeyondTheEdges_IsLetThroughTheEndBins_SoTheBinsAreStillTheValue()
    {
        var batch = Prepared().Batch(Part.Test);
        double[] edges = [1, 18.25, 35.5, 52.75, 70];

        Assert.All(batch.Features, bins => Assert.True(bins[3] > 1));
        Assert.Equal(860, Rebuilt(batch.Features[0], edges), 9);
    }

    [Fact]
    public void AValueBeyondTheEdges_IsHeldAtThem_WhenClipped_AndRefusedNamingItsRow_WhenRefused()
    {
        Assert.All(Prepared(outOfRange: OutOfRange.Clip).Batch(Part.Test).Features, bins => Assert.Equal([1, 1, 1, 1], bins));

        var refused = Assert.Throws<InvalidOperationException>(() => Prepared(outOfRange: OutOfRange.Refuse));

        Assert.Contains("Row 71", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'x'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EdgesThatFallTogether_AreOne_SoNoBinIsEmptyOnEveryRow()
    {
        // Half the training rows hold one: the first quantiles fall together there, and the bins are fewer than asked for.
        var prepared = Prepared(MadeRows(t => t <= 35 ? 1 : t), bins: 4);
        var edges = prepared.Fitted.Last().Value.Curve("edges");

        Assert.Equal(edges.Distinct().Count(), edges.Count);
        Assert.Equal(edges.Count - 1, prepared.Batch(Part.Train).FeatureNames.Count);
        Assert.Equal("x_bin1", prepared.Batch(Part.Train).FeatureNames[0]);
    }

    [Fact]
    public void AColumnThatHoldsOneValueOnItsTrainingRows_HasNoEdgesToPlace_AndIsRefused()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Prepared(MadeRows(t => t <= 70 ? 5 : t)));

        Assert.Contains("'x'", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    public void FewerThanTwoBins_AreRefusedWhereTheStepIsMade(int bins) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new PiecewiseLinearStep("x", bins));

    [Fact]
    public void TheStepAndWhatItLearned_AreWrittenAndReadBack_AndAServedRowIsEncodedAsTheSameRowInTheSet()
    {
        var prepared = Prepared();
        var again = PreparedData.FromJson(prepared.ToJson(), StepCatalog.BuiltIn());
        var served = again.Served(CsvRowSource.FromText("t,x,y\n30,30,0\n"));

        Assert.Equal(prepared.ToJson(), again.ToJson());
        Assert.Contains("bins", StepCatalog.BuiltIn().Describe(PiecewiseLinearStep.Name).Parameters.SelectMany(parameter => parameter.RequiredKeys));
        Assert.Equal(prepared.Batch(Part.Train).Features[29], served.Features[0]);
    }

    [Fact]
    public void AGap_StaysAGapInEveryBin_AndARefusingStepPassesEveryValueWithinTheEdges()
    {
        // The rows after the training rows hold what the training rows held, so nothing lies beyond the edges, and row 80 has
        // no x at all: a gap is no value beyond the edges, and the bins of a row without one hold none either.
        var rows = MadeRows(t => t <= 70 ? t : t - 70).Replace("\n80,10,3\n", "\n80,,3\n", StringComparison.Ordinal);
        var table = Pdd.Create()
            .Read(CsvRowSource.FromText(rows), "rows")
            .Declare(schema => schema.Integer("t").Optional("x", ColumnKind.Number).Number("y"))
            .SplitByTime("t", train: 0.70, validation: 0.15)
            .Target("y")
            .PiecewiseLinear("x", 4, OutOfRange.Refuse)
            .Build()
            .Run()
            .Table;
        var gaps = Enumerable.Range(0, table.RowCount).Where(row => table["x_bin1"].IsMissing(row)).ToArray();

        Assert.Single(gaps);
        Assert.All(["x_bin2", "x_bin3", "x_bin4"], name => Assert.True(table[name].IsMissing(gaps[0])));
        Assert.Equal((long?)80, ((Column<long>)table["t"])[gaps[0]]);
    }

    private static double Rebuilt(double[] bins, double[] edges) =>
        edges[0] + bins.Select((share, at) => share * (edges[at + 1] - edges[at])).Sum();

    private static string MadeRows(Func<int, double> x)
    {
        var text = new StringBuilder("t,x,y\n");

        for (var t = 1; t <= 100; t++)
        {
            text.Append(CultureInfo.InvariantCulture, $"{t},{x(t)},{t % 7}\n");
        }

        return text.ToString();
    }

    private sealed class Close : IEqualityComparer<double>
    {
        public bool Equals(double left, double right) => Math.Abs(left - right) < 1e-9;

        public int GetHashCode(double value) => 0;
    }
}
