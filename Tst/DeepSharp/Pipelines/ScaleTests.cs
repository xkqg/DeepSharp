// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Each scale says where the training rows land, and what happens to a value outside the range the fit learned is
/// declared beside it. A pair the scale cannot honour is refused where it is written: a scale with no range cannot hold a
/// value in one, and a rank cannot let a value through that it has no place for. The midrange scale lands the training
/// rows between minus one and one, as a network takes them, and comes back by the numbers it learned.
/// </summary>
public class ScaleTests
{
    // The published price series, divided in time: the first 354 days train, and the 152 after them the fit never saw.
    private static PreparedData Prices(Action<FittingBuilder> scaled)
    {
        var fitting = Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close", "AAPL.Volume"))
            .SplitByTime("Date", 0.70, 0.15);

        scaled(fitting);

        return fitting.Build().Run();
    }

    private static double[] Values(PreparedData prepared, string column, Func<Part, bool> which) =>
        [.. Enumerable.Range(0, prepared.Table.RowCount).Where(row => which(prepared.Parts[row])).Select(row => ((Column<double>)prepared.Table[column])[row]!.Value)];

    private static double AtReadAt(PreparedData prepared, string column, int readAt) =>
        ((Column<double>)prepared.Table[column])[Enumerable.Range(0, prepared.Table.RowCount).Single(row => prepared.Table.Identities[row].ReadAt == readAt)]!.Value;

    [Theory]
    [InlineData(Scale.Standard, OutOfRange.Pass, true)]
    [InlineData(Scale.Standard, OutOfRange.Clip, false)]
    [InlineData(Scale.Standard, OutOfRange.Refuse, false)]
    [InlineData(Scale.Robust, OutOfRange.Pass, true)]
    [InlineData(Scale.Robust, OutOfRange.Clip, false)]
    [InlineData(Scale.Robust, OutOfRange.Refuse, false)]
    [InlineData(Scale.Power, OutOfRange.Pass, true)]
    [InlineData(Scale.Power, OutOfRange.Clip, false)]
    [InlineData(Scale.Power, OutOfRange.Refuse, false)]
    [InlineData(Scale.Quantile, OutOfRange.Pass, false)]
    [InlineData(Scale.Quantile, OutOfRange.Clip, true)]
    [InlineData(Scale.Quantile, OutOfRange.Refuse, true)]
    [InlineData(Scale.MinMax, OutOfRange.Pass, true)]
    [InlineData(Scale.MinMax, OutOfRange.Clip, true)]
    [InlineData(Scale.MinMax, OutOfRange.Refuse, true)]
    [InlineData(Scale.MaxAbs, OutOfRange.Pass, true)]
    [InlineData(Scale.MaxAbs, OutOfRange.Clip, true)]
    [InlineData(Scale.MaxAbs, OutOfRange.Refuse, true)]
    [InlineData(Scale.MidRange, OutOfRange.Pass, true)]
    [InlineData(Scale.MidRange, OutOfRange.Clip, true)]
    [InlineData(Scale.MidRange, OutOfRange.Refuse, true)]
    public void EveryScaleAndWhatHappensOutsideItsRange_DoesWhatItSays_OrIsRefusedWhereItIsWritten(Scale scale, OutOfRange outOfRange, bool kept)
    {
        var written = $$"""{"step": "normalise", "column": "a", "scale": "{{scale.ToString().ToLowerInvariant()}}", "outOfRange": "{{outOfRange.ToString().ToLowerInvariant()}}"}""";

        if (kept)
        {
            Assert.Equal(new NormaliseStep("a", scale, outOfRange), StepCatalog.BuiltIn().ReadStep(written));

            return;
        }

        var refused = Assert.Throws<ArgumentException>(() => new NormaliseStep("a", scale, outOfRange));

        Assert.Contains(scale.ToString().ToLowerInvariant(), refused.Message, StringComparison.Ordinal);
        Assert.Throws<PipelineFileException>(() => StepCatalog.BuiltIn().ReadStep(written));
    }

    [Fact]
    public void TheMidRangeScale_LandsTheTrainingRowsBetweenMinusOneAndOne_AndComesBackByWhatItLearned()
    {
        var prepared = Prices(fitting => fitting.Normalise("AAPL.Close", Scale.MidRange).Normalise("AAPL.Volume", Scale.MidRange));

        // Measured in the row walk: the second day's close of 128.720001 and volume of 44,891,700.
        Assert.Equal(0.7993437, AtReadAt(prepared, "AAPL.Close", 1), 1e-7);
        Assert.Equal(-0.5730045, AtReadAt(prepared, "AAPL.Volume", 1), 1e-7);

        var training = Values(prepared, "AAPL.Close", part => part == Part.Train);

        // Exactly, not to twelve places: the training minimum and maximum are the ends of the range this
        // scale learned, and a value within the rounding building centre and spread apart carries lands on
        // the end itself rather than a hair beyond it (measured before this landed: 1.0000000000000004).
        Assert.Equal(-1, training.Min());
        Assert.Equal(1, training.Max());

        // After the training days, prices and volumes move on: four closes and one volume of the 152 land outside.
        Assert.Equal(152, Values(prepared, "AAPL.Close", part => part != Part.Train).Length);
        Assert.Equal(4, Values(prepared, "AAPL.Close", part => part != Part.Train).Count(value => Math.Abs(value) > 1));
        Assert.Equal(1, Values(prepared, "AAPL.Volume", part => part != Part.Train).Count(value => Math.Abs(value) > 1));

        var step = (NormaliseStep)prepared.Declaration.Steps[3];

        Assert.Equal(128.720001, step.Undo(AtReadAt(prepared, "AAPL.Close", 1), prepared.Fitted[3]), 6);
    }

    [Fact]
    public void TheMidRangeScale_HoldsALaterValueAtMinusOneOrOne_OrRefusesIt()
    {
        var clipped = Prices(fitting => fitting.Normalise("AAPL.Close", Scale.MidRange, OutOfRange.Clip));
        var later = Values(clipped, "AAPL.Close", part => part != Part.Train);

        Assert.All(later, value => Assert.InRange(value, -1, 1));
        Assert.Equal(4, later.Count(value => Math.Abs(value) == 1));

        var refused = Assert.Throws<InvalidOperationException>(() => Prices(fitting => fitting.Normalise("AAPL.Close", Scale.MidRange, OutOfRange.Refuse)));

        // Row 503 (2017-02-13), the first close the training range never reached -- not row 5, the training
        // maximum itself (2015-02-23), which the rounding building centre and spread apart once refused.
        Assert.Contains("Row 503 of 'AAPL.Close'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("outside the range this pipeline was fitted on", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(Scale.MinMax)]
    [InlineData(Scale.MaxAbs)]
    [InlineData(Scale.MidRange)]
    public void EveryBoundedScale_LandsItsTrainingMaximumExactlyOnTheEndItReaches(Scale scale)
    {
        var prepared = Prices(fitting => fitting.Normalise("AAPL.Close", scale));
        var training = Values(prepared, "AAPL.Close", part => part == Part.Train);

        // AAPL.Close never goes below nought, so all three place the training maximum on the same end:
        // min-max and midrange because it is the top of their range, max-abs because the largest close is
        // also the largest magnitude.
        Assert.Equal(1, training.Max());
    }

    [Fact]
    public void TheMinMaxScale_LandsItsTrainingMinimumExactlyOnNothing()
    {
        var prepared = Prices(fitting => fitting.Normalise("AAPL.Close", Scale.MinMax));
        var training = Values(prepared, "AAPL.Close", part => part == Part.Train);

        Assert.Equal(0, training.Min());
    }

    [Fact]
    public void TheMaxAbsScale_LandsATrainingExtremeBelowNothingExactlyOnMinusOne()
    {
        // AAPL.Close never reaches max-abs's lower end, so the low side is checked on its own column: the
        // largest magnitude here is the -10, and it comes back exactly where a value divided by its own
        // magnitude always does.
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]),
            CsvRowSource.FromText("a\n3\n-10\n5\n-2\n"));
        var parts = new[] { Part.Train, Part.Train, Part.Train, Part.Train };
        var step = new NormaliseStep("a", Scale.MaxAbs);

        step.ApplyTo(table, step.Fit(table, parts));

        Assert.Equal(-1, ((Column<double>)table["a"])[1]);
    }

    [Fact]
    public void AQuantileScale_HoldsAValueBeyondTheTrainingRowsAtTheEdge_OrRefusesIt()
    {
        var held = Prices(fitting => fitting.Normalise("AAPL.Close", Scale.Quantile, OutOfRange.Clip));

        Assert.Equal(1, Values(held, "AAPL.Close", part => part != Part.Train).Max());

        var refused = Assert.Throws<InvalidOperationException>(() => Prices(fitting => fitting.Normalise("AAPL.Close", Scale.Quantile, OutOfRange.Refuse)));

        Assert.Contains("outside the range this pipeline was fitted on", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AQuantileScale_RefusesAValueBelowTheTrainingRowsToo_AndRanksAmongRepeatedOnes()
    {
        static Table Rows() =>
            SchemaBinding.Bind(new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]), CsvRowSource.FromText("a\n1\n1\n1\n2\n-5\n\n"));

        Part[] parts = [Part.Train, Part.Train, Part.Train, Part.Train, Part.Test, Part.Test];
        var refusing = new NormaliseStep("a", Scale.Quantile, OutOfRange.Refuse);
        var holding = new NormaliseStep("a", Scale.Quantile, OutOfRange.Clip);
        var table = Rows();

        Assert.Contains("Row 5 of 'a'", Assert.Throws<InvalidOperationException>(() => refusing.ApplyTo(Rows(), refusing.Fit(Rows(), parts))).Message, StringComparison.Ordinal);

        holding.ApplyTo(table, holding.Fit(table, parts));

        Assert.Equal([0, 0, 0, 1, 0], Enumerable.Range(0, 5).Select(row => ((Column<double>)table["a"])[row]!.Value));
        Assert.True(table["a"].IsMissing(5));
    }

    [Fact]
    public void WhereEachScaleLandsTheTrainingRows_IsOneRule()
    {
        Assert.Equal(
            [null, Form.Unit, Form.Signed, null, Form.Unit, null, Form.Signed],
            Enum.GetValues<Scale>().Select(scale => scale.Lands()));
    }

    // A column the training rows hold one value of, 0.1, which no double writes exactly.
    private static PreparedData Constant(Scale scale) => Pdd.Create()
        .Read(CsvRowSource.FromText("id,x\n" + string.Join("\n", Enumerable.Repeat("1,0.1", 200)) + "\n"), "a column of one value")
        .Declare(schema => schema.Integer("id").Number("x"))
        .SplitStratified("id", train: 0.99, validation: 0.005)
        .Normalise("x", scale)
        .Build()
        .Run();

    [Theory]
    [InlineData(Scale.Standard)]
    [InlineData(Scale.Robust)]
    [InlineData(Scale.Power)]
    [InlineData(Scale.MinMax)]
    [InlineData(Scale.MidRange)]
    public void AColumnTheTrainingRowsHoldOneValueOf_IsCentredOnIt_AndALaterValueIsNotBlownUpByASpreadOfRoundingAlone(Scale scale)
    {
        // Averaged, two hundred of 0.1 are 0.10000000000000007, and the standard deviation around that came out as
        // 6.9e-17 rather than nothing: every training row became -1, and a later 0.2 became 1441151880758557.8.
        var prepared = Constant(scale);
        var served = prepared.Served(CsvRowSource.FromText("id,x\n1,0.1\n1,0.2\n"));
        var x = served.FeatureNames.ToList().IndexOf("x");

        Assert.All(Values(prepared, "x", _ => true), value => Assert.Equal(0, value, 1e-9));
        Assert.Equal(0, served.Features[0][x], 1e-9);
        Assert.InRange(served.Features[1][x], 0.05, 0.2);
    }

    [Fact]
    public void ThePowerScale_KeepsTheShapingItsTrainingValuesAreMostNormalUnder_AsScipyFindsIt()
    {
        // Seventy-one small values either side of nought, seventy of which train. Over those seventy, scipy's
        // yeojohnson_normmax puts the best shaping at 0.7145, and by its own log-likelihood 0.7 is the best point of the
        // grid. The search added 0.05 at a time and reached 2 as 2.000000000000002, where shaping a negative value
        // divides by -2e-15, and kept 2.
        double[] values =
        [
            0.188, -0.063, -0.057, 0.198, 0.053, -0.125, 0.104, 0.071, 0.016, -0.06, 0.087, 0.17, 0.108, -0.157, 0.112, 0.046,
            0.079, 0.035, 0.044, 0.149, -0.068, 0.164, -0.013, -0.037, 0.005, -0.112, 0.025, -0.053, -0.063, 0.058, 0.141,
            0.142, 0.141, -0.069, 0.03, -0.101, -0.144, -0.149, -0.073, -0.159, -0.138, 0.047, 0.059, -0.107, -0.168, 0.108,
            -0.165, -0.051, -0.157, 0.068, -0.022, 0.203, -0.125, -0.122, -0.024, -0.003, 0.114, 0.094, -0.097, -0.122,
            -0.121, -0.033, 0.156, 0.132, 0.134, 0.112, 0.103, 0.044, 0.047, -0.092, -0.109,
        ];

        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText("id,x\n" + string.Join("\n", values.Select(value => "1," + value.ToString("R", System.Globalization.CultureInfo.InvariantCulture))) + "\n"), "small values either side of nought")
            .Declare(schema => schema.Integer("id").Number("x"))
            .SplitStratified("id", train: 0.99, validation: 0.005)
            .Normalise("x", Scale.Power)
            .Build()
            .Run();

        Assert.Equal(70, prepared.Parts.Count(part => part == Part.Train));
        Assert.Equal(0.7, prepared.Fitted[3].Number("lambda"));
    }
}
