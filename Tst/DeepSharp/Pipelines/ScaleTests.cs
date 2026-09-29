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

        Assert.Equal(-1, training.Min(), 12);
        Assert.Equal(1, training.Max(), 12);

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

        Assert.Contains("outside the range this pipeline was fitted on", refused.Message, StringComparison.Ordinal);
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
}
