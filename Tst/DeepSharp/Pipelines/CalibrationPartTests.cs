// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The rows an interval is calibrated on are a part of their own. They are not training, which fits on them; not validation, which
/// chooses between models by them and so has seen what it chose; and not test, which a number is reported on once everything else is
/// settled. Nothing is fitted on them and nothing is chosen by them. A split that sets none aside is written exactly as it was.
/// </summary>
public class CalibrationPartTests
{
    private static PreparedData Flocks(Func<PipelineBuilder, FittingBuilder> split)
    {
        var rows = Enumerable.Range(0, 100).Select(day => (IReadOnlyList<string?>)
            [new DateTime(2024, 1, 1).AddDays(day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture), (day % 7).ToString(CultureInfo.InvariantCulture), (100 + (day * 3 % 17)).ToString(CultureInfo.InvariantCulture)]).ToList();

        return split(Pdd.Create()
                .Read(new InMemoryRowSource(["Date", "x", "y"], rows), "a hundred days")
                .Declare(schema => schema.Timestamp("Date").Number("x", "y")))
            .Build()
            .Run();
    }

    private static int[] Days(PreparedData prepared, Part part) =>
        [.. Enumerable.Range(0, prepared.Table.RowCount).Where(row => prepared.Parts[row] == part).Select(row => (int)(((Column<DateTime>)prepared.Table["Date"])[row]!.Value - new DateTime(2024, 1, 1)).TotalDays).Order()];

    [Fact]
    public void ThePartsOfASplit_HoldCalibrationRowsOfTheirOwn_BeforeTheTestRowsInATimeOrder()
    {
        var prepared = Flocks(builder => builder.Calibrate(0.10).SplitByTime("Date", 0.60, 0.10));

        // 60 to learn from, 10 to choose with, 10 to calibrate on, and the 20 most recent to be measured on.
        Assert.Equal(60, prepared.CountIn(Part.Train));
        Assert.Equal(10, prepared.CountIn(Part.Validation));
        Assert.Equal(10, prepared.CountIn(Part.Calibration));
        Assert.Equal(20, prepared.CountIn(Part.Test));
        Assert.Equal(Enumerable.Range(70, 10), Days(prepared, Part.Calibration));
        Assert.Equal(Enumerable.Range(80, 20), Days(prepared, Part.Test));
    }

    [Fact]
    public void NothingIsFittedOnTheCalibrationRows_ASuddenValueThereMovesNoScale()
    {
        // The same hundred days, but for one: x is a thousand on a day that lands in the calibration rows. A scale learned from the
        // training rows alone cannot have seen it.
        var rows = Enumerable.Range(0, 100).Select(day => (IReadOnlyList<string?>)
            [new DateTime(2024, 1, 1).AddDays(day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture), (day == 75 ? 1000 : day % 7).ToString(CultureInfo.InvariantCulture), "1"]).ToList();

        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["Date", "x", "y"], rows), "a hundred days")
            .Declare(schema => schema.Timestamp("Date").Number("x", "y"))
            .Calibrate(0.10)
            .SplitByTime("Date", 0.60, 0.10)
            .Normalise("x", Scale.MidRange)
            .Build()
            .Run();

        Assert.Contains(75, Days(prepared, Part.Calibration));

        var train = Enumerable.Range(0, prepared.Table.RowCount).Where(row => prepared.Parts[row] == Part.Train).Select(row => ((Column<double>)prepared.Table["x"])[row]!.Value).ToArray();

        // Mid-range scaling lands the training rows between minus one and one exactly, so what it learned came from them alone.
        Assert.Equal(-1, train.Min(), 9);
        Assert.Equal(1, train.Max(), 9);
    }

    [Fact]
    public void ARandomSplit_DealsTheCalibrationRowsByTheTestSeed_SoTheyAreTheSameOnEveryDeal()
    {
        PreparedData Deal(int seed) =>
            Flocks(builder => builder.Calibrate(0.10).SplitAtRandom(0.60, 0.10, seed, testSeed: 7));

        var first = Deal(1);
        var second = Deal(2);

        Assert.Equal(10, first.CountIn(Part.Calibration));
        Assert.Equal(Days(first, Part.Calibration), Days(second, Part.Calibration));
        Assert.Equal(Days(first, Part.Test), Days(second, Part.Test));
        Assert.NotEqual(Days(first, Part.Train), Days(second, Part.Train));
    }

    [Fact]
    public void AStratifiedSplit_KeepsTheMixtureInTheCalibrationRowsToo()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass"))
            .Calibrate(0.15)
            .SplitStratified("survived", 0.60, 0.10)
            .Build()
            .Run();

        double Share(Part part) =>
            Enumerable.Range(0, prepared.Table.RowCount).Where(row => prepared.Parts[row] == part).Average(row => ((Column<long>)prepared.Table["survived"])[row]!.Value);

        Assert.InRange(prepared.CountIn(Part.Calibration), 131, 136);
        Assert.Equal(Share(Part.Train), Share(Part.Calibration), 0.02);
    }

    [Fact]
    public void TheFitWritesHowManyRowsTheCalibrationHolds_ButOnlyWhenItHoldsAny()
    {
        var written = JsonDocument.Parse(Flocks(builder => builder.Calibrate(0.10).SplitByTime("Date", 0.60, 0.10)).ToJson()).RootElement.ToString();
        var plain = JsonDocument.Parse(Flocks(builder => builder.SplitByTime("Date", 0.60, 0.10)).ToJson()).RootElement.ToString();

        Assert.Contains("rows.calibration", written, StringComparison.Ordinal);
        Assert.DoesNotContain("calibration", plain, StringComparison.Ordinal);
    }

    [Fact]
    public void ASplitThatSetsNoneAside_IsWrittenExactlyAsItWas_AndOneThatDoesNamesIt()
    {
        var none = new SplitAtRandomStep(SplitShares.Of(0.7, 0.15), 1);
        var some = new SplitAtRandomStep(SplitShares.Of(0.6, 0.15, 0, calibration: 0.1), 1);

        Assert.DoesNotContain("calibration", Encoding.UTF8.GetString(none.Canonical()), StringComparison.Ordinal);
        Assert.Contains("\"calibration\":0.1", Encoding.UTF8.GetString(some.Canonical()), StringComparison.Ordinal);
        Assert.Equal(some, SplitAtRandomStep.ReadFrom(JsonDocument.Parse(some.Canonical()).RootElement));
        Assert.Equal(0.15, some.Shares.Test, 12);
    }

    [Fact]
    public void ShareForCalibration_IsHeldToTheRulesOfEveryShare()
    {
        Assert.Throws<ArgumentException>(() => SplitShares.Of(0.6, 0.2, 0, calibration: 0.2));
        Assert.Throws<ArgumentOutOfRangeException>(() => SplitShares.Of(0.6, 0.1, 0, calibration: double.NaN));
        Assert.Throws<InvalidOperationException>(() => Pdd.Create().ReadCsv(Repository.Data("titanic.csv")).Calibrate(0.1).Build());
        Assert.Throws<ArgumentOutOfRangeException>(() => Pdd.Create().Calibrate(0));
        Assert.Throws<InvalidOperationException>(() => Pdd.Create().Calibrate(0.1).Calibrate(0.1));
    }

    [Fact]
    public void EveryPartHasAStandingOfTheSameName_SoTheNotebookAndTheEvidenceNeverForgetOne()
    {
        Assert.All(Enum.GetNames<Part>(), name => Assert.True(Enum.IsDefined(typeof(Standing), name), $"no standing '{name}'"));
        Assert.Equal("calibration", Part.Calibration.Word());
    }

    [Fact]
    public void TheLikenessOfTheParts_LooksAtTheCalibrationRowsToo_WhenThereAreAny()
    {
        var prepared = Flocks(builder => builder.Calibrate(0.10).SplitByTime("Date", 0.60, 0.10).Target("y").Likeness(0.2));
        var likeness = Assert.Single(prepared.Evidence.Values.OfType<PartLikeness>());

        Assert.All(likeness.Columns, column => Assert.Contains(column.Pairs, pair => pair.First == Standing.Calibration || pair.Second == Standing.Calibration));
    }
}
