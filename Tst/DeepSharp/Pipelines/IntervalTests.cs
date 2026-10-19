// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// An answer with the width it is worth, by split conformal prediction: the errors a model makes on rows it was neither fitted nor
/// chosen on, in the answer's own units, and the one that all but a share alpha of them stay under. A row an answer is later given for
/// is then as likely as one less alpha to land within that margin of it, whatever the model is — a network, an ensemble or a tree —
/// as long as the rows are as like the calibration rows as those are to one another. The margin is the ⌈(n+1)(1−α)⌉-th smallest error;
/// with fewer rows than that, no margin keeps the promise, and none is given.
/// </summary>
public class IntervalTests
{
    // A hundred days: 50 to learn from, 10 to choose with, 20 to calibrate on and 20 to be measured on, the answer read on a log scale.
    private static PreparedData Priced(double calibration = 0.20) =>
        Pdd.Create()
            .Read(
                new InMemoryRowSource(
                    ["Date", "x", "y"],
                    [.. Enumerable.Range(0, 100).Select(day => (IReadOnlyList<string?>)
                        [new DateTime(2024, 1, 1).AddDays(day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture), (day % 7).ToString(CultureInfo.InvariantCulture), (10 + (day * 7 % 13)).ToString(CultureInfo.InvariantCulture)])]),
                "a hundred days")
            .Declare(schema => schema.Timestamp("Date").Number("x", "y"))
            .Reshape("y", Maths.Log)
            .Calibrate(calibration)
            .SplitByTime("Date", 0.50, 0.10)
            .Drop("Date")
            .Target("y")
            .Report(report => report.Measure(Metric.Rmse, Metric.Coverage, Metric.Width).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .Build()
            .Run();

    private static PartPredictions Shifted(PreparedData prepared, Part part, Func<int, double> by)
    {
        var batch = prepared.Batch(part);

        return new PartPredictions(batch, [.. batch.Answers!.Select((row, at) => new[] { row[0] + by(at) })]);
    }

    private static double[] OwnErrors(PreparedData prepared, Part part, Func<int, double> by) =>
        [.. prepared.BackToOriginal(prepared.Batch(part).Answers!, part).Select((row, at) => Math.Abs((row[0] * Math.Exp(by(at))) - row[0]))];

    [Fact]
    public void TheMargin_IsTheNPlusOneTimesOneLessAlphaThSmallestError_InTheAnswersOwnUnits()
    {
        var prepared = Priced();
        double Shift(int at) => 0.01 * (at + 1);

        var interval = prepared.IntervalOf(Shifted(prepared, Part.Calibration, Shift), 0.10);

        // Twenty rows and alpha a tenth: ⌈21 × 0.9⌉ = 19, so the nineteenth smallest of the errors, which are exp(y + shift) less exp(y).
        Assert.Equal(20, interval.Rows);
        Assert.Equal(0.10, interval.Alpha);
        Assert.Equal(OwnErrors(prepared, Part.Calibration, Shift).Order().ElementAt(18), Assert.Single(interval.Margins), 1e-9);
    }

    [Theory]
    [InlineData(0.05, 20)]
    [InlineData(0.50, 11)]
    [InlineData(0.10, 19)]
    public void TheRankTakenIsCeilNPlusOneTimesOneLessAlpha_AndNeverMoreThanTheRowsThereAre(double alpha, int rank)
    {
        var prepared = Priced();
        double Shift(int at) => 0.01 * (at + 1);

        // Twenty rows: ⌈21 × 0.95⌉ = 20, the last, is the first rank that exists; ⌈21 × 0.5⌉ = 11.
        var interval = prepared.IntervalOf(Shifted(prepared, Part.Calibration, Shift), alpha);

        Assert.Equal(rank, (int)Math.Ceiling(21 * (1 - alpha)));
        Assert.Equal(OwnErrors(prepared, Part.Calibration, Shift).Order().ElementAt(rank - 1), interval.Margins[0], 1e-9);
    }

    [Fact]
    public void TooFewRows_ForTheLevelAsked_AreRefused_NamingTheRowsThereAreTheLevelAndTheRowsNeeded()
    {
        var prepared = Priced(calibration: 0.05);
        var refused = Assert.Throws<InvalidOperationException>(() => prepared.IntervalOf(Shifted(prepared, Part.Calibration, _ => 0.1), 0.10));

        // Five rows hold no margin that keeps a promise of nine in ten: it takes nine.
        Assert.Contains("5 rows", refused.Message, StringComparison.Ordinal);
        Assert.Contains("0.1", refused.Message, StringComparison.Ordinal);
        Assert.Contains("9 rows", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    public void TheLevel_IsRequired_AndIsAShareBetweenNothingAndAll(double alpha)
    {
        var prepared = Priced();

        Assert.Throws<ArgumentOutOfRangeException>(() => prepared.IntervalOf(Shifted(prepared, Part.Calibration, _ => 0.1), alpha));
    }

    [Fact]
    public void RowsThatAreNotTheCalibrationRows_AreRefused_ForTheirErrorsWouldBeAnotherPartsErrors()
    {
        var prepared = Priced();

        Assert.Throws<ArgumentException>(() => prepared.IntervalOf(Shifted(prepared, Part.Test, _ => 0.1), 0.10));
        Assert.Throws<ArgumentException>(() => prepared.IntervalOf(Shifted(prepared, Part.Calibration, _ => 0.1) with { Predictions = [] }, 0.10));
    }

    [Fact]
    public void ASplitThatSetsNoRowsAside_HasNothingToCalibrateOn()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["t", "x", "y"], [.. Enumerable.Range(1, 40).Select(t => (IReadOnlyList<string?>)[$"{t}", $"{t % 5}", $"{t % 7}"])]), "forty")
            .Declare(schema => schema.Integer("t").Number("x", "y"))
            .SplitByTime("t", 0.50, 0.10)
            .Target("y")
            .Build()
            .Run();
        var refused = Assert.Throws<InvalidOperationException>(() => prepared.IntervalOf(new PartPredictions(prepared.Batch(Part.Test), []), 0.10));

        Assert.Contains("Calibrate", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAnswerThatIsAClass_HasNoMargin_ForAClassHasAChanceAndNoWidth()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass"))
            .Calibrate(0.20)
            .SplitAtRandom(0.50, 0.10)
            .Target("survived")
            .Build()
            .Run();
        var batch = prepared.Batch(Part.Calibration);
        var refused = Assert.Throws<InvalidOperationException>(() => prepared.IntervalOf(new PartPredictions(batch, [.. batch.Features.Select(_ => new[] { 0.4 })]), 0.10));

        Assert.Contains("class", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SeveralAnswers_EachHaveAMarginOfTheirOwn()
    {
        // Three bands of a flock, as shares of a whole: a prediction that is out by more in one band than another has a wider margin there.
        var prepared = Pdd.Create()
            .Read(
                new InMemoryRowSource(
                    ["t", "c1", "c2", "c3"],
                    [.. Enumerable.Range(1, 100).Select(t => (IReadOnlyList<string?>)[$"{t}", $"{1 + (t % 3)}", $"{2 + (t % 4)}", $"{1 + (t % 2)}"])]),
                "a hundred flocks")
            .Declare(schema => schema.Integer("t").Number("c1", "c2", "c3"))
            .Calibrate(0.20)
            .SplitByTime("t", 0.50, 0.10)
            .NormaliseRow(Norm.L1, "c1", "c2", "c3")
            .Distribution(["c1", "c2", "c3"], scaleBy: null, ordered: true)
            .Build()
            .Run();
        var batch = prepared.Batch(Part.Calibration);
        double[] by = [0.01, 0.05, 0.10];
        var interval = prepared.IntervalOf(new PartPredictions(batch, [.. batch.Answers!.Select(row => row.Select((share, column) => share + by[column]).ToArray())]), 0.10);

        Assert.Equal(3, interval.Margins.Count);
        Assert.True(interval.Margins[0] < interval.Margins[1] && interval.Margins[1] < interval.Margins[2], string.Join(", ", interval.Margins));
    }

    // ---- what an interval holds: written as a part of a file, and read back ----

    [Fact]
    public void AnInterval_IsWrittenAsItsLevelItsRowsAndItsMargins_AndReadBackEqual()
    {
        var interval = new ConformalInterval(0.1, 20, [3.5, 0.25]);
        var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            interval.WriteTo(writer);
        }

        var text = Encoding.UTF8.GetString(buffer.ToArray());

        Assert.Equal("""{"alpha":0.1,"rows":20,"margins":[3.5,0.25]}""", text);
        Assert.Equal(interval, ConformalInterval.ReadFrom(JsonDocument.Parse(text).RootElement), new IntervalComparer());
    }

    [Theory]
    [InlineData("""{"alpha":0,"rows":20,"margins":[1]}""")]
    [InlineData("""{"alpha":0.1,"rows":20}""")]
    [InlineData("""{"alpha":0.1,"rows":20,"margins":[]}""")]
    [InlineData("""{"alpha":0.1,"rows":20,"margins":[-1]}""")]
    [InlineData("""{"alpha":0.1,"rows":0,"margins":[1]}""")]
    [InlineData("""{"alpha":0.1,"rows":20,"margins":[1],"colour":"red"}""")]
    [InlineData("""[1]""")]
    public void AnIntervalThatSaysNonsense_IsRefusedWhereItIsRead(string text) =>
        Assert.Throws<FormatException>(() => ConformalInterval.ReadFrom(JsonDocument.Parse(text).RootElement));

    // ---- measuring an interval: coverage and width ----

    [Fact]
    public void CoverageIsTheShareOfRowsWithinTheMargin_AndWidthIsTwiceTheMargin_BesideTheTrainingAveragesOwn()
    {
        var prepared = Priced();
        double Shift(int at) => 0.01 * (at + 1);
        var margin = 0.4 * OwnErrors(prepared, Part.Test, Shift).Max();
        var test = Shifted(prepared, Part.Test, Shift) with { Margins = [margin] };
        var validation = Shifted(prepared, Part.Validation, Shift) with { Margins = [margin] };

        var measures = prepared.Measure([validation, test]);
        var part = measures.Parts.Single(each => each.Part == Part.Test);
        var errors = OwnErrors(prepared, Part.Test, Shift);

        Assert.Equal([Metric.Rmse, Metric.Coverage, Metric.Width], measures.Metrics);
        Assert.Equal(errors.Count(error => error <= margin) / (double)errors.Length, part.Values.Single(value => value.Metric == Metric.Coverage).Value, 12);
        Assert.Equal(2 * margin, part.Values.Single(value => value.Metric == Metric.Width).Value, 12);
        Assert.Equal(2 * margin, part.Values.Single(value => value.Metric == Metric.Width).Baseline, 12);

        // The training rows' average as handed over, brought back like any prediction, given the same margin on each side, covers the rows whose answer lies within it.
        var handedAverage = prepared.Batch(Part.Train).Answers!.Average(row => row[0]);
        var average = prepared.BackToOriginal(prepared.Batch(Part.Test).Answers!.Select(_ => new[] { handedAverage }).ToArray(), Part.Test)[0][0];
        var own = prepared.BackToOriginal(prepared.Batch(Part.Test).Answers!, Part.Test);

        Assert.Equal(own.Count(row => Math.Abs(row[0] - average) <= margin) / (double)own.Count, part.Values.Single(value => value.Metric == Metric.Coverage).Baseline, 12);
    }

    [Fact]
    public void WithNoMarginsGiven_TheReportIsMeasuredWithoutCoverageAndWidth_ForNothingSaysHowWideAnIntervalIs()
    {
        var prepared = Priced();
        var measures = prepared.Measure([Shifted(prepared, Part.Validation, _ => 0.1), Shifted(prepared, Part.Test, _ => 0.1)]);

        Assert.Equal([Metric.Rmse], measures.Metrics);
        Assert.All(measures.Parts, part => Assert.Equal([Metric.Rmse], part.Values.Select(value => value.Metric)));
    }

    [Fact]
    public void MarginsThatAreNotOneForEachAnswer_AreRefused()
    {
        var prepared = Priced();
        var wrong = Shifted(prepared, Part.Test, _ => 0.1) with { Margins = [1, 2] };
        var negative = Shifted(prepared, Part.Test, _ => 0.1) with { Margins = [-1] };

        Assert.Throws<ArgumentException>(() => prepared.Measure([Shifted(prepared, Part.Validation, _ => 0.1) with { Margins = [1] }, wrong]));
        Assert.Throws<ArgumentException>(() => prepared.Measure([Shifted(prepared, Part.Validation, _ => 0.1) with { Margins = [1] }, negative]));
    }

    [Fact]
    public void MarginsForSomePartsAlone_AreRefused_AndCoverageOfNothingThatHasNoMarginsIsRefusedToo()
    {
        var prepared = Priced();

        Assert.Throws<ArgumentException>(() => prepared.Measure([Shifted(prepared, Part.Validation, _ => 0.1), Shifted(prepared, Part.Test, _ => 0.1) with { Margins = [1] }]));
        Assert.Throws<InvalidOperationException>(() => new Answered([[1.0]], [[1.0]]).Of(Metric.Coverage, 0));
    }

    [Fact]
    public void MarginsCrossAsTextWithThePredictions_SoACoverageIsMeasuredAgainWhereTheyAreShown()
    {
        var prepared = Priced();
        double Shift(int at) => 0.01 * (at + 1);
        var margin = 0.4 * OwnErrors(prepared, Part.Test, Shift).Max();
        var measured = prepared.Measure([Shifted(prepared, Part.Validation, Shift) with { Margins = [margin] }, Shifted(prepared, Part.Test, Shift) with { Margins = [margin] }]);
        var text = measured.PredictionsToJson();
        var again = prepared.MeasureAgain(text);

        Assert.Equal(2, JsonDocument.Parse(text).RootElement.GetProperty("version").GetInt32());
        Assert.Contains("\"margins\":[", text, StringComparison.Ordinal);
        Assert.Equal(measured.Metrics, again.Metrics);
        Assert.Equal(
            measured.Parts.SelectMany(part => part.Values.Select(value => value.Value)),
            again.Parts.SelectMany(part => part.Values.Select(value => value.Value)));

        // Predictions without margins are written as they always were.
        var plain = prepared.Measure([Shifted(prepared, Part.Validation, Shift), Shifted(prepared, Part.Test, Shift)]).PredictionsToJson();

        Assert.Equal(1, JsonDocument.Parse(plain).RootElement.GetProperty("version").GetInt32());
        Assert.DoesNotContain("margins", plain, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("\"margins\":{}")]
    [InlineData("\"margins\":[\"a\"]")]
    public void MarginsThatAreNoNumbers_AreRefusedWhereTheyAreRead(string margins)
    {
        var prepared = Priced();
        var measured = prepared.Measure([Shifted(prepared, Part.Validation, _ => 0.1) with { Margins = [1] }, Shifted(prepared, Part.Test, _ => 0.1) with { Margins = [1] }]);
        var broken = measured.PredictionsToJson().Replace("\"margins\":[1]", margins, StringComparison.Ordinal);

        Assert.Throws<ArgumentException>(() => prepared.MeasureAgain(broken));
    }

    [Fact]
    public void ARulePinnedByMeasurement_SyntheticKnownNoise_CoversAtLeastOneLessAlphaOnFreshRows()
    {
        // Rows whose answer is x plus noise drawn from a seed, and a model that says x: the margin worked out on the calibration
        // rows keeps its promise of nine in ten on the test rows, which nothing was fitted or chosen on, to within their sampling.
        var noise = new Random(7);
        var rows = Enumerable.Range(1, 6000).Select(t =>
        {
            var x = noise.NextDouble() * 100;
            var y = x + ((noise.NextDouble() + noise.NextDouble() + noise.NextDouble() - 1.5) * 4);

            return (IReadOnlyList<string?>)[$"{t}", x.ToString("R", CultureInfo.InvariantCulture), y.ToString("R", CultureInfo.InvariantCulture)];
        }).ToList();
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["t", "x", "y"], rows), "six thousand")
            .Declare(schema => schema.Integer("t").Number("x", "y"))
            .Calibrate(0.20)
            .SplitAtRandom(0.50, 0.10, 3)
            .Target("y")
            .Report(report => report.Measure(Metric.Coverage, Metric.Width, Metric.Mae).On(Part.Test).As(Shown.Numbers))
            .Build()
            .Run();

        PartPredictions Said(Part part, double[]? margins = null)
        {
            var batch = prepared.Batch(part);

            return new PartPredictions(batch, [.. batch.Features.Select(row => new[] { row[prepared.Batch(part).FeatureNames.ToList().IndexOf("x")] })]) { Margins = margins };
        }

        const double Alpha = 0.10;
        var interval = prepared.IntervalOf(Said(Part.Calibration), Alpha);
        var measures = prepared.Measure([Said(Part.Test, [.. interval.Margins])]);
        var test = measures.Parts.Single();
        var covered = test.Values.Single(value => value.Metric == Metric.Coverage).Value;

        Assert.Equal(1200, interval.Rows);
        Assert.True(covered >= 1 - Alpha - 0.03, $"covered {covered}");
        Assert.True(covered <= 1 - Alpha + 0.03, $"covered {covered}, far over the promise");
        Assert.Equal(2 * interval.Margins[0], test.Values.Single(value => value.Metric == Metric.Width).Value, 12);
    }

    private sealed class IntervalComparer : IEqualityComparer<ConformalInterval>
    {
        public bool Equals(ConformalInterval? left, ConformalInterval? right) =>
            left is not null && right is not null && left.Alpha == right.Alpha && left.Rows == right.Rows && left.Margins.SequenceEqual(right.Margins);

        public int GetHashCode(ConformalInterval value) => value.Rows;
    }
}
