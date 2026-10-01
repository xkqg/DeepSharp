// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;
using DeepSharp.Tests.Learners;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a model predicted is measured by what its pipeline's report declared: on each part it names, in the answer's own
/// units — the predictions and the answers they are compared with both come back through the way back — and each measure
/// beside the same measure of predicting, for every row, the average answer of the training rows. The numbers are
/// scikit-learn's, worked out by it on the same rows divided the same way (scikit-learn 1.9.1, numpy 2.5.3). And the
/// predictions have to be for the part's own rows in the order the part hands them over: in any other order each would be
/// measured against another row's answer, which is how 346 of 349 price rows came back against other rows before.
/// </summary>
public class MeasureTests
{
    private const double Close = 1e-12;

    private static readonly Part[] ThreeParts = [Part.Train, Part.Validation, Part.Test];

    private static int At(Batch batch, string feature) => batch.FeatureNames.ToList().IndexOf(feature);

    private static PartPredictions Predicted(PreparedData prepared, Part part, Func<Batch, int, double[]> model)
    {
        var batch = prepared.Batch(part);

        return new PartPredictions(batch, [.. Enumerable.Range(0, batch.RowCount).Select(row => model(batch, row))]);
    }

    private static Measures Measured(PreparedData prepared, Func<Batch, int, double[]> model) =>
        prepared.Measure([.. prepared.Declaration.Report!.Parts.Select(part => Predicted(prepared, part, model))]);

    private static void AssertClose(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual[at], Close);
        }
    }

    private static void AssertMeasured(PartMeasures part, Metric metric, double value, double baseline)
    {
        var measured = part.Values.Single(each => each.Metric == metric);

        Assert.Equal(value, measured.Value, Close);
        Assert.Equal(baseline, measured.Baseline, Close);
    }

    // ---- Titanic: whether a passenger survived, a column of noughts and ones ----

    private static PreparedData Passengers(Action<ReportBuilder> report) =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass").Category("sex"))
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories()
            .Target("survived")
            .Report(report)
            .Build()
            .Run();

    // A guess a person might make: most women and few men, and a woman in third class as likely as not.
    private static double[] Guessed(Batch batch, int row)
    {
        var female = batch.Features[row][At(batch, "sex_female")] == 1;
        var pclass = batch.Features[row][At(batch, "pclass")];

        return [female ? (pclass == 3 ? 0.5 : 0.95) : (pclass == 1 ? 0.37 : 0.13)];
    }

    [Fact]
    public void ThePassengersClasses_AreMeasuredAsScikitLearnMeasuresThem_EachBesideTheTrainingRowsAverage()
    {
        var measures = Measured(
            Passengers(report => report
                .Measure(Metric.Accuracy, Metric.Precision, Metric.Recall, Metric.ConfusionMatrix, Metric.Rmse, Metric.Mae, Metric.R2)
                .On(Part.Train, Part.Validation, Part.Test)
                .As(Shown.Numbers)),
            Guessed);

        // accuracy, precision, recall, rmse, mae, r2: the guess, then P(survived) = 0.38362760834670945 for everyone, the
        // training rows' average, whose class is nought.
        double[][] guess =
        [
            [0.7897271268057785, 0.7571428571428571, 0.6652719665271967, 0.3777620591784399, 0.2812680577849117, 0.3964911512813808],
            [0.7744360902255639, 0.7058823529411765, 0.7058823529411765, 0.38713619837238883, 0.29330827067669174, 0.36606195600191305],
            [0.7851851851851852, 0.7169811320754716, 0.7307692307692307, 0.3765171778089085, 0.2793333333333333, 0.40137384151992583],
        ];
        double[][] average =
        [
            [0.6163723916532905, 0, 0, 0.48626892401313615, 0.4729149329217863, 0],
            [0.6165413533834586, 0, 0, 0.4862284869273067, 0.47287560796051126, -1.207524491508849e-07],
            [0.6148148148148148, 0, 0, 0.4866415357782847, 0.47327745080554073, -1.0244365431288927e-05],
        ];
        int[][][] counts = [[[333, 51], [80, 159]], [[67, 15], [15, 36]], [[68, 15], [14, 38]]];
        int[][][] averageCounts = [[[384, 0], [239, 0]], [[82, 0], [51, 0]], [[83, 0], [52, 0]]];
        Metric[] numbers = [Metric.Accuracy, Metric.Precision, Metric.Recall, Metric.Rmse, Metric.Mae, Metric.R2];

        Assert.Equal(ThreeParts, measures.Parts.Select(part => part.Part));
        Assert.Equal([623, 133, 135], measures.Parts.Select(part => part.Rows));
        Assert.Equal(["survived"], measures.Answers);

        for (var at = 0; at < 3; at++)
        {
            var part = measures.Parts[at];

            Assert.Equal(numbers, part.Values.Select(each => each.Metric));

            for (var metric = 0; metric < numbers.Length; metric++)
            {
                AssertMeasured(part, numbers[metric], guess[at][metric], average[at][metric]);
            }

            var confusion = Assert.Single(part.Confusions);

            Assert.Equal("survived", confusion.Answer);
            Assert.Equal(["0", "1"], confusion.Classes);
            Assert.Equal(counts[at], confusion.Counts);
            Assert.Equal(averageCounts[at], confusion.Baseline);
            Assert.Equal(part.Rows, part.Predicted.Count);
            Assert.All(part.Actual, answer => Assert.True(answer is [0] or [1]));
        }
    }

    // ---- Apple: the price five days on, predicted as a return and measured in dollars ----

    private static PreparedData Prices() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close", "AAPL.Volume"))
            .OrderBy("Date")
            .SplitByTime("Date", 0.70, 0.15, 5)
            .Ahead("AAPL.Close", 5, AheadAs.Return)
            .Drop("Date")
            .Report(report => report.Measure(Metric.Rmse, Metric.Mae, Metric.R2).On(Part.Train, Part.Validation, Part.Test).As(Shown.Numbers, Shown.Drawn))
            .Build()
            .Run();

    [Fact]
    public void AReturnFiveDaysOn_IsMeasuredAsThePriceItComesBackAs_BesideTheTrainingRowsAverageReturn()
    {
        // The model says nothing changes: a return of nought, which comes back as the day's own price. In the answer's
        // own units that is 0.8201 of R² on the test part; the training rows' average return, -0.0036213850658739850,
        // comes back as each day's price times one and that.
        var measures = Measured(Prices(), (_, _) => [0]);

        double[][] unchanged =
        [
            [3.804783626537739, 2.898481392550143, 0.9047347409801602],
            [3.3317934858160343, 2.3638028873239416, 0.6020876865058413],
            [3.2504672966780155, 2.4966197887323944, 0.8200728885907418],
        ];
        double[][] average =
        [
            [3.7691584661627786, 2.87885851815315, 0.9065103749013239],
            [3.491081611752931, 2.4765356695862173, 0.5631309961948852],
            [3.4785775643866317, 2.696562462007443, 0.7939330281977685],
        ];

        Assert.Equal([349, 71, 71], measures.Parts.Select(part => part.Rows));
        Assert.Equal(["AAPL.Close.ahead5"], measures.Answers);
        Assert.Equal([Metric.Rmse, Metric.Mae, Metric.R2], measures.Metrics);
        Assert.Equal([Shown.Numbers, Shown.Drawn], measures.Shown);

        for (var at = 0; at < 3; at++)
        {
            AssertMeasured(measures.Parts[at], Metric.Rmse, unchanged[at][0], average[at][0]);
            AssertMeasured(measures.Parts[at], Metric.Mae, unchanged[at][1], average[at][1]);
            AssertMeasured(measures.Parts[at], Metric.R2, unchanged[at][2], average[at][2]);
            Assert.Empty(measures.Parts[at].Confusions);
        }

        // The seventeenth of February 2015 closed at 127.830002 and the twenty-fourth, five days on, at 132.169998.
        Assert.Equal(127.830002, measures.Parts[0].Predicted[0][0], Close);
        Assert.Equal(132.169998, measures.Parts[0].Actual[0][0], Close);
    }

    // ---- Twenty rows, t from 1 to 20, divided in time: ten to train on, five to choose, five to test ----

    private static IReadOnlyList<IReadOnlyList<string?>> Twenty(Func<int, IEnumerable<double>> cells) =>
        [.. Enumerable.Range(1, 20).Select(t => (IReadOnlyList<string?>)[Written(t), .. cells(t).Select(Written)])];

    private static string Written(double value) => value.ToString(CultureInfo.InvariantCulture);

    private static int T(Batch batch, int row) => (int)batch.Features[row][At(batch, "t")];

    // Likelihoods of three answers, arbitrary and repeatable, with ties among them.
    private static double[] Likely(int t) => [(t * 5 % 7) / 10.0, (t * 3 % 5) / 10.0, (t * 2 % 9) / 10.0];

    private static PreparedData Labelled(int ones, Func<int, IEnumerable<double>> labels) =>
        Pdd.Create()
            .Read(new InMemoryRowSource(["t", "a", "b", "c"], Twenty(labels)), "twenty rows")
            .Declare(schema => schema.Integer("t", "a", "b", "c"))
            .SplitByTime("t", 0.50, 0.25)
            .Labels(["a", "b", "c"], ones)
            .Report(report => report.Measure(Metric.Accuracy, Metric.Precision, Metric.Recall, Metric.ConfusionMatrix).On(ThreeParts).As(Shown.Numbers))
            .Build()
            .Run();

    [Fact]
    public void LabelsOfWhichARowHoldsOne_AreMeasuredByTheLikeliest_WithOneMatrixAcrossThem()
    {
        var measures = Measured(Labelled(1, t => [t * 7 % 3 == 0 ? 1 : 0, t * 7 % 3 == 1 ? 1 : 0, t * 7 % 3 == 2 ? 1 : 0]), (batch, row) => Likely(T(batch, row)));

        // accuracy, precision, recall: the likeliest label, then the training rows' likeliest, b, for every row.
        double[][] likeliest = [[0.3, 0.2222222222222222, 0.3333333333333333], [0.2, 0.16666666666666666, 0.16666666666666666], [0.6, 0.38888888888888884, 0.6666666666666666]];
        double[][] average = [[0.4, 0.13333333333333333, 0.3333333333333333], [0.2, 0.06666666666666667, 0.3333333333333333], [0.4, 0.13333333333333333, 0.3333333333333333]];
        int[][][] counts = [[[1, 1, 1], [1, 0, 3], [1, 0, 2]], [[1, 0, 1], [0, 0, 1], [1, 1, 0]], [[1, 0, 0], [1, 0, 1], [0, 0, 2]]];
        int[][][] averageCounts = [[[0, 3, 0], [0, 4, 0], [0, 3, 0]], [[0, 2, 0], [0, 1, 0], [0, 2, 0]], [[0, 1, 0], [0, 2, 0], [0, 2, 0]]];

        for (var at = 0; at < 3; at++)
        {
            var part = measures.Parts[at];

            AssertMeasured(part, Metric.Accuracy, likeliest[at][0], average[at][0]);
            AssertMeasured(part, Metric.Precision, likeliest[at][1], average[at][1]);
            AssertMeasured(part, Metric.Recall, likeliest[at][2], average[at][2]);

            var confusion = Assert.Single(part.Confusions);

            Assert.Null(confusion.Answer);
            Assert.Equal(["a", "b", "c"], confusion.Classes);
            Assert.Equal(counts[at], confusion.Counts);
            Assert.Equal(averageCounts[at], confusion.Baseline);
        }
    }

    [Fact]
    public void LabelsOfWhichARowHoldsAnyNumber_AreEachMeasuredByItself_WithAMatrixForEach()
    {
        var measures = Measured(Labelled(0, t => [t % 2, t % 3 == 0 ? 1 : 0, t % 5 == 0 ? 1 : 0]), (batch, row) => Likely(T(batch, row)));

        // accuracy (every label of a row right), precision, recall: each label at a half or more, then the training rows'
        // averages, 0.5, 0.3 and 0.2, which say a and nothing else.
        double[][] each = [[0.2, 0.1111111111111111, 0.06666666666666667], [0.4, 0.3333333333333333, 0.2222222222222222], [0, 0, 0]];
        double[][] average = [[0.2, 0.16666666666666666, 0.3333333333333333], [0.4, 0.19999999999999998, 0.3333333333333333], [0.4, 0.13333333333333333, 0.3333333333333333]];
        int[][][][] counts =
        [
            [[[3, 2], [4, 1]], [[7, 0], [3, 0]], [[4, 4], [2, 0]]],
            [[[2, 0], [1, 2]], [[3, 0], [2, 0]], [[2, 2], [1, 0]]],
            [[[2, 1], [2, 0]], [[4, 0], [1, 0]], [[2, 2], [1, 0]]],
        ];
        int[][][][] averageCounts =
        [
            [[[0, 5], [0, 5]], [[7, 0], [3, 0]], [[8, 0], [2, 0]]],
            [[[0, 2], [0, 3]], [[3, 0], [2, 0]], [[4, 0], [1, 0]]],
            [[[0, 3], [0, 2]], [[4, 0], [1, 0]], [[4, 0], [1, 0]]],
        ];

        for (var at = 0; at < 3; at++)
        {
            var part = measures.Parts[at];

            AssertMeasured(part, Metric.Accuracy, each[at][0], average[at][0]);
            AssertMeasured(part, Metric.Precision, each[at][1], average[at][1]);
            AssertMeasured(part, Metric.Recall, each[at][2], average[at][2]);
            Assert.Equal(["a", "b", "c"], part.Confusions.Select(confusion => confusion.Answer));
            Assert.All(part.Confusions, confusion => Assert.Equal(["0", "1"], confusion.Classes));
            Assert.Equal(counts[at], part.Confusions.Select(confusion => confusion.Counts.ToArray()));
            Assert.Equal(averageCounts[at], part.Confusions.Select(confusion => confusion.Baseline.ToArray()));
        }
    }

    [Fact]
    public void SharesOfAWhole_AreMeasuredAsTheCountsTheyComeBackAs_ColumnByColumnAndAveraged()
    {
        // Counts as read, made shares by dividing each row by its sum, and brought back as a share times the row's total.
        double[][] rows =
        [
            [1, 4, 4, 1, 9], [2, 2, 6, 2, 10], [3, 5, 8, 3, 16], [4, 3, 3, 0, 6], [5, 1, 5, 1, 7], [6, 4, 7, 2, 13], [7, 2, 2, 3, 7],
            [8, 5, 4, 0, 9], [9, 3, 6, 1, 10], [10, 1, 8, 2, 11], [11, 4, 3, 3, 10], [12, 2, 5, 0, 7], [13, 5, 7, 1, 13],
            [14, 3, 2, 2, 7], [15, 1, 4, 3, 8], [16, 4, 6, 0, 10], [17, 2, 8, 1, 11], [18, 5, 3, 2, 10], [19, 3, 5, 3, 11], [20, 1, 7, 0, 8],
        ];
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["t", "c1", "c2", "c3", "n"], Twenty(t => rows[t - 1].Skip(1))), "twenty totals")
            .Declare(schema => schema.Integer("t").Number("c1", "c2", "c3", "n"))
            .SplitByTime("t", 0.50, 0.25)
            .NormaliseRow(Norm.L1, "c1", "c2", "c3")
            .Distribution(["c1", "c2", "c3"], scaleBy: "n")
            .Report(report => report.Measure(Metric.Rmse, Metric.Mae, Metric.R2).On(ThreeParts).As(Shown.Numbers))
            .Build()
            .Run();

        var measures = Measured(prepared, (batch, row) => T(batch, row) % 2 == 0 ? [0.2, 0.3, 0.5] : [0.4, 0.4, 0.2]);

        // rmse, mae, r2 of the counts: the predicted shares, then the training rows' average shares.
        double[][] predicted = [[2.217910409400669, 1.86, -2.27391551056118], [1.62698333094487, 1.3466666666666667, -0.5409220985691574], [2.7431903275851446, 2.3466666666666667, -3.234887122416533]];
        double[][] average = [[1.0725998825734349, 0.8382462814962813, 0.38952132849137805], [1.226796996002584, 1.0809966884966886, 0.20669723851294894], [1.4561106262951427, 1.3373691678691675, -0.015903284340572237]];

        for (var at = 0; at < 3; at++)
        {
            AssertMeasured(measures.Parts[at], Metric.Rmse, predicted[at][0], average[at][0]);
            AssertMeasured(measures.Parts[at], Metric.Mae, predicted[at][1], average[at][1]);
            AssertMeasured(measures.Parts[at], Metric.R2, predicted[at][2], average[at][2]);
        }

        // The first row counted 4, 4 and 1 of 9.
        AssertClose([4, 4, 1], measures.Parts[0].Actual[0]);
        AssertClose([3.6, 3.6, 1.8], measures.Parts[0].Predicted[0]);
    }

    [Fact]
    public void AnAnswerThatWasScaled_IsMeasuredInTheUnitsItWasReadIn()
    {
        // Measured in the scaled units, every fare would lie between nought and one and every error would look small.
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("pclass").Number("fare"))
            .SplitStratified("pclass", 0.70, 0.15)
            .Normalise("fare", Scale.MinMax)
            .Target("fare")
            .Report(report => report.Measure(Metric.Rmse, Metric.R2).On(Part.Test).As(Shown.Numbers))
            .Build()
            .Run();

        var measures = Measured(prepared, (batch, row) => batch.Answers![row]);
        var test = Assert.Single(measures.Parts);

        Assert.Equal(0, test.Values[0].Value, Close);
        Assert.Equal(1, test.Values[1].Value, Close);
        Assert.Contains(test.Actual, answer => answer[0] > 100);
        Assert.Equal(test.Actual.Select(answer => answer[0]), test.Predicted.Select(answer => answer[0]));
    }

    [Fact]
    public void AValueReadAhead_IsCountedAsItsClasses()
    {
        // Whether it rains two days on: a column of noughts and ones read ahead, predicted exactly.
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["t", "rain"], Twenty(t => [t % 3 == 0 ? 1 : 0])), "twenty days")
            .Declare(schema => schema.Integer("t", "rain"))
            .OrderBy("t")
            .SplitByTime("t", 0.50, 0.25, 2)
            .Ahead("rain", 2)
            .Report(report => report.Measure(Metric.Accuracy, Metric.Recall).On(Part.Train, Part.Test).As(Shown.Numbers))
            .Build()
            .Run();

        var measures = Measured(prepared, (batch, row) => batch.Answers![row]);

        Assert.All(measures.Parts, part => Assert.Equal(1, part.Values[0].Value));
        Assert.All(measures.Parts, part => Assert.Equal(1, part.Values[1].Value));
    }

    // ---- What is refused ----

    private static PreparedData Guessing(Action<ReportBuilder>? report = null) =>
        Passengers(report ?? (each => each.Measure(Metric.Accuracy).On(Part.Train, Part.Validation, Part.Test).As(Shown.Numbers)));

    [Fact]
    public void PredictionsInAnotherOrder_AreRefused_TheyWouldBeMeasuredAgainstOtherRows()
    {
        var prepared = Guessing();
        var train = prepared.Batch(Part.Train);
        int[] reversed = [.. Enumerable.Range(0, train.RowCount).Reverse()];
        var shuffled = train with
        {
            Features = [.. reversed.Select(row => train.Features[row])],
            Keys = [.. reversed.Select(row => train.Keys![row])],
        };
        PartPredictions[] predictions =
        [
            new(shuffled, [.. reversed.Select(row => Guessed(train, row))]),
            Predicted(prepared, Part.Validation, Guessed),
            Predicted(prepared, Part.Test, Guessed),
        ];

        var refused = Assert.Throws<ArgumentException>(() => prepared.Measure(predictions));

        Assert.Contains("'train'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("another order", refused.Message, StringComparison.Ordinal);

        // Fewer rows than the part holds are other rows too.
        predictions[0] = new(train with { Keys = [.. train.Keys!.Skip(1)] }, [.. Enumerable.Range(1, train.RowCount - 1).Select(row => Guessed(train, row))]);

        Assert.Contains("622 rows", Assert.Throws<ArgumentException>(() => prepared.Measure(predictions)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APartTheReportNames_WithoutPredictions_IsRefusedByName()
    {
        var prepared = Guessing();

        var refused = Assert.Throws<ArgumentException>(() => prepared.Measure(
            [Predicted(prepared, Part.Train, Guessed), Predicted(prepared, Part.Test, Guessed)]));

        Assert.Contains("'validation'", refused.Message, StringComparison.Ordinal);

        // Every part without predictions is named at once.
        var two = Assert.Throws<ArgumentException>(() => prepared.Measure([Predicted(prepared, Part.Train, Guessed)]));

        Assert.Contains("'validation' and 'test'", two.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PredictionsForAPartTheReportDoesNotName_OrTwiceForOne_AreRefused()
    {
        var prepared = Guessing(report => report.Measure(Metric.Accuracy).On(Part.Test).As(Shown.Numbers));
        var test = Predicted(prepared, Part.Test, Guessed);

        Assert.Contains("'train'", Assert.Throws<ArgumentException>(() => prepared.Measure([test, Predicted(prepared, Part.Train, Guessed)])).Message, StringComparison.Ordinal);
        Assert.Contains("two", Assert.Throws<ArgumentException>(() => prepared.Measure([test, test])).Message, StringComparison.Ordinal);
        Assert.Single(prepared.Measure([test]).Parts);
    }

    [Fact]
    public void PredictionsForABatchThePipelineDidNotHandOver_AreRefused()
    {
        var prepared = Guessing(report => report.Measure(Metric.Accuracy).On(Part.Test).As(Shown.Numbers));
        var test = Predicted(prepared, Part.Test, Guessed);

        Assert.Throws<ArgumentException>(() => prepared.Measure([test with { Batch = test.Batch with { Keys = null } }]));
        Assert.Throws<ArgumentException>(() => prepared.Measure([test with { Batch = new Batch(test.Batch.FeatureNames, test.Batch.Features, null) }]));
        Assert.Throws<ArgumentNullException>(() => prepared.Measure(null!));
    }

    [Fact]
    public void NotOnePredictionPerRowAndAnswer_IsRefused()
    {
        var prepared = Guessing(report => report.Measure(Metric.Accuracy).On(Part.Test).As(Shown.Numbers));
        var test = Predicted(prepared, Part.Test, Guessed);

        Assert.Throws<ArgumentException>(() => prepared.Measure([test with { Predictions = [.. test.Predictions.Skip(1)] }]));
        Assert.Throws<ArgumentException>(() => prepared.Measure([test with { Predictions = [.. test.Predictions.Select(_ => new double[] { 0.5, 0.5 })] }]));
    }

    [Fact]
    public void WhatAModelSaysItWasHandedAndLearnedNothingAbout_IsCountedPartByPart_AndWhatItDoesNotSay_IsNotCountedAtAll()
    {
        // A model may say, of each row it predicted, which features it was handed a value of that it learned nothing about;
        // the report counts the rows of each part that name any, and leaves a part whose predictions say nothing uncounted.
        var prepared = Guessing(report => report.Measure(Metric.Accuracy).On(Part.Validation, Part.Test).As(Shown.Numbers));
        var validation = Predicted(prepared, Part.Validation, Guessed);
        var test = Predicted(prepared, Part.Test, Guessed);
        IReadOnlyList<string>[] said = [.. Enumerable.Range(0, validation.Batch.RowCount).Select(row => row % 40 == 0 ? (IReadOnlyList<string>)["sex_other", "pclass_other"] : [])];

        var measures = prepared.Measure([validation with { Unfamiliar = said }, test]);

        Assert.Equal(133, measures.Parts[0].Rows);
        Assert.Equal<int?>([4, null], measures.Parts.Select(part => part.UnfamiliarRows));
        Assert.Equal(prepared.Measure([validation, test]).Parts.SelectMany(part => part.Values), measures.Parts.SelectMany(part => part.Values));
    }

    [Fact]
    public void WhatAModelSaysOfOtherRowsThanThePartHandsOver_IsRefused()
    {
        var prepared = Guessing(report => report.Measure(Metric.Accuracy).On(Part.Test).As(Shown.Numbers));
        var test = Predicted(prepared, Part.Test, Guessed);

        var refused = Assert.Throws<ArgumentException>(() => prepared.Measure([test with { Unfamiliar = [[], ["sex_other"]] }]));

        Assert.Contains("'test'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("2 rows", refused.Message, StringComparison.Ordinal);
        Assert.Contains("135", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APredictionThatIsNotAFiniteNumber_IsRefused_NamingItsRow()
    {
        var prepared = Guessing(report => report.Measure(Metric.Accuracy).On(Part.Test).As(Shown.Numbers));
        var test = Predicted(prepared, Part.Test, Guessed);

        var refused = Assert.Throws<ArgumentException>(() => prepared.Measure([test with { Predictions = [[double.NaN], .. test.Predictions.Skip(1)] }]));

        Assert.Contains("Row ", refused.Message, StringComparison.Ordinal);
        Assert.Contains("not a finite number", refused.Message, StringComparison.Ordinal);
    }

    private static PreparedData Ten(string answers, Action<ReportBuilder> report, double train = 0.50) =>
        Pdd.Create()
            .Read(CsvRowSource.FromText("t,y\n" + string.Join('\n', answers.Split(',').Select((y, at) => $"{at + 1},{y}")) + "\n"), "ten rows")
            .Declare(schema => schema.Integer("t", "y"))
            .SplitByTime("t", train)
            .Target("y")
            .Report(report)
            .Build()
            .Run();

    [Fact]
    public void ATargetHoldingATwo_IsRefusedWhenItsClassesAreCounted_NamingTheRowAsRead()
    {
        // Whether a column's answers are noughts and ones is known only from its rows, so it is asked where they are
        // measured; an amount measured of the same rows is taken as it is.
        const string answers = "1,0,1,2,0,1,0,1,0,1";

        var refused = Assert.Throws<InvalidOperationException>(() => Measured(
            Ten(answers, report => report.Measure(Metric.Rmse, Metric.Accuracy).On(Part.Train, Part.Test).As(Shown.Numbers)), (batch, _) => [0.5]));

        Assert.Contains("Row 4", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'y'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("accuracy", refused.Message, StringComparison.Ordinal);

        var amounts = Measured(Ten(answers, report => report.Measure(Metric.Rmse).On(Part.Train, Part.Test).As(Shown.Numbers)), (batch, _) => [0.5]);

        Assert.Equal(Math.Sqrt((0.25 * 4) + 2.25) / Math.Sqrt(5), amounts.Parts[0].Values[0].Value, Close);
    }

    [Fact]
    public void AnOutputFromElsewhere_WhoseRowHoldsOtherThanTheOnesItSays_IsRefusedWhenItsClassesAreCounted()
    {
        // The labels this library ships refuse such a row where it is handed over; an output a package brings is held to
        // what it says where its classes are counted.
        var prepared = new Pipeline(
            new PipelineDeclaration(
            [
                new ReadRowsStep("three numbers"),
                new DeclareStep([.. new[] { "t", "b", "c" }.Select(name => new ColumnDeclaration(name, ColumnKind.Number, Optional: false))]),
                new SplitByTimeStep("t", new SplitShares(0.50, 0, 0.50)),
                new NamesTheseAnswers("b", "c") { Ones = 1 },
                new ReportStep([Metric.Accuracy], [Part.Train, Part.Test], [Shown.Numbers]),
            ]),
            new InMemoryRowSource(["t", "b", "c"], [["1", "1", "0"], ["2", "1", "1"], ["3", "0", "1"], ["4", "0", "1"]])).Run();

        var refused = Assert.Throws<InvalidOperationException>(() => Measured(prepared, (_, _) => [0.5, 0.5]));

        Assert.Contains("Row 2 holds 2 of 'b', 'c' as one", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'test.answers' says a row holds 1", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnswersThatNeverVary_HaveAnR2OfNought_UnlessThePredictionsAreThem()
    {
        // As scikit-learn keeps it finite: nothing is left over, one; something is, where there was no spread to explain, nought.
        var constant = Ten("0,1,0,1,0,1,1,1,1,1", report => report.Measure(Metric.R2).On(Part.Test).As(Shown.Numbers));

        Assert.Equal(0, Measured(constant, (_, _) => [0.5]).Parts[0].Values[0].Value);
        Assert.Equal(1, Measured(constant, (_, _) => [1]).Parts[0].Values[0].Value);
    }

    [Fact]
    public void APartOfOneRow_HasNoSpreadForR2_AndAPartOfNoneIsRefused()
    {
        var one = Ten("1,0", report => report.Measure(Metric.R2).On(Part.Test).As(Shown.Numbers));

        Assert.Contains("R²", Assert.Throws<InvalidOperationException>(() => Measured(one, (batch, _) => [0.5])).Message, StringComparison.Ordinal);
        Assert.Single(Measured(Ten("1,0", report => report.Measure(Metric.Mae).On(Part.Test).As(Shown.Numbers)), (batch, _) => [0.5]).Parts);

        var none = Ten("1,0,1,0", report => report.Measure(Metric.Mae).On(Part.Validation).As(Shown.Numbers));

        Assert.Contains("no rows", Assert.Throws<InvalidOperationException>(() => Measured(none, (batch, _) => [0.5])).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ATrainingPartOfNoRows_LeavesNoAverageToStandBeside()
    {
        var untrained = Ten("1,0,1", report => report.Measure(Metric.Mae).On(Part.Test).As(Shown.Numbers), train: 0.1);

        Assert.Equal(0, untrained.CountIn(Part.Train));
        Assert.Contains("training", Assert.Throws<InvalidOperationException>(() => Measured(untrained, (batch, _) => [0.5])).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineWithNoReport_HasNothingToMeasure_AndOneFromAFileHasNoRows()
    {
        var unreported = Pdd.Create().ReadCsv(Repository.Data("titanic.csv")).Declare(schema => schema.Integer("survived", "pclass")).SplitStratified("survived", 0.70, 0.15).Target("survived").Build().Run();

        Assert.Contains("report", Assert.Throws<InvalidOperationException>(() => unreported.Measure([])).Message, StringComparison.Ordinal);

        // A pipeline loaded from its file keeps no rows, so what a model predicted is measured on the run that fitted it.
        var trained = Guessing();
        var loaded = PreparedData.FromJson(trained.ToJson(), StepCatalog.BuiltIn());
        var refused = Assert.Throws<InvalidOperationException>(() => loaded.Measure([.. ThreeParts.Select(part => Predicted(trained, part, Guessed))]));

        Assert.Contains("holds no rows", refused.Message, StringComparison.Ordinal);
        Assert.Contains("file", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARunForALearnerThatTakesCategories_IsMeasuredOnTheSameRows_ToTheSameMeasures()
    {
        // The report reads the rows' keys and answers, never the features a learner was handed: the same guess, made from the
        // sex handed over one-hot or as its place, measures the same on both runs.
        var titanic = WikiTitanic.In(WikiTitanic.DataFolder);
        var every = titanic.Run();
        var places = titanic.RunFor(Needs.Categories);

        var oneHot = every.Measure([.. ThreeParts.Select(part => Guess(every.Batch(part), "sex_female", female: 1))]);
        var asPlaces = places.Measure([.. ThreeParts.Select(part => Guess(places.Batch(part, Needs.Categories), "sex", female: 0))]);

        Assert.Equal([623, 133, 135], asPlaces.Parts.Select(part => part.Rows));
        Assert.Equal(oneHot.Parts.SelectMany(part => part.Values), asPlaces.Parts.SelectMany(part => part.Values));

        static PartPredictions Guess(Batch batch, string sex, double female) =>
            new(batch, [.. batch.Features.Select(row => new[] { row[At(batch, sex)] == female ? 0.74 : 0.19 })]);
    }

    [Fact]
    public void APartAFeatureOfWhichStillHoldsWords_IsMeasured_ForTheReportReadsNoFeature()
    {
        // No learner of numbers is handed a column of words, and the report is none: what a learner handed something else
        // predicted is measured on the rows' keys and answers alone.
        var words = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Category("sex"))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Report(report => report.Measure(Metric.Accuracy).On(Part.Test).As(Shown.Numbers))
            .Build()
            .Run();
        RowKey[] keys = [.. Enumerable.Range(0, words.Table.RowCount).Where(row => words.Parts[row] == Part.Test).Select(row => words.Table.Identities[row].Key)];
        var batch = new Batch(["sex"], [.. keys.Select(_ => new[] { 0.0 })], null) { Part = Part.Test, Keys = keys };

        var measures = words.Measure([new PartPredictions(batch, [.. keys.Select(_ => new[] { 0.0 })])]);

        Assert.Contains("still holds words", Assert.Throws<InvalidOperationException>(() => words.Batch(Part.Test)).Message, StringComparison.Ordinal);
        Assert.Equal(135, Assert.Single(measures.Parts).Rows);
        Assert.InRange(Assert.Single(measures.Parts[0].Values).Value, 0.5, 0.7);
    }

    [Fact]
    public void MeasuresAreVisitedAsWhatTheyAre()
    {
        var measures = Measured(Guessing(report => report.Measure(Metric.Accuracy).On(Part.Test).As(Shown.Drawn)), Guessed);

        Assert.Equal("measures", measures.Accept(new EvidenceKind()));
        Assert.Throws<ArgumentNullException>(() => measures.Accept<string>(null!));
    }

    /// <summary>Says which kind of evidence it was handed.</summary>
    private sealed class EvidenceKind : IEvidenceVisitor<string>
    {
        public string Visit(DataProfile profile) => "profile";

        public string Visit(CorrelationInput correlation) => "correlation";

        public string Visit(Measures measures) => "measures";
    }
}
