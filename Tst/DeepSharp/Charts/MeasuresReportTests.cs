// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Reflection;
using DeepSharp.Charts;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Charts;

/// <summary>
/// A trained model's measures as its pipeline's report says they are shown — the numbers, the charts, or both — are one
/// piece of HTML, drawn in one place: a notebook's C# cell that ends with it shows it, and a notebook's report block shows
/// the same. As numbers, a part to a row, each measure beside predicting the training rows' average, and beside the rows
/// how many of them the model said it learned nothing about; then each confusion matrix. Drawn, the charts of the charts
/// package, as they are drawn anywhere else.
/// </summary>
public class MeasuresReportTests
{
    // Validation and test predicted by a guess from the class alone; the report as a pipeline declares it.
    private static Measures Guessed(Shown shown, Shown? also = null, Metric? only = null)
    {
        Metric[] metrics = only is { } one ? [one] : [Metric.Accuracy, Metric.ConfusionMatrix, Metric.Rmse];
        Shown[] ways = also is { } other ? [shown, other] : [shown];
        var prepared = Passengers(report => report.Measure(metrics).On(Part.Validation, Part.Test).As(ways));

        return prepared.Measure([.. new[] { Part.Validation, Part.Test }.Select(part => Predicted(prepared.Batch(part)))]);
    }

    private static PreparedData Passengers(Action<ReportBuilder> report) =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass"))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Report(report)
            .Build()
            .Run();

    private static PartPredictions Predicted(Batch batch) => new(batch, [.. batch.Features.Select(row => new[] { row[0] == 1 ? 0.6 : 0.2 })]);

    private static string Invariant(double value) => value.ToString("G6", CultureInfo.InvariantCulture);

    [Fact]
    public void AsNumbers_EachMeasureIsWrittenBesideTheTrainingRowsAverage_ThenEachConfusionMatrix()
    {
        var measures = Guessed(Shown.Numbers);
        var written = measures.Report().ToHtml();
        var validation = measures.Parts[0];
        var confusion = Assert.Single(validation.Confusions);

        Assert.Contains("Measures of survived", written, StringComparison.Ordinal);
        Assert.Contains("<th>accuracy</th><th>baseline</th><th>rmse</th><th>baseline</th>", written, StringComparison.Ordinal);
        Assert.Contains(
            $"<td>validation</td><td class=\"deepsharp-number\">133</td><td class=\"deepsharp-number\">{Invariant(validation.Values[0].Value)}</td>"
            + $"<td class=\"deepsharp-number\">{Invariant(validation.Values[0].Baseline)}</td>",
            written,
            StringComparison.Ordinal);
        Assert.Contains("<td>test</td>", written, StringComparison.Ordinal);
        Assert.Contains("<div>validation: survived</div>", written, StringComparison.Ordinal);
        Assert.Contains(
            string.Create(CultureInfo.InvariantCulture, $">{confusion.Counts[1][0]} ({confusion.Baseline[1][0]})<"),
            written,
            StringComparison.Ordinal);
        Assert.DoesNotContain("<svg", written, StringComparison.Ordinal);
        Assert.DoesNotContain("unfamiliar", written, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TheRowsAModelSaidItLearnedNothingAbout_AreCountedBesideEachPartsRows_BlankWhereItSaidNothing()
    {
        // The validation rows' predictions say which features each row moved; the test rows' say nothing, and nothing is
        // written for them.
        var prepared = Passengers(report => report.Measure(Metric.Accuracy).On(Part.Validation, Part.Test).As(Shown.Numbers));
        var validation = Predicted(prepared.Batch(Part.Validation));
        IReadOnlyList<string>[] said = [.. Enumerable.Range(0, validation.Batch.RowCount).Select(row => row < 3 ? (IReadOnlyList<string>)["sex_other"] : [])];
        var measures = prepared.Measure([validation with { Unfamiliar = said }, Predicted(prepared.Batch(Part.Test))]);

        var written = measures.Report().ToHtml();

        Assert.Contains("<th>part</th><th>rows</th><th>unfamiliar</th><th>accuracy</th>", written, StringComparison.Ordinal);
        Assert.Contains("<td>validation</td><td class=\"deepsharp-number\">133</td><td class=\"deepsharp-number\">3</td>", written, StringComparison.Ordinal);
        Assert.Contains("<td>test</td><td class=\"deepsharp-number\">135</td><td class=\"deepsharp-number\"></td>", written, StringComparison.Ordinal);
        Assert.Contains("<div>Unfamiliar counts the rows", written, StringComparison.Ordinal);
        Assert.DoesNotContain("confusion matrix counts", written, StringComparison.Ordinal);
    }

    [Fact]
    public void Drawn_TheyAreTheChartsPackagesCharts_AndAskedForBoth_TheyAreBoth()
    {
        // Every measure as bars, the confusion matrices as heatmaps, and — rmse being an amount — what was predicted
        // against what was there, and what was left over: the charts DeepSharp.Charts draws, wherever they are shown.
        var measures = Guessed(Shown.Drawn);
        var drawn = measures.Report().ToHtml();
        var charts = measures.Bars() + measures.ConfusionMatrices() + measures.PredictedAgainstActual() + measures.Residuals();

        Assert.Contains(charts, drawn, StringComparison.Ordinal);
        Assert.DoesNotContain("<table>", drawn, StringComparison.Ordinal);

        var both = Guessed(Shown.Numbers, Shown.Drawn).Report().ToHtml();

        Assert.Contains("<th>accuracy</th><th>baseline</th>", both, StringComparison.Ordinal);
        Assert.Contains(charts, both, StringComparison.Ordinal);

        // A report of the confusion matrix alone has no number for bars, and no amount to draw against its answers.
        var matrices = Guessed(Shown.Drawn, only: Metric.ConfusionMatrix);

        Assert.Equal(matrices.ConfusionMatrices().TrimEnd(), Assert.Single(Svgs(matrices.Report().ToHtml())));
    }

    // Every SVG a piece of HTML holds, in order.
    private static IEnumerable<string> Svgs(string html)
    {
        for (var at = html.IndexOf("<svg", StringComparison.Ordinal); at >= 0; at = html.IndexOf("<svg", at + 1, StringComparison.Ordinal))
        {
            var end = html.IndexOf("</svg>", at, StringComparison.Ordinal) + "</svg>".Length;

            yield return html[at..end];
        }
    }

    [Fact]
    public void LabelsOfWhichARowHoldsOne_AreCountedInOneMatrixAcrossThem()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["t", "a", "b"], [.. Enumerable.Range(1, 8).Select(t => (IReadOnlyList<string?>)[$"{t}", t % 2 == 0 ? "1" : "0", t % 2 == 0 ? "0" : "1"])]), "eight rows")
            .Declare(schema => schema.Integer("t", "a", "b"))
            .SplitByTime("t", 0.50)
            .Labels(["a", "b"], ones: 1)
            .Report(report => report.Measure(Metric.ConfusionMatrix).On(Part.Test).As(Shown.Numbers))
            .Build()
            .Run();

        // Every test row predicted b; the training rows hold a and b alike, so their average predicts a, the first of equals.
        var test = prepared.Batch(Part.Test);
        var written = prepared.Measure([new PartPredictions(test, [.. test.Features.Select(_ => new[] { 0.3, 0.7 })])]).Report().ToHtml();

        Assert.Contains("<div>test</div>", written, StringComparison.Ordinal);
        Assert.Contains("<th>a</th><th>b</th>", written, StringComparison.Ordinal);
        Assert.Contains("<tr><th>b</th><td class=\"deepsharp-number\">0 (2)</td><td class=\"deepsharp-number\">2 (0)</td></tr>", written, StringComparison.Ordinal);
    }

    [Fact]
    public void TheReport_IsWrittenTheSameWayWhateverLanguageTheInterfaceSpeaks()
    {
        var measures = Guessed(Shown.Numbers);
        var before = CultureInfo.CurrentCulture;

        try
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            var invariant = measures.Report().ToHtml();

            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("nl-NL");

            Assert.Equal(invariant, measures.Report().ToHtml());
            Assert.Contains(Invariant(measures.Parts[0].Values[0].Value), invariant, StringComparison.Ordinal);
        }
        finally
        {
            CultureInfo.CurrentCulture = before;
        }
    }

    [Fact]
    public void AnAnswerNamedInWordsAPageCouldRead_IsWrittenAsText()
    {
        // The names come from a pipeline, which a person wrote: none reaches the page as markup.
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText("t,<b>\n" + string.Join('\n', Enumerable.Range(1, 12).Select(t => $"{t},{t % 2}")) + "\n"), "twelve rows")
            .Declare(schema => schema.Integer("t", "<b>"))
            .SplitByTime("t", 0.50, 0.25)
            .Target("<b>")
            .Report(report => report.Measure(Metric.ConfusionMatrix).On(Part.Test).As(Shown.Numbers))
            .Build()
            .Run();
        var test = prepared.Batch(Part.Test);

        var written = prepared.Measure([new PartPredictions(test, [.. test.Features.Select(_ => new[] { 0.7 })])]).Report().ToHtml();

        Assert.Contains("Measures of &lt;b&gt;", written, StringComparison.Ordinal);
        Assert.Contains("<div>test: &lt;b&gt;</div>", written, StringComparison.Ordinal);
        Assert.DoesNotContain("<b>", written, StringComparison.Ordinal);
    }

    [Fact]
    public void TheReport_IsAValueANotebookShowsAsHtml_ByItsPublicToHtml()
    {
        // Verso's engine shows any value whose type has a public, parameterless ToHtml() as HTML, found by reflection so the
        // assembly a cell loaded the type from does not matter; this is the method it finds.
        var found = typeof(MeasuresReport).GetMethod("ToHtml", BindingFlags.Instance | BindingFlags.Public, null, Type.EmptyTypes, null);

        Assert.NotNull(found);
        Assert.Equal(typeof(string), found.ReturnType);
        Assert.Equal(Guessed(Shown.Numbers).Report().ToHtml(), found.Invoke(Guessed(Shown.Numbers).Report(), null));
        Assert.Throws<ArgumentNullException>(() => ((Measures)null!).Report());
    }
}
