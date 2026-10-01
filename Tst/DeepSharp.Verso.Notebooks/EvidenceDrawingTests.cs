// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Charts;
using DeepSharp.Verso.Notebooks;
using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Numerics;
using MatPlotLibNet.Rendering.TickFormatters;
using MatPlotLibNet.Rendering.TickLocators;
using MatPlotLibNet.Styling.ColorMaps;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A block that declares evidence shows it under its grid: a profile of the columns with every alert naming the
/// step that answers it and the rows that are there more than once, or the correlation of some columns — drawn,
/// or written as numbers — with how many rows it was drawn from, out of how many, and by which rule. The numbers
/// are the core's, measured over the training rows of the split below; the notebook only draws them.
/// </summary>
public sealed class EvidenceDrawingTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-evidence-").FullName;

    public EvidenceDrawingTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private const string Read = """{"step": "read.csv", "path": "titanic.csv"}""";
    private const string Declare = """{"step": "declare", "remainder": "keep", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}, {"name": "sibsp", "kind": "integer", "optional": false}]}""";
    private const string Split = """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""";

    private async Task<string> ShownAtAsync(string evidence)
    {
        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in new[] { Read, Declare, evidence, Split })
        {
            notebook.AddBlock(block);
        }

        var cell = notebook.Scaffold.Cells[2];

        await notebook.GestureAsync(cell, StepRenderer.Show);

        Assert.Equal(3, cell.Outputs.Count);

        return cell.Outputs[2].Content;
    }

    private Evidence Measured(string evidence)
    {
        var declaration = new PipelineDeclaration([.. new[] { Read, Declare, evidence, Split }.Select(block => StepCatalog.BuiltIn().ReadStep(block))]);

        return new Pipeline(declaration, rows: null, SourceFolder.Of(_folder)).ViewAt(3).Evidence[2];
    }

    [Fact]
    public async Task AProfile_ShowsWhatTheCoreMeasured_EveryAlertWithTheStepThatAnswersIt_AndTheRowsThereTwice()
    {
        const string evidence = """{"step": "evidence.profile"}""";

        var shown = await ShownAtAsync(evidence);
        var profile = Assert.IsType<DataProfile>(Measured(evidence));

        Assert.Contains(string.Create(CultureInfo.InvariantCulture, $"{profile.Rows} training rows"), shown, StringComparison.Ordinal);
        Assert.All(profile.Columns, column => Assert.Contains($">{column.Name}<", shown, StringComparison.Ordinal));
        Assert.All(profile.Alerts, alert => Assert.Contains(
            alert.Answer.Action switch
            {
                AlertAction.LeaveOut => "answered by leaving it out",
                AlertAction.SayMissing => $"that <code>{alert.Answer.Value}</code> stands for a gap",
                _ => $"answered by <code>{alert.Answer.Verb}</code>",
            },
            shown,
            StringComparison.Ordinal));
        Assert.Contains(profile.Alerts, alert => alert.Answer.Action == AlertAction.LeaveOut);
        Assert.Contains(string.Create(CultureInfo.InvariantCulture, $"{profile.Duplicates.Groups} groups of rows"), shown, StringComparison.Ordinal);
        Assert.Contains(string.Create(CultureInfo.InvariantCulture, $"{profile.Duplicates.ExtraCopies} extra copies"), shown, StringComparison.Ordinal);
        Assert.Contains(string.Create(CultureInfo.InvariantCulture, $"{profile.Duplicates.GroupsAcrossParts} of the groups"), shown, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnThatNeverChanges_IsMarkedConstant_AndItsAlertLeavesItOut()
    {
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText("k,v\n1,2\n1,3\n"), "two rows")
            .Declare(schema => schema.Integer("k", "v"))
            .Profile()
            .Build()
            .Run();

        var shown = prepared.Evidence[2].Accept(new EvidenceView()).Content;

        Assert.Contains("<td>integer, constant</td>", shown, StringComparison.Ordinal);
        Assert.Contains("<code>k</code> Every row holds the same value, so it tells a model nothing. — answered by leaving it out", shown, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ACorrelation_IsDrawn_WithTheRowsItWasDrawnFrom_OutOfHowMany_AndTheRule()
    {
        const string evidence = """{"step": "evidence.correlation", "columns": ["age", "fare", "sibsp"], "shown": "drawn"}""";

        var shown = await ShownAtAsync(evidence);
        var input = Assert.IsType<CorrelationInput>(Measured(evidence));

        Assert.Contains("<svg", shown, StringComparison.Ordinal);
        Assert.Contains(string.Create(CultureInfo.InvariantCulture, $"{input.Kept} of the {input.Total} training rows"), shown, StringComparison.Ordinal);
        Assert.Contains(input.Policy, shown, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ACorrelation_IsDrawnAsItAlwaysWas_WhoeverDrawsIt()
    {
        const string evidence = """{"step": "evidence.correlation", "columns": ["age", "fare", "sibsp"], "shown": "drawn"}""";

        var shown = await ShownAtAsync(evidence);
        var input = Assert.IsType<CorrelationInput>(Measured(evidence));

        Assert.Contains(AsTheNotebookDrewIt(input), shown, StringComparison.Ordinal);
    }

    // The heatmap the notebook drew a correlation as, kept here exactly as it was written: the picture a move of the drawing
    // must leave as it stood.
    private static string AsTheNotebookDrewIt(CorrelationInput correlation)
    {
        var names = correlation.Columns.ToArray();
        var matrix = NpStats.Corrcoef([.. Enumerable.Range(0, names.Length).Select(column => correlation.Rows.Select(row => row[column]).ToArray())]);
        var data = new double[names.Length, names.Length];
        var positions = Enumerable.Range(0, names.Length).Select(position => (double)position).ToArray();

        for (var row = 0; row < names.Length; row++)
        {
            for (var column = 0; column < names.Length; column++)
            {
                data[row, column] = matrix[row, column];
            }
        }

        var size = 160 + (60 * names.Length);

        return new FigureBuilder()
            .WithSize(size + 120, size)
            .AddSubPlot(1, 1, 1, axes => axes
                .Heatmap(data, series =>
                {
                    series.ColorMap = ColorMaps.Coolwarm;
                    series.Normalizer = new MinusOneToOne();
                    series.ShowLabels = true;
                    series.LabelFormat = "0.00";
                })
                .SetXTickLocator(new FixedLocator(positions))
                .SetXTickFormatter(new CategoryFormatter(names))
                .SetYTickLocator(new FixedLocator(positions))
                .SetYTickFormatter(new CategoryFormatter(names, reversed: true))
                .WithColorBar())
            .ToSvg();
    }

    // The whole of a correlation's scale, minus one to one, whatever the coefficients span.
    private sealed class MinusOneToOne : INormalizer
    {
        public double Normalize(double value, double min, double max) => Math.Clamp((value + 1) / 2, 0, 1);
    }

    [Fact]
    public async Task ACorrelationAskedForAsNumbers_WritesEachCoefficient()
    {
        const string evidence = """{"step": "evidence.correlation", "columns": ["age", "fare"], "shown": "numbers"}""";

        var shown = await ShownAtAsync(evidence);
        var input = Assert.IsType<CorrelationInput>(Measured(evidence));
        var matrix = NpStats.Corrcoef([.. Enumerable.Range(0, 2).Select(column => input.Rows.Select(row => row[column]).ToArray())]);

        Assert.DoesNotContain("<svg", shown, StringComparison.Ordinal);
        Assert.Contains($">{matrix[0, 1].ToString("0.000", CultureInfo.InvariantCulture)}<", shown, StringComparison.Ordinal);
        Assert.Contains(">1.000<", shown, StringComparison.Ordinal);
    }

    [Fact]
    public void ACorrelationOverTooFewCompleteRows_SaysSoRatherThanDrawingNothing()
    {
        var view = new Pipeline(
                Pdd.Create().Read(CsvRowSource.FromText("a,b\n1,\n,2\n3,4\n"), "rows")
                    .Declare(schema => schema.Optional("a", ColumnKind.Number).Optional("b", ColumnKind.Number))
                    .Correlation(["a", "b"])
                    .Declaration,
                CsvRowSource.FromText("a,b\n1,\n,2\n3,4\n"))
            .ViewAt(3);

        var shown = view.Evidence[2].Accept(new EvidenceView()).Content;

        Assert.Contains("1 of the 3 rows", shown, StringComparison.Ordinal);
        Assert.Contains("too few", shown, StringComparison.Ordinal);
        Assert.DoesNotContain("<svg", shown, StringComparison.Ordinal);
    }

    // Validation and test predicted by a guess from the class alone; the report as a pipeline declares it.
    private static Measures Guessed(Shown shown, Shown? also = null, Metric? only = null)
    {
        Metric[] metrics = only is { } one ? [one] : [Metric.Accuracy, Metric.ConfusionMatrix, Metric.Rmse];
        Shown[] ways = also is { } other ? [shown, other] : [shown];
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass"))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Report(report => report.Measure(metrics).On(Part.Validation, Part.Test).As(ways))
            .Build()
            .Run();

        return prepared.Measure([.. new[] { Part.Validation, Part.Test }.Select(part => Predicted(prepared.Batch(part), row => [row[0] == 1 ? 0.6 : 0.2]))]);
    }

    private static PartPredictions Predicted(Batch batch, Func<double[], double[]> model) => new(batch, [.. batch.Features.Select(model)]);

    [Fact]
    public void AModelsMeasures_AreDrawnAsTheChartsPackageRendersTheReport_TheOneRenderingACellShowsToo()
    {
        // The notebook renders no report of its own: numbers, drawings or both, it shows what DeepSharp.Charts renders, so
        // the report under a block and the one a C# cell ends with are one and the same.
        foreach (var measures in new[] { Guessed(Shown.Numbers), Guessed(Shown.Drawn), Guessed(Shown.Numbers, Shown.Drawn) })
        {
            var drawn = measures.Accept(new EvidenceView());

            Assert.Equal("text/html", drawn.MimeType);
            Assert.Equal(measures.Report().ToHtml(), drawn.Content);
        }

        Assert.Throws<ArgumentNullException>(() => new EvidenceView().Visit((Measures)null!));
    }

    [Fact]
    public void EvidenceIsWrittenTheSameWayWhateverLanguageTheInterfaceSpeaks()
    {
        var view = new Pipeline(
                Pdd.Create().Read(CsvRowSource.FromText("a,b\n1.5,2\n2.5,3\n4.25,1\n"), "rows")
                    .Declare(schema => schema.Number("a", "b"))
                    .Profile()
                    .Correlation(["a", "b"])
                    .Declaration,
                CsvRowSource.FromText("a,b\n1.5,2\n2.5,3\n4.25,1\n"))
            .ViewAt(4);
        var before = CultureInfo.CurrentCulture;

        try
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            var invariant = view.Evidence.Values.Select(evidence => evidence.Accept(new EvidenceView()).Content).ToArray();

            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("nl-NL");
            var dutch = view.Evidence.Values.Select(evidence => evidence.Accept(new EvidenceView()).Content).ToArray();

            Assert.Equal(invariant, dutch);
            Assert.Contains(">2.75<", dutch[0], StringComparison.Ordinal);
        }
        finally
        {
            CultureInfo.CurrentCulture = before;
        }
    }
}
