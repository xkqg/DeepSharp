// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Charts;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Microsoft.CodeAnalysis.CSharp;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A report block names what a trained model is held to, and the notebook trains none: the measures reach the block from
/// the C# cell that trains one, which hands its predictions back as text, under a key no C# variable can have. "Show the
/// data here" at the block then measures them on the notebook's own run of its blocks and draws them under the grid, as
/// the report says they are shown — the rendering a cell that ends with the report shows too. Predictions made behind any
/// other fit than the blocks make now — a block edited since the cell ran — are refused there, saying so, and so are those
/// of another answer; with nothing handed back, the block says where its measures come from. A report drawn once is drawn
/// again from memory while what it was measured from stays, and a show runs the whole pipeline once at most.
/// </summary>
public sealed class ReportBlockTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-report-").FullName;

    public ReportBlockTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private const string Split = """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""";

    private const string Fare = """{"step": "normalise", "column": "fare", "scale": "midrange", "outOfRange": "pass"}""";

    // The class and the fare, the fare scaled onto a range, and whether a passenger survived; the report is the last block.
    private static readonly string[] Blocks =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "fare", "kind": "number", "optional": false}]}""",
        Split,
        Fare,
        """{"step": "target", "column": "survived"}""",
        """{"step": "evidence.report", "metrics": ["accuracy", "confusionmatrix"], "parts": ["validation", "test"], "shown": ["numbers", "drawn"]}""",
    ];

    private async Task<Notebook> NotebookAsync(params string[] blocks)
    {
        var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in blocks.Length > 0 ? blocks : Blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private static NotebookSession Session(Notebook notebook) =>
        notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;

    // The report is the last block; a C# cell added after it is no block.
    private static CellModel Report(Notebook notebook) => notebook.Scaffold.Cells.Last(cell => cell.Type == StepCellType.StepType);

    // A run of blocks over the file beside the notebook, as the notebook runs them.
    private PreparedData Run(params string[] blocks) =>
        new Pipeline(new PipelineDeclaration([.. blocks.Select(block => NotebookVerbs.Catalog().ReadStep(block))]), rows: null, SourceFolder.Of(_folder)).Run();

    // A guess from the class alone for every part the report names; of the first rows of each part, when said, the model
    // says it learned nothing about the class.
    private static Measures Guessed(PreparedData prepared, int? unfamiliar = null) =>
        prepared.Measure([.. prepared.Declaration.Report!.Parts.Select(part =>
        {
            var batch = prepared.Batch(part);

            return new PartPredictions(batch, [.. batch.Features.Select(row => new[] { row[0] == 1 ? 0.6 : 0.2 })])
            {
                Unfamiliar = unfamiliar is { } rows ? [.. Enumerable.Range(0, batch.RowCount).Select(row => row < rows ? (IReadOnlyList<string>)["pclass"] : [])] : null,
            };
        })]);

    private static async Task<string> ShownAtTheReportAsync(Notebook notebook)
    {
        var report = Report(notebook);

        await notebook.GestureAsync(report, StepRenderer.Show);

        Assert.Equal(3, report.Outputs.Count);
        Assert.True(report.Outputs[1].Content.Contains("891 rows", StringComparison.Ordinal), report.Outputs[1].Content);

        return report.Outputs[2].Content;
    }

    [Fact]
    public async Task WithNothingHandedBack_TheBlockNamesItsMeasures_AndSaysWhereTheyComeFrom()
    {
        await using var notebook = await NotebookAsync();

        var shown = await ShownAtTheReportAsync(notebook);

        Assert.False(Report(notebook).Outputs[2].IsError);
        Assert.Contains("accuracy and confusionmatrix on validation and test, shown as numbers and drawn", shown, StringComparison.Ordinal);
        Assert.Contains("trains none", shown, StringComparison.Ordinal);
        Assert.Contains("Variables.Set(&quot;deepsharp.predictions&quot;, trained.Measures!.PredictionsToJson())", shown, StringComparison.Ordinal);
        Assert.Contains("trained.Measures!.Report()", shown, StringComparison.Ordinal);
        Assert.Equal(0, Session(notebook).ReportsMeasured);
        Assert.Equal(0, Session(notebook).WholeRuns);

        // Something handed back under the key that is not text is nothing to measure either; and a report of one measure, on
        // one part, shown one way, says so.
        notebook.Scaffold.Variables.Set(StepKernel.HandedBack, 42);
        Report(notebook).Source = """{"step": "evidence.report", "metrics": ["rmse"], "parts": ["test"], "shown": ["numbers"]}""";

        Assert.Contains("This report measures rmse on test, shown as numbers.", await ShownAtTheReportAsync(notebook), StringComparison.Ordinal);
    }

    [Fact]
    public async Task PredictionsACSharpCellHandsBack_AreMeasuredOnTheNotebooksOwnRun_AndDrawnAsTheReportRendersThem()
    {
        await using var notebook = await NotebookAsync();

        await ShownAtTheReportAsync(notebook);

        // The cell trains nothing here, it guesses; what it hands back is what a cell that trains hands back, made behind the
        // fit of the pipeline it was handed — its own run of the same blocks over the same file.
        var cell = notebook.Scaffold.AddCell("code", "csharp", $$"""
            #r "{{Path.Join(AppContext.BaseDirectory, "DeepSharp.Pipelines.dll")}}"
            using DeepSharp.Pipelines;

            var text = Variables.Get<string>("deepsharp.pipeline");
            var folder = Variables.TryGet<string>("deepsharp.folder", out var saved) ? SourceFolder.Of(saved) : SourceFolder.WorkingDirectory;
            var prepared = new Pipeline(PipelineDeclaration.FromJson(text, StepCatalog.BuiltIn()), null, folder).Run();
            var measures = prepared.Measure(prepared.Declaration.Report.Parts.Select(part =>
            {
                var batch = prepared.Batch(part);

                return new PartPredictions(batch, batch.Features.Select(row => new[] { row[0] == 1 ? 0.6 : 0.2 }).ToArray());
            }).ToArray());

            Variables.Set("deepsharp.predictions", measures.PredictionsToJson());
            """);

        await notebook.RunAsync(cell);

        var drawn = await ShownAtTheReportAsync(notebook);

        Assert.False(Report(notebook).Outputs[2].IsError, drawn);
        Assert.Equal("text/html", Report(notebook).Outputs[2].MimeType);
        Assert.Equal(Guessed(Run(Blocks)).Report().ToHtml(), drawn);
        Assert.Contains("<th>accuracy</th><th>baseline</th>", drawn, StringComparison.Ordinal);
        Assert.Contains("<svg", drawn, StringComparison.Ordinal);
    }

    [Fact]
    public async Task PredictionsHandedBackOverAParquetFile_AreMeasuredOnTheNotebooksOwnRunOfIt()
    {
        // The report's block measures on the notebook's own run of its blocks whichever file the first block reads: the
        // passenger list read from its Parquet file is measured as its comma-separated file is.
        File.Copy(Repository.Fixture("titanic.parquet"), Path.Join(_folder, "titanic.parquet"));
        string[] blocks = ["""{"step": "read.parquet", "path": "titanic.parquet"}""", .. Blocks[1..]];
        await using var notebook = await NotebookAsync(blocks);

        notebook.Scaffold.Variables.Set(StepKernel.HandedBack, Guessed(Run(blocks)).PredictionsToJson());

        var drawn = await ShownAtTheReportAsync(notebook);

        Assert.False(Report(notebook).Outputs[2].IsError, drawn);
        Assert.Equal(Guessed(Run(blocks)).Report().ToHtml(), drawn);
        Assert.Equal(Guessed(Run(Blocks)).Report().ToHtml(), drawn);
        Assert.Equal(1, Session(notebook).WholeRuns);
    }

    [Fact]
    public async Task TheRowsTheModelSaidItLearnedNothingAbout_AreCountedInTheReportTheBlockDraws()
    {
        await using var notebook = await NotebookAsync();

        notebook.Scaffold.Variables.Set(StepKernel.HandedBack, Guessed(Run(Blocks), unfamiliar: 3).PredictionsToJson());

        var drawn = await ShownAtTheReportAsync(notebook);

        Assert.Contains("<th>part</th><th>rows</th><th>unfamiliar</th>", drawn, StringComparison.Ordinal);
        Assert.Contains("<td>validation</td><td class=\"deepsharp-number\">133</td><td class=\"deepsharp-number\">3</td>", drawn, StringComparison.Ordinal);
        Assert.Contains("<td>test</td><td class=\"deepsharp-number\">135</td><td class=\"deepsharp-number\">3</td>", drawn, StringComparison.Ordinal);
    }

    [Fact]
    public async Task PredictionsHandedBackBeforeABlockWasEdited_AreRefusedAtTheReport_AsMadeBehindAnotherFit()
    {
        // Handed back behind the blocks as they were; then the fare is scaled another way. The rows, the answer and its numbers
        // are what they were, and the fit is not: the model was handed other numbers than the blocks now hand over.
        await using var notebook = await NotebookAsync();

        notebook.Scaffold.Variables.Set(StepKernel.HandedBack, Guessed(Run(Blocks)).PredictionsToJson());

        Assert.False(string.IsNullOrEmpty(await ShownAtTheReportAsync(notebook)));
        Assert.False(Report(notebook).Outputs[2].IsError);

        notebook.Scaffold.Cells[3].Source = Fare.Replace("midrange", "minmax", StringComparison.Ordinal);

        var refused = await ShownAtTheReportAsync(notebook);

        Assert.True(Report(notebook).Outputs[2].IsError);
        Assert.Contains("deepsharp.predictions", refused, StringComparison.Ordinal);
        Assert.Contains("made behind another fit", refused, StringComparison.Ordinal);
        Assert.Contains("Run the C# cell that trains again", refused, StringComparison.Ordinal);
    }

    [Fact]
    public async Task PredictionsOfAnotherSplit_AreRefusedAtTheBlock_AsMadeBehindAnotherFit()
    {
        // The cell ran the blocks divided by another seed: as many rows in each part, and other rows in them.
        await using var notebook = await NotebookAsync();
        var elsewhere = Run([.. Blocks.Select(block => block == Split ? Split.Replace("20260923", "7", StringComparison.Ordinal) : block)]);

        notebook.Scaffold.Variables.Set(StepKernel.HandedBack, Guessed(elsewhere).PredictionsToJson());

        var refused = await ShownAtTheReportAsync(notebook);

        Assert.True(Report(notebook).Outputs[2].IsError);
        Assert.Contains("deepsharp.predictions", refused, StringComparison.Ordinal);
        Assert.Contains("made behind another fit", refused, StringComparison.Ordinal);
    }

    [Fact]
    public async Task PredictionsOfAnotherAnswer_AreRefusedAtTheBlock_NamingBoth()
    {
        await using var notebook = await NotebookAsync();
        var elsewhere = Run([.. Blocks[..4], """{"step": "target", "column": "pclass"}""", """{"step": "evidence.report", "metrics": ["rmse"], "parts": ["validation", "test"], "shown": ["numbers"]}"""]);

        notebook.Scaffold.Variables.Set(StepKernel.HandedBack, Guessed(elsewhere).PredictionsToJson());

        var refused = await ShownAtTheReportAsync(notebook);

        Assert.True(Report(notebook).Outputs[2].IsError);
        Assert.Contains("answer &#39;pclass&#39;", refused, StringComparison.Ordinal);
        Assert.Contains("answers &#39;survived&#39;", refused, StringComparison.Ordinal);
    }

    [Fact]
    public async Task PredictionsHandedBack_ToBlocksWhoseWholeRunStops_AreRefusedInTheWordsOfWhatStopsIt()
    {
        // The blocks show their rows, and their whole run stops: the answer is clipped onto the training rows' range, so the
        // test rows' answers no longer lead back to what they were — which only a whole run checks. Whatever was handed back
        // cannot be measured on such a run.
        File.WriteAllText(Path.Join(_folder, "amounts.csv"), "t,y\n" + string.Join('\n', new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 200, 300 }.Select((y, at) => $"{at + 1},{y}")) + "\n");

        var steps = Pdd.Create()
            .ReadCsv("amounts.csv")
            .Declare(schema => schema.Integer("t").Number("y"))
            .SplitByTime("t", 0.50, 0.25)
            .Normalise("y", Scale.MinMax, OutOfRange.Clip)
            .Target("y")
            .Report(report => report.Measure(Metric.Rmse).On(Part.Test).As(Shown.Numbers))
            .Build()
            .Declaration.Steps;

        await using var notebook = await NotebookAsync([.. steps.Select(step => step.AsBlockText())]);

        notebook.Scaffold.Variables.Set(StepKernel.HandedBack, Guessed(Run(Blocks)).PredictionsToJson());
        await notebook.GestureAsync(Report(notebook), StepRenderer.Show);

        Assert.Equal(3, Report(notebook).Outputs.Count);
        Assert.Contains("12 rows", Report(notebook).Outputs[1].Content, StringComparison.Ordinal);
        Assert.True(Report(notebook).Outputs[2].IsError);
        Assert.Contains("does not lead back", Report(notebook).Outputs[2].Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AReportDrawnOnce_IsDrawnAgainFromMemory_UntilWhatItIsMeasuredFromChanges()
    {
        await using var notebook = await NotebookAsync();
        var session = Session(notebook);
        var report = Report(notebook);

        notebook.Scaffold.Variables.Set(StepKernel.HandedBack, Guessed(Run(Blocks)).PredictionsToJson());

        var first = await ShownAtTheReportAsync(notebook);

        await ShownAtTheReportAsync(notebook);
        await notebook.GestureAsync(report, StepRenderer.Page, "1");

        Assert.Equal(1, session.ReportsMeasured);
        Assert.Equal(1, session.WholeRuns);
        Assert.Equal(first, report.Outputs[2].Content);

        // Other predictions handed back are measured.
        notebook.Scaffold.Variables.Set(StepKernel.HandedBack, Guessed(Run(Blocks), unfamiliar: 1).PredictionsToJson());
        await ShownAtTheReportAsync(notebook);

        Assert.Equal(2, session.ReportsMeasured);

        // So are the same predictions under other steps — made behind another fit now, and refused.
        report.Source = """{"step": "evidence.report", "metrics": ["accuracy", "confusionmatrix"], "parts": ["validation", "test"], "shown": ["numbers"]}""";

        Assert.Contains("made behind another fit", await ShownAtTheReportAsync(notebook), StringComparison.Ordinal);
        Assert.Equal(3, session.ReportsMeasured);

        // And over other bytes.
        File.AppendAllText(Path.Join(_folder, "titanic.csv"), "1,1,female,30.0,0,0,80.0,S,First,woman,False,B,Southampton,yes,True\n");
        await notebook.GestureAsync(report, StepRenderer.Show);

        Assert.Equal(4, session.ReportsMeasured);
    }

    [Fact]
    public async Task TheToolbarsRunEndingAtTheReport_RunsTheWholePipelineOnce_ForTheFitItHandsOverAndTheMeasuresItDraws()
    {
        await using var notebook = await NotebookAsync();
        var session = Session(notebook);

        notebook.Scaffold.Variables.Set(StepKernel.HandedBack, Guessed(Run(Blocks)).PredictionsToJson());

        await notebook.PressAsync(RunPipelineAction.Id);

        Assert.Equal(1, session.WholeRuns);
        Assert.Equal(1, session.RunsFitted);
        Assert.Equal(1, session.ReportsMeasured);
        Assert.Equal(3, Report(notebook).Outputs.Count);
        Assert.False(Report(notebook).Outputs[2].IsError, Report(notebook).Outputs[2].Content);
        Assert.Equal(Guessed(Run(Blocks)).Report().ToHtml(), Report(notebook).Outputs[2].Content);
    }

    [Fact]
    public void TheKeyPredictionsAreHandedBackUnder_IsNeverAValidCSharpIdentifier()
    {
        // As with the pipeline handed over: Verso makes a C# variable of every value a cell can name, and a cell's own
        // variables are copied into the notebook's under their names; a key no variable can have is neither.
        Assert.Equal("deepsharp.predictions", StepKernel.HandedBack);
        Assert.False(SyntaxFacts.IsValidIdentifier(StepKernel.HandedBack));
    }
}
