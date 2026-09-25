// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Net;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The list sets the output's own values beside its answer: how many rows ahead, whether a return, how many ones a row
/// holds, what the shares are shares of. Each is a select offering every value the rules keep, and nothing is typed. A
/// select commits the value it ends on, as one pick from the state the list was drawn in — the same rule as a row's kind
/// select — so a keyboard walk ends where it stops, a return taken back leaves the output where it stood, and a send
/// made stale by another change draws the list again and changes nothing. A value left out is not said, and not written.
/// </summary>
public sealed class OutputParameterTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-parameters-").FullName;

    private static readonly string[] Labels = [.. Enumerable.Range(1, 12).Select(label => $"l{label}")];

    public OutputParameterTests()
    {
        File.Copy(Repository.Data("apple.csv"), Path.Join(_folder, "apple.csv"));
        File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));
        File.WriteAllText(
            Path.Join(_folder, "labels.csv"),
            $"id,{string.Join(',', Labels)}\n1,1,0,0,0,0,0,0,0,0,0,0,0\n2,0,1,0,0,0,0,0,0,0,0,0,0\n3,0,0,1,0,0,0,0,0,0,0,0,0\n4,0,0,0,1,0,0,0,0,0,0,0,0\n");
        File.WriteAllText(Path.Join(_folder, "flock.csv"), "farm,chicks,w500,w550,w600\n1,10,0.2,0.5,0.3\n2,20,0.1,0.6,0.3\n3,15,0.3,0.3,0.4\n4,12,0.25,0.5,0.25\n");
    }

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private async Task<Notebook> NotebookAsync(string name, IEnumerable<string> blocks)
    {
        var notebook = await Notebook.OpenAsync(Path.Join(_folder, $"{name}.verso"));

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    // A price series split in time, the answer read five rows ahead; the price scaled above the answer when asked.
    private Task<Notebook> SeriesAsync(int gap, bool scaled = false) => NotebookAsync(
        "series",
        [
            """{"step": "read.csv", "path": "apple.csv"}""",
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "Date", "kind": "timestamp", "optional": false}, {"name": "AAPL.Close", "kind": "number", "optional": false}]}""",
            """{"step": "order.by", "columns": ["Date"]}""",
            $$"""{"step": "split.byTime", "column": "Date", "train": 0.7, "validation": 0.15, "test": 0.15, "gap": {{gap}}}""",
            .. scaled ? ["""{"step": "normalise", "column": "AAPL.Close", "scale": "standard", "outOfRange": "pass"}"""] : Array.Empty<string>(),
            """{"step": "target.ahead", "column": "AAPL.Close", "ahead": 5, "as": "value"}""",
        ]);

    private Task<Notebook> LabelledAsync() => NotebookAsync(
        "labels",
        Pdd.Create()
            .ReadCsv("labels.csv")
            .Declare(schema => schema.Integer("id").Integer(Labels))
            .SplitAtRandom(0.50, seed: 3)
            .Labels(Labels)
            .Declaration.Steps.Select(step => step.AsBlockText()));

    private Task<Notebook> FlockAsync() => NotebookAsync(
        "flock",
        Pdd.Create()
            .ReadCsv("flock.csv")
            .Declare(schema => schema.Integer("farm", "chicks").Number("w500", "w550", "w600"))
            .SplitAtRandom(0.50, seed: 3)
            .Distribution(["w500", "w550", "w600"], scaleBy: "chicks")
            .Declaration.Steps.Select(step => step.AsBlockText()));

    private static IReadOnlyList<IPipelineStep> Steps(Notebook notebook) =>
        [.. notebook.Scaffold.Cells.Where(cell => cell.Type == StepCellType.StepType).Select(cell => NotebookVerbs.Catalog().ReadStep(cell.Source))];

    private static INamesTheAnswer Output(Notebook notebook) => Steps(notebook).OfType<INamesTheAnswer>().Single();

    private static AheadStep Ahead(Notebook notebook) => (AheadStep)Output(notebook);

    private static CellModel SchemaBlock(Notebook notebook) => notebook.Scaffold.Cells[1];

    private static string List(Notebook notebook) =>
        SchemaBlock(notebook).Outputs.Single(output => output.Content.Contains("<tr data-column=", StringComparison.Ordinal)).Content;

    private static bool SaysNotMade(Notebook notebook, string words) =>
        SchemaBlock(notebook).Outputs.Any(output => output.IsError && WebUtility.HtmlDecode(output.Content).Contains(words, StringComparison.Ordinal));

    private static async Task<DrawnList> ChooseAsync(Notebook notebook)
    {
        await notebook.GestureAsync(SchemaBlock(notebook), StepRenderer.Columns);

        return new DrawnList(SchemaBlock(notebook), List(notebook));
    }

    // A walk as the router sends it: every value, in order, to the block the list was drawn on; how many committed.
    private static async Task<int> WalkAsync(Notebook notebook, CellModel drawnOn, string action, params string[] values)
    {
        var commits = 0;

        foreach (var value in values)
        {
            commits += (await notebook.GestureAsync(drawnOn, action, value)).StateChanged ? 1 : 0;
        }

        return commits;
    }

    private static string Number(int value) => value.ToString(CultureInfo.InvariantCulture);

    [Fact]
    public async Task EachOfTheOutputsOwnValues_OffersEveryValueTheRulesKeep_AndNothingElse()
    {
        await using var series = await SeriesAsync(gap: 5);
        await using var labelled = await LabelledAsync();
        await using var flock = await FlockAsync();

        var ahead = (await ChooseAsync(series)).Html.ParameterSelects();
        var ones = (await ChooseAsync(labelled)).Html.ParameterSelects()["ones"];
        var scaleBy = (await ChooseAsync(flock)).Html.ParameterSelects()["scaleBy"];

        // Rows ahead only as far as the gap keeps the parts apart; its answer's column is the row's own box.
        Assert.Equal(["ahead", "as"], ahead.Keys.Order(StringComparer.Ordinal));
        Assert.Equal(["1", "2", "3", "4", "5"], ahead["ahead"].Options);
        Assert.Equal("5", ahead["ahead"].Value);
        Assert.Equal(["value", "return"], ahead["as"].Options);
        Assert.Equal("value", ahead["as"].Value);

        // Not said, or as many ones as there are labels; a number of the flock, never one of its own shares.
        Assert.Equal(["", .. Labels.Select((_, at) => Number(at + 1))], ones.Options);
        Assert.Equal("", ones.Value);
        Assert.Equal(["", "farm", "chicks"], scaleBy.Options);
        Assert.Equal("chicks", scaleBy.Value);
    }

    [Theory]
    [InlineData(5, new[] { "5", "1", "1", "1" }, 1, 1)]
    [InlineData(20, new[] { "5", "10", "10", "12", "12" }, 12, 2)]
    [InlineData(20, new[] { "5", "4", "4", "3" }, 3, 2)]
    public async Task AKeyboardWalkOnHowFarAhead_EndsOnTheValueItStopsAt(int gap, string[] sent, int ends, int commits)
    {
        await using var notebook = await SeriesAsync(gap);
        var list = await ChooseAsync(notebook);

        Assert.Equal(commits, await WalkAsync(notebook, list.On, list.Html.ParameterSelects()["ahead"].Action, sent));
        Assert.Equal(ends, Ahead(notebook).Ahead);
        Assert.Equal(Number(ends), List(notebook).ParameterSelects()["ahead"].Value);
    }

    [Fact]
    public async Task AKeyboardWalkOnHowManyOnes_EndsOnTheValueItStopsAt()
    {
        await using var notebook = await LabelledAsync();
        var list = await ChooseAsync(notebook);

        Assert.Equal(2, await WalkAsync(notebook, list.On, list.Html.ParameterSelects()["ones"].Action, "", "1", "1", "12", "12"));
        Assert.Equal(12, ((LabelsStep)Output(notebook)).Ones);
    }

    [Fact]
    public async Task AnEchoOrATab_ChangesNothing()
    {
        await using var notebook = await SeriesAsync(gap: 20);
        var list = await ChooseAsync(notebook);
        var output = notebook.Scaffold.Cells[^1];

        Assert.Equal(0, await WalkAsync(notebook, list.On, list.Html.ParameterSelects()["ahead"].Action, "5", "5"));
        Assert.Same(output, notebook.Scaffold.Cells[^1]);
    }

    [Fact]
    public async Task AReturnPickedAndTakenBack_LeavesTheOutputWhereItWasDrawn()
    {
        await using var notebook = await SeriesAsync(gap: 5, scaled: true);
        var list = await ChooseAsync(notebook);
        var drawn = Steps(notebook);
        var select = list.Html.ParameterSelects()["as"].Action;

        Assert.Equal(1, await WalkAsync(notebook, list.On, select, "value", "return"));

        // A return is made from the price as it was read: directly after the split, above the scaling.
        Assert.Equal(new AheadStep("AAPL.Close", 5, AheadAs.Return), Steps(notebook)[4]);

        Assert.Equal(1, await WalkAsync(notebook, list.On, select, "return", "value"));
        Assert.Equal(drawn, Steps(notebook));
    }

    [Fact]
    public async Task AValueSetToNotSaid_IsLeftOut_AndNotWritten()
    {
        await using var notebook = await FlockAsync();
        var list = await ChooseAsync(notebook);
        var select = list.Html.ParameterSelects()["scaleBy"].Action;

        Assert.Equal(1, await WalkAsync(notebook, list.On, select, "chicks", ""));
        Assert.Null(((DistributionStep)Output(notebook)).ScaleBy);
        Assert.DoesNotContain("scaleBy", notebook.Scaffold.Cells[^1].Source, StringComparison.Ordinal);
        Assert.Equal("", List(notebook).ParameterSelects()["scaleBy"].Value);
    }

    [Fact]
    public async Task ASendAnotherChangeMadeStale_DrawsTheListAgain_AndCommitsNothing()
    {
        await using var notebook = await SeriesAsync(gap: 20);
        var list = await ChooseAsync(notebook);
        var select = list.Html.ParameterSelects()["ahead"].Action;

        await notebook.GestureAsync(SchemaBlock(notebook), List(notebook).Row("AAPL.Open").Included.Action, "true");

        Assert.Equal(0, await WalkAsync(notebook, SchemaBlock(notebook), select, "5", "10", "10", "12"));
        Assert.Equal(5, Ahead(notebook).Ahead);
        Assert.Equal("5", List(notebook).ParameterSelects()["ahead"].Value);
    }

    [Fact]
    public async Task ARedrawBetweenKeysEndsTheWalk_AndTheSelectItDrawsStartsFromItsOwnValue()
    {
        await using var notebook = await SeriesAsync(gap: 20);
        var list = await ChooseAsync(notebook);

        Assert.Equal(1, await WalkAsync(notebook, list.On, list.Html.ParameterSelects()["ahead"].Action, "5", "10"));

        var redrawn = List(notebook).ParameterSelects()["ahead"];

        Assert.Equal("10", redrawn.Value);
        Assert.Equal(1, await WalkAsync(notebook, SchemaBlock(notebook), redrawn.Action, "10", "20"));
        Assert.Equal(20, Ahead(notebook).Ahead);
    }

    [Fact]
    public async Task ASendWhileTheBlocksMakeNoPipeline_IsRefused_SayingWhichBlockStopsThem()
    {
        await using var notebook = await SeriesAsync(gap: 20);
        var list = await ChooseAsync(notebook);
        var select = list.Html.ParameterSelects()["ahead"].Action;

        notebook.AddBlock("""{"step": "normalise", "column": "colour", "scale": "standard", "outOfRange": "pass"}""");

        Assert.False((await notebook.GestureAsync(SchemaBlock(notebook), select, "10")).StateChanged);
        Assert.Equal(5, Ahead(notebook).Ahead);
        Assert.True(SaysNotMade(notebook, "the blocks do not make a pipeline yet: block 6"));
    }

    [Fact]
    public async Task AValueTheOutputCannotHold_IsRefusedInTheFormsWords()
    {
        await using var notebook = await SeriesAsync(gap: 20);
        var list = await ChooseAsync(notebook);

        Assert.False((await notebook.GestureAsync(list.On, list.Html.ParameterSelects()["ahead"].Action, "ten")).StateChanged);
        Assert.Equal(5, Ahead(notebook).Ahead);
        Assert.True(SaysNotMade(notebook, "'ten' is not a number."));
    }

    [Fact]
    public async Task ASendNamingNothingTheOutputHoldsBesideItsAnswer_ChangesNothing()
    {
        await using var notebook = await SeriesAsync(gap: 20);
        var list = await ChooseAsync(notebook);
        var select = list.Html.ParameterSelects()["ahead"].Action;
        var drawn = Steps(notebook);

        // Its verb, its answer's column, and no key at all: none is a value this select sets.
        Assert.Equal(0, await WalkAsync(notebook, list.On, select.Replace("\"key\":\"ahead\"", "\"key\":\"step\"", StringComparison.Ordinal), "target"));
        Assert.Equal(0, await WalkAsync(notebook, list.On, select.Replace("\"key\":\"ahead\"", "\"key\":\"column\"", StringComparison.Ordinal), "AAPL.Open"));
        Assert.Equal(0, await WalkAsync(notebook, list.On, select.Replace("\"key\":\"ahead\",", string.Empty, StringComparison.Ordinal), "3"));
        Assert.Equal(drawn, Steps(notebook));
        Assert.DoesNotContain(SchemaBlock(notebook).Outputs, output => output.IsError);
    }

    [Fact]
    public async Task ASendWhileNoOutputStands_ChangesNothing()
    {
        await using var notebook = await NotebookAsync(
            "titanic",
            [
                """{"step": "read.csv", "path": "titanic.csv"}""",
                """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "fare", "kind": "number", "optional": false}]}""",
                """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
            ]);
        var list = await ChooseAsync(notebook);
        var drawnFrom = JsonNode.Parse(list.Html.TypeSelect().Action[(list.Html.TypeSelect().Action.IndexOf(' ', StringComparison.Ordinal) + 1)..])!;
        var select = ControlAction.Of(StepRenderer.ListParameter, new JsonObject
        {
            [StepRenderer.ParameterKey] = "ahead",
            [StepRenderer.TypeKey] = "target.ahead",
            [StepRenderer.DrawnKey] = drawnFrom[StepRenderer.DrawnKey]!.GetValue<string>(),
            [StepRenderer.SourceKey] = drawnFrom[StepRenderer.SourceKey]!.GetValue<string>(),
        });

        Assert.Equal(0, await WalkAsync(notebook, list.On, select, "3"));
        Assert.Empty(Steps(notebook).OfType<INamesTheAnswer>());
        Assert.Empty(list.Html.ParameterSelects());
    }

    [Fact]
    public async Task TheOutputsValues_AreDrawnOnlyWhileTheListMakesTheKindOfOutputThatStands()
    {
        await using var notebook = await SeriesAsync(gap: 20);
        var list = await ChooseAsync(notebook);

        await notebook.GestureAsync(SchemaBlock(notebook), list.Html.TypeSelect().Action, "target");

        Assert.Empty(List(notebook).ParameterSelects());
    }

    [Fact]
    public void AValueTheListCannotOffer_IsSaidToBeSetInTheOutputBlocksForm()
    {
        var catalog = NotebookVerbs.Catalog();

        catalog.Register<NotedStep>();

        var source = new SourceRows(CsvRowSource.FromText("survived,fare\n1,7.25\n0,8.05\n"), "fingerprint");
        var declaration = new PipelineDeclaration(
        [
            new ReadCsvStep("rows.csv"),
            new DeclareStep([new ColumnDeclaration("survived", ColumnKind.Integer, Optional: false), new ColumnDeclaration("fare", ColumnKind.Number, Optional: false)]),
            new SplitAtRandomStep(new SplitShares(0.50, 0, 0.50), 3),
            new NotedStep("survived", "kept apart", 1),
        ]);

        var list = ColumnList.Of(catalog, declaration, source, [], stored: null, "key", ListPicks.None, whole: true).Content;

        Assert.Equal(["note: kept apart — set in the output block's form", "window: 1 — set in the output block's form"], list.ParametersSetElsewhere());
        Assert.Empty(list.ParameterSelects());
    }

    // A kind of output another package brings: its answer, a note in words, and a whole number no rule bounds.
    private sealed record NotedStep : IPipelineStep<NotedStep>, INamesTheAnswer, IDescribesColumns
    {
        private static readonly ColumnParameter ColumnKey = new("column", "The answer.", "answer", ColumnKinds.Any);

        private static readonly TextParameter NoteKey = new("note", "Words about the answer.", "none");

        private static readonly WholeNumberParameter WindowKey = new("window", "How many rows it looks at.", 1);

        public NotedStep(string column, string note, int window)
        {
            Column = ColumnKey.Require(column);
            Note = NoteKey.Require(note);
            Window = WindowKey.Require(window);
        }

        public string Column { get; }

        public string Note { get; }

        public int Window { get; }

        public IReadOnlyList<string> Answers => [Column];

        public static string Name => "target.noted";

        public static string Purpose => "Names the answer, with a note and a window.";

        public static StepParameters<NotedStep> Parameters { get; } = new StepParameters<NotedStep>()
            .With(ColumnKey, step => step.Column)
            .With(NoteKey, step => step.Note)
            .With(WindowKey, step => step.Window);

        public string Verb => Name;

        public ColumnState After(ColumnState before) => before;

        public static NotedStep ReadFrom(JsonElement element) => new(ColumnKey.Read(element), NoteKey.Read(element), WindowKey.Read(element));
    }
}
