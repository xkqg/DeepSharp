// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The list says what the model is asked to predict. A select picks which kind of output its boxes make, and picking
/// commits nothing; each row's output box puts its column into that output or takes it out, through the one builder
/// every door makes an output with; a box of its own takes the output away. A column of an output of many is put in
/// after the nearest column the output holds before it, and taken out where it stands. An output gesture needs blocks
/// that make one pipeline, and a kind of output no row can take is drawn disabled with the rule that stops it.
/// </summary>
public sealed class OutputListTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-output-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "sibsp", "kind": "integer", "optional": false}, {"name": "parch", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public OutputListTests()
    {
        File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));
        File.Copy(Repository.Data("apple.csv"), Path.Join(_folder, "apple.csv"));
    }

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private async Task<Notebook> NotebookAsync(params string[] blocks)
    {
        var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private static IReadOnlyList<IPipelineStep> Steps(Notebook notebook) =>
        [.. notebook.Scaffold.Cells.Where(cell => cell.Type == StepCellType.StepType).Select(cell => NotebookVerbs.Catalog().ReadStep(cell.Source))];

    private static INamesTheAnswer? Output(Notebook notebook) => Steps(notebook).OfType<INamesTheAnswer>().SingleOrDefault();

    private static CellModel Schema(Notebook notebook) => notebook.Scaffold.Cells[1];

    private static string List(Notebook notebook) =>
        Schema(notebook).Outputs.Single(output => output.Content.Contains("<tr data-column=", StringComparison.Ordinal)).Content;

    private static async Task ChooseAsync(Notebook notebook) => await notebook.GestureAsync(Schema(notebook), StepRenderer.Columns);

    private static Task<CellInteractionContext> PickAsync(Notebook notebook, string verb) =>
        notebook.GestureAsync(Schema(notebook), List(notebook).TypeSelect().Action, verb);

    private static Task<CellInteractionContext> OutputAsync(Notebook notebook, string column, bool ticked) =>
        notebook.GestureAsync(Schema(notebook), List(notebook).Row(column).Output.Action, ticked ? "true" : "false");

    private static bool SaysNotMade(Notebook notebook, string words) =>
        Schema(notebook).Outputs.Any(output => output.IsError && WebUtility.HtmlDecode(output.Content).Contains(words, StringComparison.Ordinal));

    [Fact]
    public async Task TheTypeSelect_OffersEveryKindOfOutput_AndPickingOneDrawsTheListWithIt_CommittingNothing()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await ChooseAsync(notebook);

        Assert.Equal(["target", "target.ahead", "target.distribution", "target.labels"], List(notebook).TypeSelect().Options);
        Assert.Equal("target", List(notebook).TypeSelect().Value);

        var picked = await PickAsync(notebook, "target.distribution");

        Assert.False(picked.StateChanged);
        Assert.Equal("target.distribution", List(notebook).TypeSelect().Value);
        Assert.All(List(notebook).Rows(), row => Assert.Contains("\"type\":\"target.distribution\"", row.Output.Action, StringComparison.Ordinal));
        Assert.Null(Output(notebook));
    }

    [Fact]
    public async Task ARowsOutputBox_MakesItsColumnTheAnswer_AndAnotherRowsMovesIt_InPlace()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await ChooseAsync(notebook);
        var made = await OutputAsync(notebook, "survived", ticked: true);

        Assert.True(made.StateChanged);
        Assert.Equal(new TargetStep("survived"), Steps(notebook)[^1]);
        Assert.True(List(notebook).Row("survived").Output.Ticked);
        Assert.Equal("answer", List(notebook).Row("survived").Role);

        await OutputAsync(notebook, "pclass", ticked: true);

        Assert.Equal(new TargetStep("pclass"), Steps(notebook)[^1]);
        Assert.Equal(6, Steps(notebook).Count);
    }

    [Fact]
    public async Task UntickingTheAnswerOfAnOutputOfOne_IsRefusedInTheVerbsOwnWords()
    {
        await using var notebook = await NotebookAsync([.. Titanic, """{"step": "target", "column": "survived"}"""]);

        await ChooseAsync(notebook);
        var refused = await OutputAsync(notebook, "survived", ticked: false);

        Assert.False(refused.StateChanged);
        Assert.Equal(new TargetStep("survived"), Output(notebook));
        Assert.True(SaysNotMade(notebook, "This change is not made:"));
    }

    [Fact]
    public async Task TheRemovalBox_TakesTheOutputAway_AndItsEchoOrItsUntickChangesNothing()
    {
        await using var notebook = await NotebookAsync([.. Titanic, """{"step": "target", "column": "survived"}"""]);

        await ChooseAsync(notebook);
        var removal = List(notebook).RemovalBox();

        Assert.True(removal is { Ticked: false, Enabled: true, CarriesAPayload: false });

        var removed = await notebook.GestureAsync(Schema(notebook), removal.Action, "true");
        var echo = await notebook.GestureAsync(Schema(notebook), removal.Action, "true");
        var untick = await notebook.GestureAsync(Schema(notebook), List(notebook).RemovalBox().Action, "false");

        Assert.True(removed.StateChanged);
        Assert.False(echo.StateChanged);
        Assert.False(untick.StateChanged);
        Assert.Null(Output(notebook));
        Assert.False(List(notebook).RemovalBox().Enabled);
    }

    [Fact]
    public async Task OutputGestures_WhileTheBlocksMakeNoPipeline_AreRefused_SayingWhichBlockStopsThem()
    {
        await using var notebook = await NotebookAsync([.. Titanic, """{"step": "target", "column": "survived"}"""]);

        await ChooseAsync(notebook);
        var tick = List(notebook).Row("pclass").Output.Action;
        var removal = List(notebook).RemovalBox().Action;

        notebook.AddBlock("""{"step": "normalise", "column": "colour", "scale": "standard", "outOfRange": "pass"}""");

        var ticked = await notebook.GestureAsync(Schema(notebook), tick, "true");
        var removed = await notebook.GestureAsync(Schema(notebook), removal, "true");

        Assert.False(ticked.StateChanged);
        Assert.False(removed.StateChanged);
        Assert.Equal(new TargetStep("survived"), Output(notebook));
        Assert.True(SaysNotMade(notebook, "the blocks do not make a pipeline yet: block 7"));
    }

    [Fact]
    public async Task AnOutputBoxFromAListDrawnBeforeTheBlocksChanged_DrawsTheListAgain_AndChangesNothing()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await ChooseAsync(notebook);
        var old = List(notebook).Row("survived").Output.Action;

        await notebook.TickAsync(Schema(notebook), StepRenderer.Category, "pclass", ticked: true);
        var changed = Steps(notebook);

        var stale = await notebook.GestureAsync(Schema(notebook), old, "true");

        Assert.False(stale.StateChanged);
        Assert.Equal(changed, Steps(notebook));
        Assert.Equal("category", List(notebook).Row("pclass").Kind.Value);
    }

    [Fact]
    public async Task ABoxOfAnotherKindOfOutputThanTheOneStanding_IsRefusedWithWords()
    {
        await using var notebook = await NotebookAsync([.. Titanic, """{"step": "target", "column": "survived"}"""]);

        await ChooseAsync(notebook);
        await PickAsync(notebook, "target.distribution");
        var refused = await OutputAsync(notebook, "pclass", ticked: true);

        Assert.False(refused.StateChanged);
        Assert.Equal(new TargetStep("survived"), Output(notebook));
        Assert.True(SaysNotMade(notebook, "the output is 'target'"));
    }

    [Fact]
    public async Task AColumnOfAnOutputOfMany_GoesInAfterItsNearestHeldColumn_AndOutWhereItStands_AndBelowTwoIsRefused()
    {
        await using var notebook = await NotebookAsync([.. Titanic, """{"step": "target.distribution", "columns": ["pclass", "parch"], "scaleBy": "fare"}"""]);

        await ChooseAsync(notebook);
        await OutputAsync(notebook, "sibsp", ticked: true);

        Assert.Equal(["pclass", "sibsp", "parch"], Output(notebook)!.Answers);
        Assert.Equal("fare", ((DistributionStep)Output(notebook)!).ScaleBy);

        await OutputAsync(notebook, "survived", ticked: true);

        Assert.Equal(["survived", "pclass", "sibsp", "parch"], Output(notebook)!.Answers);

        await OutputAsync(notebook, "sibsp", ticked: false);
        await OutputAsync(notebook, "survived", ticked: false);

        Assert.Equal(["pclass", "parch"], Output(notebook)!.Answers);

        var belowTwo = await OutputAsync(notebook, "pclass", ticked: false);

        Assert.False(belowTwo.StateChanged);
        Assert.Equal(["pclass", "parch"], Output(notebook)!.Answers);
        Assert.True(SaysNotMade(notebook, "This change is not made:"));
    }

    [Fact]
    public async Task ATickOnAColumnTheSchemaDoesNotTake_TakesItInFirst_InTheSameChange()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await ChooseAsync(notebook);
        var made = await OutputAsync(notebook, "sex", ticked: true);

        Assert.True(made.StateChanged);
        Assert.Contains(((DeclareStep)Steps(notebook)[1]).Taking, column => column.Name == "sex");
        Assert.Equal(new TargetStep("sex"), Output(notebook));
    }

    [Fact]
    public async Task ARowAnAnswerIsMadeFrom_IsDrawnAsTheAnswer_WithItsBoxDisabled_SinceTakingItOutChangesNothing()
    {
        await using var notebook = await NotebookAsync(
            [.. Titanic, """{"step": "maths", "column": "age", "maths": "log1p", "into": "age_log"}""", """{"step": "target", "column": "age_log"}"""]);

        await ChooseAsync(notebook);
        var age = List(notebook).Row("age");

        Assert.Equal("answer", age.Role);
        Assert.True(age.Output is { Ticked: true, Enabled: false });

        var untick = await OutputAsync(notebook, "age", ticked: false);

        Assert.False(untick.StateChanged);
        Assert.Equal(new TargetStep("age_log"), Output(notebook));
    }

    [Fact]
    public async Task AnOutputBoxSentWithAVerbThatMakesNoOutput_IsRefusedWithWords_AndChangesNothing()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await ChooseAsync(notebook);
        var action = List(notebook).Row("survived").Output.Action;

        foreach (var verb in new[] { "normalise", "target.nothing" })
        {
            var sent = await notebook.GestureAsync(
                Schema(notebook), action.Replace("\"type\":\"target\"", $"\"type\":\"{verb}\"", StringComparison.Ordinal), "true");

            Assert.False(sent.StateChanged);
            Assert.True(SaysNotMade(notebook, $"'{verb}' is not a kind of output the list can make."));
        }

        Assert.Null(Output(notebook));
    }

    [Fact]
    public async Task AColumnAStepBelowTakesAway_CannotBeMadeTheAnswer()
    {
        await using var notebook = await NotebookAsync(
            """{"step": "read.csv", "path": "titanic.csv"}""",
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "sex", "kind": "category", "optional": false}, {"name": "fare", "kind": "number", "optional": false}]}""",
            """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
            """{"step": "encode", "column": "sex", "as": "onehot", "unseen": "reserve"}""");

        await ChooseAsync(notebook);

        Assert.False(List(notebook).Row("sex").Output.Enabled);
        Assert.True(List(notebook).Row("fare").Output.Enabled);
    }

    [Fact]
    public async Task AnAnswerReadRowsAhead_IsDisabledWithoutASplitInTime_InTheWordsOfTheRuleThatStopsIt()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var declaration = new PipelineDeclaration(Steps(notebook));
        var stopped = PipelineDeclaration.FaultsIn(declaration.WithOutput(
            (INamesTheAnswer)NotebookVerbs.Catalog().Make("target.ahead", new() { ["column"] = "survived" }, carrying: null)))[0].Message;

        await ChooseAsync(notebook);

        Assert.Contains(stopped, List(notebook).TypeLabels()["target.ahead"], StringComparison.Ordinal);
        Assert.Contains("value=\"target.ahead\" disabled", List(notebook), StringComparison.Ordinal);
    }

    [Fact]
    public async Task OnAPriceSeriesSplitInTime_AnAnswerReadRowsAheadIsOffered_AndATickPlacesIt()
    {
        await using var notebook = await NotebookAsync(
            """{"step": "read.csv", "path": "apple.csv"}""",
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "Date", "kind": "timestamp", "optional": false}, {"name": "AAPL.Close", "kind": "number", "optional": false}]}""",
            """{"step": "order.by", "columns": ["Date"]}""",
            """{"step": "split.byTime", "column": "Date", "train": 0.7, "validation": 0.15, "test": 0.15, "gap": 5}""");

        await ChooseAsync(notebook);
        await PickAsync(notebook, "target.ahead");
        var made = await OutputAsync(notebook, "AAPL.Close", ticked: true);

        Assert.True(made.StateChanged);
        Assert.IsType<AheadStep>(Output(notebook));
        Assert.Equal("answer", List(notebook).Row("AAPL.Close").Role);
    }
}
