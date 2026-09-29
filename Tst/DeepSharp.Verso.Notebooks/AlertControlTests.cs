// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A profile under a block says what should not be there, and an alert whose answer is a change to the columns carries a
/// box that makes it: a column that hands a model the answer is left out, and a value that is how the file writes that
/// nothing is known is said on the schema. Drawn unticked, a box asks for what its alert says; an alert answered by a
/// step to be written names the step and carries no box, since where a step belongs is a person's to say.
/// </summary>
public sealed class AlertControlTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-alerts-").FullName;

    private static readonly string[] Blocks =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}, {"name": "alive", "kind": "boolean", "optional": false}]}""",
        """{"step": "evidence.profile"}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "target", "column": "survived"}""",
    ];

    public AlertControlTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private async Task<Notebook> NotebookAsync()
    {
        var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private static DeclareStep Declared(Notebook notebook) =>
        notebook.Scaffold.Cells.Select(cell => NotebookVerbs.Catalog().ReadStep(cell.Source)).OfType<DeclareStep>().Single();

    // The profile drawn under its block once the block shows its data.
    private static async Task<string> ProfileAsync(Notebook notebook)
    {
        var profile = notebook.Scaffold.Cells[2];

        await notebook.GestureAsync(profile, StepRenderer.Show);

        return profile.Outputs.Single(output => output.Content.Contains("deepsharp-evidence", StringComparison.Ordinal)).Content;
    }

    private static DrawnBox? AnswerBox(string profile, string column) =>
        profile.Boxes().Cast<DrawnBox?>().SingleOrDefault(box => box!.Value.Action.StartsWith($"{StepRenderer.Answer} ", StringComparison.Ordinal)
            && JsonNode.Parse(box.Value.Action[(StepRenderer.Answer.Length + 1)..])!["column"]!.GetValue<string>() == column);

    [Fact]
    public async Task AColumnThatHandsAModelTheAnswer_HasABoxThatLeavesItOut()
    {
        await using var notebook = await NotebookAsync();
        var box = AnswerBox(await ProfileAsync(notebook), "alive")!.Value;

        Assert.True(box is { Ticked: false, Enabled: true, CarriesAPayload: false });

        var ticked = await notebook.GestureAsync(notebook.Scaffold.Cells[2], box.Action, "true");

        Assert.True(ticked.StateChanged);
        Assert.True(Declared(notebook).Columns.Single(column => column.Name == "alive").Excluded);
    }

    [Fact]
    public async Task AValueThatIsHowTheFileWritesNothingIsKnown_HasABoxThatSaysSoOnTheSchema()
    {
        await using var notebook = await NotebookAsync();
        var profile = await ProfileAsync(notebook);
        var box = AnswerBox(profile, "fare")!.Value;

        Assert.Contains("answered by saying in the schema that <code>0</code> stands for a gap <label>", profile, StringComparison.Ordinal);

        await notebook.GestureAsync(notebook.Scaffold.Cells[2], box.Action, "true");

        Assert.Equal("0", Declared(notebook).Columns.Single(column => column.Name == "fare").Missing);
    }

    [Fact]
    public async Task AnAlertAnsweredByAStepToBeWritten_HasNoBox_AndABoxSentUntickedChangesNothing()
    {
        await using var notebook = await NotebookAsync();
        var profile = await ProfileAsync(notebook);
        var before = Declared(notebook);

        Assert.Null(AnswerBox(profile, "age"));
        Assert.Contains("answered by <code>fill.missing</code>", profile, StringComparison.Ordinal);

        var unticked = await notebook.GestureAsync(notebook.Scaffold.Cells[2], AnswerBox(profile, "alive")!.Value.Action, "false");

        Assert.False(unticked.StateChanged);
        Assert.Equal(before, Declared(notebook));
    }

    [Fact]
    public async Task AnAnswerTheRulesRefuse_IsNotMade_AndTheBlockSaysWhy_AndOneOfNoKnownKindChangesNothing()
    {
        await using var notebook = await NotebookAsync();
        var profile = notebook.Scaffold.Cells[2];
        var before = Declared(notebook);

        await ProfileAsync(notebook);

        var unknown = await notebook.GestureAsync(profile, $$"""{{StepRenderer.Answer}} {"column":"fare","answer":"step"}""", "true");
        var nowhere = await notebook.GestureAsync(profile, $$"""{{StepRenderer.Answer}} {"column":"nowhere","answer":"saymissing","value":"0"}""", "true");

        Assert.False(unknown.StateChanged);
        Assert.False(nowhere.StateChanged);
        Assert.Contains(profile.Outputs, output => output.IsError && System.Net.WebUtility.HtmlDecode(output.Content).Contains("'nowhere'", StringComparison.Ordinal)
            && !output.Content.Contains("Parameter", StringComparison.Ordinal));
        Assert.Equal(before, Declared(notebook));
    }

    [Fact]
    public void EveryBoxTakesAColumnInByOneRule_TheSavedFileFirst_ThenWhatItsCellsPropose_ThenAsText()
    {
        var proposal = KindProposal.Of(CsvRowSource.FromText("a,b\n1,x\n1,y\n"));
        var saved = new PipelinePreset(new DeclareStep([new ColumnDeclaration("a", ColumnKind.Timestamp, false) { Format = "yyyyMMdd" }]));

        Assert.Equal(new TakenIn(ColumnKind.Timestamp, "yyyyMMdd"), TakenIn.Of("a", saved, proposal));
        Assert.Equal(ColumnKind.Integer, TakenIn.Of("a", stored: null, proposal).Kind);
        Assert.Equal(new TakenIn(ColumnKind.Text), TakenIn.Of("a", stored: null, proposal: null));
        Assert.Equal(new TakenIn(ColumnKind.Text), TakenIn.Of("elsewhere", stored: null, proposal));
        Assert.Null(TakenIn.Of("a", saved, proposal).Said());
    }

    [Theory]
    [InlineData("category", ColumnKind.Category)]
    [InlineData("Timestamp", ColumnKind.Timestamp)]
    [InlineData("1", null)]
    [InlineData("text, number", null)]
    [InlineData(" text", null)]
    [InlineData(null, null)]
    public void AKindIsReadFromItsWord_AndFromNothingElse(string? word, ColumnKind? kind)
    {
        Assert.Equal(kind, word.AsKind());
    }
}
