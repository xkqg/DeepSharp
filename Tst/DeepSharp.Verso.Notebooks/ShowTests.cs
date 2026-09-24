// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Verso.Notebooks;
using DeepSharp.Pipelines;
using Microsoft.CodeAnalysis.CSharp;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// "Show the data here" runs the pipeline the blocks declare up to that block and shows the rows there, each
/// placed where the split declared anywhere in the notebook puts it — so a range drawn above the split is still
/// drawn over the training rows alone. The run is started by the gesture and written by the block's own kernel
/// while it runs, which is how it reaches the screen. A relative path is read from the notebook's folder, and
/// the declaration the blocks make is handed to C# cells as text, under a key no C# variable can have.
/// </summary>
public sealed class ShowTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-notebook-").FullName;

    // The Titanic pipeline as blocks, its data read from beside the notebook.
    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "data/titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private async Task<Notebook> NotebookAsync(params string[] blocks)
    {
        Directory.CreateDirectory(Path.Join(_folder, "data"));
        File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "data", "titanic.csv"), overwrite: true);

        var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private PipelineDeclaration Declared() =>
        new([.. Titanic.Select(block => StepCatalog.BuiltIn().ReadStep(block))]);

    private static NotebookSession Session(Notebook notebook) =>
        notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;

    [Fact]
    public async Task ShowingAViewAgain_OrAnotherPageOfIt_RunsNoStep_UntilWhatItIsWorkedOutFromChanges()
    {
        // A view is worked out from the file's bytes and the steps down to its block — on down to the split, when
        // the split stands below, since the split places the rows above it. While those stay, it is not run again.
        await using var notebook = await NotebookAsync(Titanic);
        var declare = notebook.Scaffold.Cells[1];
        var session = Session(notebook);

        await notebook.GestureAsync(declare, StepRenderer.Show);
        await notebook.GestureAsync(declare, StepRenderer.Show);
        await notebook.GestureAsync(declare, StepRenderer.Page, "1");

        Assert.Equal(1, session.ViewsRun);
        Assert.Contains("rows 51\u2013100 of 891", declare.Outputs[1].Content, StringComparison.Ordinal);

        // A step below the split is not one the schema's rows are worked out from.
        notebook.Scaffold.Cells[3].Source = """{"step": "fill.missing", "column": "age", "with": "mean"}""";
        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.Equal(1, session.ViewsRun);

        // The split below the schema places its rows.
        notebook.Scaffold.Cells[2].Source = """{"step": "split.stratified", "column": "survived", "train": 0.6, "validation": 0.2, "test": 0.2, "seed": 20260923}""";
        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.Equal(2, session.ViewsRun);

        // A file changed on disk is other bytes.
        File.AppendAllText(Path.Join(_folder, "data", "titanic.csv"), "1,1,female,30.0,0,0,80.0,S,First,woman,False,B,Southampton,yes,True\n");
        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.Equal(3, session.ViewsRun);
        Assert.Contains("892 rows", declare.Outputs[1].Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ABlockRunByHand_KeepsTheHandOverWhileItHoldsTheStepTheLastGestureRead_AndWithdrawsItOtherwise()
    {
        // A C# cell never sees a pipeline the blocks no longer make, as far as a part is told: a block run by hand
        // with other text, a block no gesture has read, and a block that is not a step all take the hand-over back
        // until a gesture reads the notebook again. An edit never run, or a block deleted or moved, tells no part.
        await using var notebook = await NotebookAsync(Titanic);
        var fill = notebook.Scaffold.Cells[3];
        bool HandedOver() => notebook.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out _);

        await notebook.GestureAsync(fill, StepRenderer.Show);
        await notebook.RunAsync(fill);

        Assert.True(HandedOver());

        fill.Source = """{"step": "fill.missing", "column": "age", "with": "mean"}""";
        await notebook.RunAsync(fill);

        Assert.False(HandedOver());

        await notebook.GestureAsync(fill, StepRenderer.Show);
        await notebook.RunAsync(notebook.AddBlock(Titanic[4]));

        Assert.False(HandedOver());

        await notebook.GestureAsync(fill, StepRenderer.Show);
        fill.Source = """{"step": "fill.missing", "column": """;
        await notebook.RunAsync(fill);

        Assert.False(HandedOver());
    }

    [Fact]
    public async Task TheNotebooksFolder_IsHandedToCSharpCells_ForTheRelativePathInThePipeline()
    {
        // The handed-over pipeline names its file as the blocks do; a C# cell reads that path from the folder
        // handed beside it, the one the notebook reads it from, wherever its own process happens to stand.
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        Assert.True(notebook.Scaffold.Variables.TryGet<string>(StepKernel.Folder, out var folder));
        Assert.Equal(Path.GetFullPath(_folder), folder);

        var handed = PipelineDeclaration.FromJson(notebook.Scaffold.Variables.Get<string>(StepKernel.HandOver)!, StepCatalog.BuiltIn());

        Assert.Equal(891, new Pipeline(handed, rows: null, SourceFolder.Of(folder!)).ViewAt(2).Table.RowCount);

        // A notebook never saved has no folder: a relative path is read from the working directory, and none is handed.
        await using var unsaved = await Notebook.OpenAsync();
        var read = unsaved.AddBlock($$"""{"step": "read.csv", "path": {{System.Text.Json.JsonSerializer.Serialize(Path.Join(_folder, "data", "titanic.csv"))}}}""");

        await unsaved.GestureAsync(read, StepRenderer.Show);

        Assert.False(unsaved.Scaffold.Variables.TryGet<string>(StepKernel.Folder, out _));
    }

    [Fact]
    public async Task ABlockTypeSwitchedOff_LeavesOneSessionForEveryPart()
    {
        // Verso leaves a switched-off part out of what it lists as enabled. The blocks are then run by the kernel
        // Verso loaded on its own, and the gesture, that kernel and the block type still meet in one session.
        await using var notebook = await NotebookAsync(Titanic);
        var declare = notebook.Scaffold.Cells[1];

        await notebook.Host.DisableExtensionAsync(StepCellType.Id);
        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.True(declare.Outputs[^1].Content.Heads("fare"));

        await notebook.Host.EnableExtensionAsync(StepCellType.Id);

        Assert.Contains(declare.Id, Session(notebook).Shown.Keys);
    }

    [Fact]
    public async Task AKeyNamingAColumn_IsOfferedTheColumnsTheLastGestureSaw_OfAKindItTakes()
    {
        // The kernel is never told which block is being written, so it offers every column the pipeline knew at the
        // last gesture, at whichever block, of a kind the key takes. Before any gesture it knows none.
        await using var notebook = await NotebookAsync(Titanic);
        var kernel = notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Kernel;
        const string fill = """{"step": "fill.missing", "column": " """;
        const string before = """{"step": "fill.missing", "column": """;
        const string drop = """{"step": "drop.columns", "columns": [" """;
        const string moments = """{"step": "feature.timeParts", "column": " """;

        Assert.Empty(await kernel.GetCompletionsAsync(fill, fill.Length - 1));

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        string[] numbers = ["survived", "pclass", "age", "fare", "age_was_missing"];

        Assert.Equal(numbers, (await kernel.GetCompletionsAsync(fill, fill.Length - 1)).Select(completion => completion.InsertText));
        Assert.Equal(numbers.Select(name => $"\"{name}\""), (await kernel.GetCompletionsAsync(before, before.Length)).Select(completion => completion.InsertText));
        Assert.Equal(numbers, (await kernel.GetCompletionsAsync(drop, drop.Length - 1)).Select(completion => completion.InsertText));
        Assert.Empty(await kernel.GetCompletionsAsync(moments, moments.Length - 1));
    }

    [Fact]
    public async Task ANameAlreadyInASetOfColumns_IsNotOfferedAgain_WhileAnIndicatorsRolesMayRepeatIt()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var kernel = notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Kernel;

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        const string drop = """{"step": "drop.columns", "columns": ["age", " """;
        const string roles = """{"step": "feature.indicator", "column": "range", "indicator": "atr", "columns": ["fare", " """;

        var offered = (await kernel.GetCompletionsAsync(drop, drop.Length - 1)).Select(completion => completion.InsertText).ToArray();

        Assert.DoesNotContain("age", offered);
        Assert.Contains("fare", offered);
        Assert.Contains("fare", (await kernel.GetCompletionsAsync(roles, roles.Length - 1)).Select(completion => completion.InsertText));
    }

    [Fact]
    public async Task ANotebookNeverSaved_ReadsAWholePath_FromWhereItSays()
    {
        // A notebook that was never saved has no folder of its own: a relative path is read from the working
        // directory, and a whole one from where it says.
        Directory.CreateDirectory(Path.Join(_folder, "data"));
        File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "data", "titanic.csv"), overwrite: true);

        await using var notebook = await Notebook.OpenAsync();
        var read = notebook.AddBlock($$"""{"step": "read.csv", "path": {{System.Text.Json.JsonSerializer.Serialize(Path.Join(_folder, "data", "titanic.csv"))}}}""");

        await notebook.GestureAsync(read, StepRenderer.Show);

        Assert.Contains("891 rows", read.Outputs[1].Content, StringComparison.Ordinal);
    }

    [Theory]
    // A file that is not there, a column the schema declares and the rows lack, and a cell that cannot be read
    // as the kind its column was declared: each is said at the block, and nothing is shown there.
    [InlineData("""{"step": "read.csv", "path": "data/elsewhere.csv"}""", """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}]}""", "elsewhere.csv")]
    [InlineData("""{"step": "read.csv", "path": "data/titanic.csv"}""", """{"step": "declare", "remainder": "drop", "columns": [{"name": "colour", "kind": "text", "optional": false}]}""", "colour")]
    [InlineData("""{"step": "read.csv", "path": "data/titanic.csv"}""", """{"step": "declare", "remainder": "drop", "columns": [{"name": "sex", "kind": "number", "optional": false}]}""", "sex")]
    // Rows handed in: a notebook hands none in, and says where its rows come from instead.
    [InlineData("""{"step": "read.rows", "description": "rows handed in"}""", """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}]}""", "&#39;read.csv&#39;")]
    public async Task ARunTheRowsRefuse_IsSaidAtTheBlock_AndShowsNothingThere(string read, string declare, string named)
    {
        await using var notebook = await NotebookAsync(read, declare);
        var block = notebook.Scaffold.Cells[1];

        var gesture = await notebook.GestureAsync(block, StepRenderer.Show);

        Assert.False(gesture.StateChanged);
        Assert.Equal(2, block.Outputs.Count);
        Assert.True(block.Outputs[1].IsError);
        Assert.Contains(named, block.Outputs[1].Content, StringComparison.Ordinal);
        Assert.DoesNotContain(block.Id, Session(notebook).Shown.Keys);
    }

    [Fact]
    public async Task ShowingTheDataAtABlock_PushesTheRowsThereToThatBlock_PlacedByTheSplitBelow()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var declare = notebook.Scaffold.Cells[1];

        var gesture = await notebook.GestureAsync(declare, StepRenderer.Show);

        // Nothing in the notebook changed, so nothing is marked as changed; the block's output was pushed.
        Assert.False(gesture.StateChanged);
        Assert.Equal([declare.Id], notebook.Pushed.Distinct());
        Assert.Equal(2, declare.Outputs.Count);

        var grid = declare.Outputs[1].Content;
        var full = new Pipeline(Declared(), rows: null, SourceFolder.Of(_folder)).Run();

        Assert.Contains("891 rows", grid, StringComparison.Ordinal);
        Assert.Contains(string.Create(CultureInfo.InvariantCulture, $"{full.CountIn(Part.Train)} train"), grid, StringComparison.Ordinal);
        Assert.True(grid.Heads("fare"));
        Assert.False(grid.Heads("age_was_missing"));
    }

    [Fact]
    public async Task ARelativePath_IsReadFromTheNotebooksFolder_AsTheFileDoorReadsItFromTheFilesFolder()
    {
        // The process stands in the test's own folder, never the notebook's; both doors still open the same bytes.
        Assert.NotEqual(Path.GetFullPath(_folder), Path.GetFullPath(Environment.CurrentDirectory));

        await using var notebook = await NotebookAsync(Titanic);
        var last = notebook.Scaffold.Cells[^1];

        await notebook.GestureAsync(last, StepRenderer.Show);

        var fromTheFile = new Pipeline(Declared(), rows: null, SourceFolder.OfDocument(Path.Join(_folder, "titanic.pipeline.json"))).Run();

        Assert.Contains(string.Create(CultureInfo.InvariantCulture, $"{fromTheFile.Table.RowCount} rows"), last.Outputs[1].Content, StringComparison.Ordinal);
        Assert.True(last.Outputs[1].Content.Heads("age_was_missing"));
    }

    [Fact]
    public async Task AViewIsShownOnce_RunningTheBlockAgainShowsItsCardAlone()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var declare = notebook.Scaffold.Cells[1];

        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.Single(await notebook.RunAsync(declare));
    }

    [Fact]
    public async Task AnotherPage_ShowsTheNextRows()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var declare = notebook.Scaffold.Cells[1];

        await notebook.GestureAsync(declare, StepRenderer.Page, "1");

        Assert.Contains("rows 51–100 of 891", declare.Outputs[1].Content, StringComparison.Ordinal);

        await notebook.GestureAsync(declare, StepRenderer.Page, "not a page");

        Assert.Contains("rows 1–50 of 891", declare.Outputs[1].Content, StringComparison.Ordinal);

        await notebook.GestureAsync(declare, StepRenderer.Page, "99");

        Assert.Contains("rows 851–891 of 891", declare.Outputs[1].Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ABlockBelowOneThatIsNotAStep_ShowsNoData_AndSaysWhichBlockStopsIt()
    {
        await using var notebook = await NotebookAsync(
            Titanic[0], Titanic[1], """{"step": "split.stratified", "column": "survived"}""", Titanic[3]);
        var fill = notebook.Scaffold.Cells[3];

        await notebook.GestureAsync(fill, StepRenderer.Show);

        var refusal = Assert.Single(fill.Outputs, output => output.IsError);

        Assert.Contains("block 3", refusal.Content, StringComparison.Ordinal);
        Assert.Contains("split.stratified", refusal.Content, StringComparison.Ordinal);
        Assert.False(notebook.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out _));
    }

    [Fact]
    public async Task ARuleTheBlocksBreak_IsShownAtTheBlockThatBreaksIt()
    {
        await using var notebook = await NotebookAsync(
            Titanic[0], Titanic[1], Titanic[2], """{"step": "target", "column": "survived"}""", """{"step": "target", "column": "survived"}""");
        var second = notebook.Scaffold.Cells[4];

        await notebook.GestureAsync(second, StepRenderer.Show);

        var refusal = Assert.Single(second.Outputs, output => output.IsError);

        Assert.Contains("Step 5, &#39;target&#39;", refusal.Content, StringComparison.Ordinal);

        // The blocks above it still make a pipeline, and their rows are shown.
        await notebook.GestureAsync(notebook.Scaffold.Cells[3], StepRenderer.Show);

        Assert.Contains("891 rows", notebook.Scaffold.Cells[3].Outputs[1].Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheDeclarationTheBlocksMake_IsHandedToCSharpCellsAsText()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.GestureAsync(notebook.Scaffold.Cells[0], StepRenderer.Show);

        var handed = notebook.Scaffold.Variables.Get<string>(StepKernel.HandOver);

        Assert.NotNull(handed);
        Assert.Equal(Declared(), PipelineDeclaration.FromJson(handed, StepCatalog.BuiltIn()));
    }

    [Fact]
    public void TheHandOverKey_IsNeverAValidCSharpIdentifier()
    {
        // Verso declares a C# variable for every value a cell can name, once, and never again: a value handed
        // over under such a name would be read as it was the first time, forever. A key no variable can have is
        // read afresh every time, with Variables.Get.
        Assert.Equal("deepsharp.pipeline", StepKernel.HandOver);
        Assert.False(SyntaxFacts.IsValidIdentifier(StepKernel.HandOver));
    }

    [Fact]
    public async Task AGestureNoPartOfThisPackageKnows_ChangesNothing()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var declare = notebook.Scaffold.Cells[1];

        var gesture = await notebook.GestureAsync(declare, "somebody.else");

        Assert.False(gesture.StateChanged);
        Assert.Empty(declare.Outputs);
        Assert.Empty(notebook.Pushed);

        // A renderer Verso never loaded reaches no notebook at all, and says so rather than failing.
        var answer = await new StepRenderer().OnCellInteractionAsync(notebook.Gesture(declare, StepRenderer.Show));

        Assert.False(string.IsNullOrWhiteSpace(answer));
        Assert.Empty(declare.Outputs);
    }
}
