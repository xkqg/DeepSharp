// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Notebooks;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The grid's actions write the declaration, never a value. Excluding a column that no step reads takes it out of
/// the schema, since a column nobody names is not there; excluding one a step reads, or one a step made, is a
/// drop after the last step that reads it. Marking a column a category changes its kind in the schema, and the
/// encoder below takes it from there. Every gesture carries the state it wants, whole: the same gesture twice
/// changes the notebook once, and a change that would break a step is not made, and says which step.
/// </summary>
public sealed class GridActionTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-actions-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public GridActionTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

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
        [.. notebook.Scaffold.Cells.Where(cell => cell.Type == StepCellType.StepType).Select(cell => StepCatalog.BuiltIn().ReadStep(cell.Source))];

    private static DeclareStep Declared(Notebook notebook) => Steps(notebook).OfType<DeclareStep>().Single();

    // The buttons a grid's header carries for a column.
    private static string Excluding(string column) =>
        $"data-action=\"{StepRenderer.Drop}\" data-extension-id=\"{StepRenderer.Id}\" data-payload=\"{column}\"";

    private static string Marking(string column) =>
        $"data-action=\"{StepRenderer.Category}\" data-extension-id=\"{StepRenderer.Id}\" data-payload=\"{column}\"";

    // The block a gesture rewrote is a new block in its place, so a test reads the notebook as it is now.
    private static CellModel Declare(Notebook notebook) => notebook.Scaffold.Cells[1];

    [Fact]
    public async Task ExcludingAColumnNoStepReads_TakesItOutOfTheSchema_MarksTheNotebookChanged_AndShowsTheResult()
    {
        await using var notebook = await NotebookAsync(Titanic);

        var gesture = await notebook.GestureAsync(Declare(notebook), StepRenderer.Drop, "pclass");

        Assert.True(gesture.StateChanged);
        Assert.DoesNotContain(Declared(notebook).Columns, column => column.Name == "pclass");
        Assert.Contains(Declare(notebook).Id, notebook.Pushed);
        Assert.False(Declare(notebook).Outputs[1].Content.Heads("pclass"));
        Assert.Equal(5, notebook.Scaffold.Cells.Count);
    }

    [Fact]
    public async Task ExcludingAColumnAStepReads_IsADropAfterTheLastStepThatReadsIt()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Drop, "age");

        var steps = Steps(notebook);

        Assert.Contains(Declared(notebook).Columns, column => column.Name == "age");
        Assert.Equal(["read.csv", "declare", "split.stratified", "fill.missing", "drop.columns", "normalise"], steps.Select(step => step.Verb));
        Assert.Equal(["age"], ((DropColumnsStep)steps[4]).Columns);
        Assert.Empty(PipelineDeclaration.FaultsIn(steps));
    }

    [Fact]
    public async Task ExcludingAColumnAStepMade_IsADropAfterIt_OrAnotherNameInTheDropAlreadyThere()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var normalise = notebook.Scaffold.Cells[4];

        await notebook.GestureAsync(normalise, StepRenderer.Drop, "age_was_missing");
        await notebook.GestureAsync(notebook.Scaffold.Cells[4], StepRenderer.Drop, "age");

        var steps = Steps(notebook);

        Assert.Equal(["read.csv", "declare", "split.stratified", "fill.missing", "drop.columns", "normalise"], steps.Select(step => step.Verb));
        Assert.Equal(["age_was_missing", "age"], ((DropColumnsStep)steps[4]).Columns);
    }

    [Fact]
    public async Task TheSameGestureTwice_ChangesTheNotebookOnce_AndRunsNothingTheSecondTime()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.GestureAsync(Declare(notebook), StepRenderer.Drop, "pclass");
        var once = Declare(notebook).Source;
        var id = Declare(notebook).Id;
        notebook.Pushed.Clear();

        for (var flood = 0; flood < 5; flood++)
        {
            var again = await notebook.GestureAsync(Declare(notebook), StepRenderer.Drop, "pclass");

            Assert.False(again.StateChanged);
        }

        Assert.Equal(once, Declare(notebook).Source);
        Assert.Equal(id, Declare(notebook).Id);
        Assert.Empty(notebook.Pushed);
        Assert.Equal(5, notebook.Scaffold.Cells.Count);
    }

    [Fact]
    public async Task ExcludingAColumnAStepBelowAlreadyTakesAway_ChangesNothing()
    {
        // Excluding age drops it after the fill that reads it. The schema's grid still shows it, since it is there
        // down to that drop, and excluding it again asks for what already is.
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.GestureAsync(Declare(notebook), StepRenderer.Drop, "age");
        string?[] once = [.. notebook.Scaffold.Cells.Select(cell => cell.Source)];
        notebook.Pushed.Clear();

        var again = await notebook.GestureAsync(Declare(notebook), StepRenderer.Drop, "age");

        Assert.False(again.StateChanged);
        Assert.Equal(once, notebook.Scaffold.Cells.Select(cell => cell.Source));
        Assert.Empty(notebook.Pushed);
    }

    [Fact]
    public async Task ACommit_EndsWithARunOfABlockTheClientHasNeverSeen()
    {
        // Verso tells a front end nothing of a block's text a part changed, nor of a block a part added; a front end
        // reads the notebook again when a block it does not know shows output. So a rewritten block is replaced by a
        // new one holding its new text, and every block a commit writes is run.
        await using var notebook = await NotebookAsync(Titanic);
        Guid[] before = [.. notebook.Scaffold.Cells.Select(cell => cell.Id)];

        await notebook.GestureAsync(Declare(notebook), StepRenderer.Drop, "pclass");

        Assert.DoesNotContain(Declare(notebook).Id, before);
        Assert.DoesNotContain(before[1], notebook.Scaffold.Cells.Select(cell => cell.Id));
        Assert.Contains(Declare(notebook).Id, notebook.Pushed);
        Assert.Equal(5, notebook.Scaffold.Cells.Count);

        notebook.Pushed.Clear();
        await notebook.GestureAsync(Declare(notebook), StepRenderer.Drop, "age");

        var dropped = notebook.Scaffold.Cells[4];

        Assert.Equal("drop.columns", Steps(notebook)[4].Verb);
        Assert.Contains(dropped.Id, notebook.Pushed);
    }

    [Fact]
    public async Task AChangeUnderALayoutThatCannotAddOrRemoveABlock_IsNotMade_AndSaysWhy()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var layout = notebook.Host.GetLayouts().First(each => !each.Capabilities.HasFlag(LayoutCapabilities.CellInsert));
        string?[] before = [.. notebook.Scaffold.Cells.Select(cell => cell.Source)];

        notebook.Scaffold.NotebookOps.SetActiveLayout(layout.LayoutId);

        var gesture = await notebook.GestureAsync(Declare(notebook), StepRenderer.Drop, "age");

        Assert.False(gesture.StateChanged);
        Assert.Equal(before, notebook.Scaffold.Cells.Select(cell => cell.Source));
        Assert.Contains(Declare(notebook).Outputs, output => output.IsError && output.Content.Contains("layout", StringComparison.Ordinal));
    }

    [Fact]
    public async Task MarkingAColumnACategory_ChangesItsKindInTheSchema_Once()
    {
        await using var notebook = await NotebookAsync(Titanic);

        var marked = await notebook.GestureAsync(Declare(notebook), StepRenderer.Category, "pclass");
        var once = Declare(notebook).Source;
        var again = await notebook.GestureAsync(Declare(notebook), StepRenderer.Category, "pclass");

        Assert.True(marked.StateChanged);
        Assert.False(again.StateChanged);
        Assert.Equal(once, Declare(notebook).Source);
        Assert.Equal(ColumnKind.Category, Declared(notebook).Columns.Single(column => column.Name == "pclass").Kind);
    }

    [Fact]
    public async Task AChangeThatWouldBreakAStep_IsNotMade_AndSaysWhichStep()
    {
        // fare is normalised below, so it cannot become words.
        await using var notebook = await NotebookAsync(Titanic);
        var declare = notebook.Scaffold.Cells[1];
        var before = declare.Source;

        var gesture = await notebook.GestureAsync(declare, StepRenderer.Category, "fare");

        Assert.False(gesture.StateChanged);
        Assert.Equal(before, declare.Source);
        Assert.Contains(declare.Outputs, output => output.IsError
            && output.Content.Contains("This change is not made", StringComparison.Ordinal)
            && output.Content.Contains("normalise", StringComparison.Ordinal));
        Assert.Contains(declare.Outputs, output => !output.IsError && output.Content.Heads("fare"));
    }

    [Fact]
    public async Task AGestureForAColumnThatIsNotThere_OrOnABlockThatShowsNothing_ChangesNothing()
    {
        await using var notebook = await NotebookAsync(
            Titanic[0], Titanic[1], """{"step": "split.stratified", "column": "survived"}""", Titanic[3]);

        var absent = await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Drop, "colour");
        var below = await notebook.GestureAsync(notebook.Scaffold.Cells[3], StepRenderer.Drop, "age");
        var unmarkable = await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Category, "colour");

        Assert.False(absent.StateChanged || below.StateChanged || unmarkable.StateChanged);
        Assert.Equal(4, notebook.Scaffold.Cells.Count);
    }

    [Fact]
    public async Task AChange_ClearsEveryViewItMadeStale_AndLeavesTheOthers()
    {
        // Excluding the marker the fill made is a drop right after the fill, below the split. The rows at the
        // source are worked out from the steps down to the split and did not change, and nothing their grid offers
        // changed either; the rows at the normalise below the drop did, and its view is gone.
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];
        var normalise = notebook.Scaffold.Cells[4];

        await notebook.GestureAsync(read, StepRenderer.Show);
        await notebook.GestureAsync(normalise, StepRenderer.Show);
        await notebook.GestureAsync(notebook.Scaffold.Cells[3], StepRenderer.Drop, "age_was_missing");

        Assert.Equal(2, read.Outputs.Count);
        Assert.Empty(normalise.Outputs);
    }

    [Fact]
    public async Task AGridWhoseOffersAChangeMadeStale_IsCleared_ThoughItsRowsAreTheSame()
    {
        // Excluding age drops it below the split, so the rows at the source are what they were; but their grid
        // offered to exclude age, and no longer can. A grid that offers what no longer holds is cleared.
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];

        await notebook.GestureAsync(read, StepRenderer.Show);

        Assert.Contains(Excluding("age"), read.Outputs[1].Content, StringComparison.Ordinal);

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Drop, "age");

        Assert.Empty(read.Outputs);
    }

    [Fact]
    public async Task AViewOfABlockDeletedSince_IsForgotten_WhenAChangeIsMade()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var normalise = notebook.Scaffold.Cells[4];

        await notebook.GestureAsync(normalise, StepRenderer.Show);
        notebook.Scaffold.RemoveCell(normalise.Id);

        var gesture = await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Drop, "pclass");

        Assert.True(gesture.StateChanged);
        Assert.DoesNotContain(normalise.Id, notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session.Shown.Keys);
    }

    [Fact]
    public async Task AChangeBetweenABlockAndTheSplitBelowIt_ClearsThatBlocksView_SinceTheSplitPlacesItsRows()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];

        await notebook.GestureAsync(read, StepRenderer.Show);
        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Category, "pclass");

        Assert.Empty(read.Outputs);
    }

    [Fact]
    public async Task TheGrid_OffersToExcludeAColumn_AndToMarkADeclaredOneACategory()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var fill = notebook.Scaffold.Cells[3];

        await notebook.GestureAsync(fill, StepRenderer.Show);

        var grid = fill.Outputs[1].Content;

        Assert.Contains(Excluding("age_was_missing"), grid, StringComparison.Ordinal);
        Assert.Contains(Marking("pclass"), grid, StringComparison.Ordinal);
        Assert.DoesNotContain(Marking("age_was_missing"), grid, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheGrid_OffersToExcludeOnlyAColumnThatReachesTheEnd()
    {
        // At the source every column of the file is there. The schema keeps four and leaves the rest out, and a
        // drop further down takes age away: excluding any of those asks for what already is, so none is offered.
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Drop, "age");
        await notebook.GestureAsync(read, StepRenderer.Show);

        var grid = read.Outputs[1].Content;

        Assert.True(grid.Heads("sex"));
        Assert.DoesNotContain(Excluding("sex"), grid, StringComparison.Ordinal);
        Assert.DoesNotContain(Excluding("age"), grid, StringComparison.Ordinal);
        Assert.Contains(Excluding("pclass"), grid, StringComparison.Ordinal);
        Assert.False((await notebook.GestureAsync(read, StepRenderer.Drop, "sex")).StateChanged);
    }

    [Fact]
    public async Task ExcludingADeclaredColumnFromTheSourcesGrid_TakesItOutOfTheSchema_AsFromAnyOther()
    {
        await using var notebook = await NotebookAsync(Titanic);

        var gesture = await notebook.GestureAsync(notebook.Scaffold.Cells[0], StepRenderer.Drop, "pclass");

        Assert.True(gesture.StateChanged);
        Assert.DoesNotContain(Declared(notebook).Columns, column => column.Name == "pclass");
    }
}
