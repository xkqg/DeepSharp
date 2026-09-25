// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Text.Json;
using DeepSharp.Verso.Notebooks;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The grid's boxes write the declaration, never a value. Every column has a box saying whether it is in: unticking a
/// column no step reads excludes it in the schema with its kind, unticking one a step reads or made drops it after the
/// last step that reads it, and ticking either brings it back as it was; a column the schema does not name comes in
/// as text, where the source has it. Every column the schema takes has a box saying whether it is a category: ticking
/// it makes one and remembers the kind it was, unticking gives that kind back. A box sends the state it is in, so the
/// same state twice changes the notebook once; a box whose change would break a rule is drawn but cannot be clicked,
/// and a change sent anyway is not made, and says which step.
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

    private static IEnumerable<string> Names(Notebook notebook) => Declared(notebook).Columns.Select(column => column.Name);

    // The block a gesture rewrote is a new block in its place, so a test reads the notebook as it is now.
    private static CellModel Declare(Notebook notebook) => notebook.Scaffold.Cells[1];

    [Fact]
    public async Task UntickingAColumnNoStepReads_ExcludesItInTheSchemaWithItsKind_MarksTheNotebookChanged_AndShowsTheResult()
    {
        await using var notebook = await NotebookAsync(Titanic);

        var gesture = await notebook.TickAsync(Declare(notebook), StepRenderer.Include, "pclass", ticked: false);

        Assert.True(gesture.StateChanged);
        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Integer, Optional: false) { Excluded = true }, Declared(notebook).Columns[1]);
        Assert.DoesNotContain(Declared(notebook).Taking, column => column.Name == "pclass");
        Assert.Contains(Declare(notebook).Id, notebook.Pushed);
        Assert.False(Declare(notebook).Outputs[1].Content.Heads("pclass"));
        Assert.Equal(5, notebook.Scaffold.Cells.Count);
    }

    [Fact]
    public async Task UntickingAColumnAStepReads_IsADropAfterTheLastStepThatReadsIt()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "age", ticked: false);

        var steps = Steps(notebook);

        Assert.Contains(Declared(notebook).Taking, column => column.Name == "age");
        Assert.Equal(["read.csv", "declare", "split.stratified", "fill.missing", "drop.columns", "normalise"], steps.Select(step => step.Verb));
        Assert.Equal(["age"], ((DropColumnsStep)steps[4]).Columns);
        Assert.Empty(PipelineDeclaration.FaultsIn(steps));
    }

    [Fact]
    public async Task UntickingAColumnAStepMade_IsADropAfterIt_OrAnotherNameInTheDropAlreadyThere()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var normalise = notebook.Scaffold.Cells[4];

        await notebook.TickAsync(normalise, StepRenderer.Include, "age_was_missing", ticked: false);
        await notebook.TickAsync(notebook.Scaffold.Cells[4], StepRenderer.Include, "age", ticked: false);

        var steps = Steps(notebook);

        Assert.Equal(["read.csv", "declare", "split.stratified", "fill.missing", "drop.columns", "normalise"], steps.Select(step => step.Verb));
        Assert.Equal(["age_was_missing", "age"], ((DropColumnsStep)steps[4]).Columns);
    }

    [Fact]
    public async Task UntickingAndTickingAgain_GivesTheStepsBackAsTheyWere()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];
        var start = Steps(notebook);

        await notebook.TickAsync(read, StepRenderer.Include, "pclass", ticked: false);
        var back = await notebook.TickAsync(read, StepRenderer.Include, "pclass", ticked: true);

        Assert.True(back.StateChanged);
        Assert.Equal(start, Steps(notebook));
    }

    [Fact]
    public async Task TickingAgainAColumnAStepReads_TakesItsDropAwayAgain()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var start = Steps(notebook);

        await notebook.TickAsync(Declare(notebook), StepRenderer.Include, "age", ticked: false);

        Assert.True(Declare(notebook).Outputs[1].Content.Box(StepRenderer.Include, "age") is { Ticked: false, Enabled: true });

        var back = await notebook.TickAsync(Declare(notebook), StepRenderer.Include, "age", ticked: true);

        Assert.True(back.StateChanged);
        Assert.Equal(start, Steps(notebook));
        Assert.Equal(5, notebook.Scaffold.Cells.Count);
    }

    [Fact]
    public async Task TickingAColumnTheSchemaDoesNotName_DeclaresItAsText_WhereTheSourceHasIt()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];

        await notebook.GestureAsync(read, StepRenderer.Show);
        var gesture = await notebook.TickAsync(read, StepRenderer.Include, "sex", ticked: true);

        Assert.True(gesture.StateChanged);
        Assert.Equal(["survived", "pclass", "sex", "age", "fare"], Names(notebook));
        Assert.Equal(new ColumnDeclaration("sex", ColumnKind.Text, Optional: false), Declared(notebook).Columns[2]);
    }

    [Fact]
    public async Task TickingANewColumnBeforeTheSourceWasRead_IsNotMade_AndShowsTheSource_SoTheNextTickTakesItInWhereItStands()
    {
        // A grid kept in a saved notebook can be clicked before anything read its source, and where a column the
        // schema does not name stands among the others is known only from the source's header.
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];

        var first = await notebook.TickAsync(read, StepRenderer.Include, "sex", ticked: true);

        Assert.False(first.StateChanged);
        Assert.DoesNotContain("sex", Names(notebook));
        Assert.Contains(read.Outputs, output => output.IsError && output.Content.Contains("not read in this session", StringComparison.Ordinal));
        Assert.True(read.Outputs[^1].Content.Heads("sex"));

        var second = await notebook.TickAsync(read, StepRenderer.Include, "sex", ticked: true);

        Assert.True(second.StateChanged);
        Assert.Equal(["survived", "pclass", "sex", "age", "fare"], Names(notebook));
    }

    [Fact]
    public async Task TheSameStateTwice_ChangesTheNotebookOnce_AndRunsNothingTheSecondTime()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.TickAsync(Declare(notebook), StepRenderer.Include, "pclass", ticked: false);
        var once = Declare(notebook).Source;
        var id = Declare(notebook).Id;
        notebook.Pushed.Clear();

        for (var flood = 0; flood < 5; flood++)
        {
            var again = await notebook.TickAsync(Declare(notebook), StepRenderer.Include, "pclass", ticked: false);

            Assert.False(again.StateChanged);
        }

        Assert.Equal(once, Declare(notebook).Source);
        Assert.Equal(id, Declare(notebook).Id);
        Assert.Empty(notebook.Pushed);
        Assert.Equal(5, notebook.Scaffold.Cells.Count);
    }

    [Fact]
    public async Task ABoxSendsTheStateItIsIn_OnAKeyThatDoesNotChangeIt_AndNothingChanges()
    {
        // Verso's router sends a box's state on every key: a Tab past a ticked box sends true, past an unticked one
        // false; a Space sends the state it is in, and then the new one on the click and again on the change, from
        // the grid drawn before the change was made.
        await using var notebook = await NotebookAsync(Titanic);
        var declare = notebook.Scaffold.Cells[1];

        await notebook.GestureAsync(declare, StepRenderer.Show);
        notebook.Pushed.Clear();

        Assert.False((await notebook.TickAsync(declare, StepRenderer.Include, "pclass", ticked: true)).StateChanged);
        Assert.False((await notebook.TickAsync(declare, StepRenderer.Category, "pclass", ticked: false)).StateChanged);
        Assert.Empty(notebook.Pushed);

        bool[] space =
        [
            (await notebook.TickAsync(declare, StepRenderer.Include, "pclass", ticked: true)).StateChanged,
            (await notebook.TickAsync(declare, StepRenderer.Include, "pclass", ticked: false)).StateChanged,
            (await notebook.TickAsync(declare, StepRenderer.Include, "pclass", ticked: false)).StateChanged,
        ];

        Assert.Equal([false, true, false], space);
        Assert.True(Declared(notebook).Columns[1].Excluded);
    }

    [Fact]
    public async Task UntickingAColumnAStepBelowAlreadyTakesAway_ChangesNothing()
    {
        // Unticking age drops it after the fill that reads it. The schema's grid still shows it, since it is there
        // down to that drop, and unticking it again asks for what already is.
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.TickAsync(Declare(notebook), StepRenderer.Include, "age", ticked: false);
        string?[] once = [.. notebook.Scaffold.Cells.Select(cell => cell.Source)];
        notebook.Pushed.Clear();

        var again = await notebook.TickAsync(Declare(notebook), StepRenderer.Include, "age", ticked: false);

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

        await notebook.TickAsync(Declare(notebook), StepRenderer.Include, "pclass", ticked: false);

        Assert.DoesNotContain(Declare(notebook).Id, before);
        Assert.DoesNotContain(before[1], notebook.Scaffold.Cells.Select(cell => cell.Id));
        Assert.Contains(Declare(notebook).Id, notebook.Pushed);
        Assert.Equal(5, notebook.Scaffold.Cells.Count);

        notebook.Pushed.Clear();
        await notebook.TickAsync(Declare(notebook), StepRenderer.Include, "age", ticked: false);

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

        var gesture = await notebook.TickAsync(Declare(notebook), StepRenderer.Include, "age", ticked: false);

        Assert.False(gesture.StateChanged);
        Assert.Equal(before, notebook.Scaffold.Cells.Select(cell => cell.Source));
        Assert.Contains(Declare(notebook).Outputs, output => output.IsError && output.Content.Contains("layout", StringComparison.Ordinal));
    }

    [Fact]
    public async Task TickingACategory_ChangesItsKindInTheSchema_Once_RememberingTheKindItWas()
    {
        await using var notebook = await NotebookAsync(Titanic);

        var marked = await notebook.TickAsync(Declare(notebook), StepRenderer.Category, "pclass", ticked: true);
        var once = Declare(notebook).Source;
        var again = await notebook.TickAsync(Declare(notebook), StepRenderer.Category, "pclass", ticked: true);

        Assert.True(marked.StateChanged);
        Assert.False(again.StateChanged);
        Assert.Equal(once, Declare(notebook).Source);
        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Category, Optional: false) { Was = ColumnKind.Integer }, Declared(notebook).Columns[1]);
    }

    [Fact]
    public async Task UntickingACategory_GivesBackTheKindItWas()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var start = Steps(notebook);

        await notebook.TickAsync(Declare(notebook), StepRenderer.Category, "pclass", ticked: true);
        var back = await notebook.TickAsync(Declare(notebook), StepRenderer.Category, "pclass", ticked: false);

        Assert.True(back.StateChanged);
        Assert.Equal(start, Steps(notebook));
    }

    [Fact]
    public async Task ACategoryThatSaysWhatItWas_CanBeUnticked_AndOneThatDoesNot_Cannot()
    {
        // A category written before the schema said which kind a column was has no kind to go back to from here:
        // the schema block says which kind it takes.
        await using var notebook = await NotebookAsync(
            Titanic[0],
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "category", "optional": false, "was": "integer"}, {"name": "sex", "kind": "category", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
            Titanic[2],
            Titanic[3],
            Titanic[4]);
        var declare = notebook.Scaffold.Cells[1];

        await notebook.GestureAsync(declare, StepRenderer.Show);

        var grid = declare.Outputs[1].Content;

        Assert.True(grid.Box(StepRenderer.Category, "pclass") is { Ticked: true, Enabled: true });
        Assert.True(grid.Box(StepRenderer.Category, "sex") is { Ticked: true, Enabled: false });
        Assert.False((await notebook.TickAsync(declare, StepRenderer.Category, "sex", ticked: false)).StateChanged);
    }

    [Fact]
    public async Task MakingACategoryOfAColumnAStepBelowScales_CannotBeTicked_AndSentAnywayIsNotMade_SayingWhichStep()
    {
        // fare is normalised below, so it cannot become words. A grid drawn before the normalise was written still
        // offers it, and what that grid sends is judged by the rules as they are now.
        await using var notebook = await NotebookAsync(Titanic);
        var declare = notebook.Scaffold.Cells[1];
        var before = declare.Source;

        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.True(declare.Outputs[1].Content.Box(StepRenderer.Category, "fare") is { Ticked: false, Enabled: false });

        var gesture = await notebook.TickAsync(declare, StepRenderer.Category, "fare", ticked: true);

        Assert.False(gesture.StateChanged);
        Assert.Equal(before, declare.Source);
        Assert.Contains(declare.Outputs, output => output.IsError
            && output.Content.Contains("This change is not made", StringComparison.Ordinal)
            && output.Content.Contains("normalise", StringComparison.Ordinal));
        Assert.Contains(declare.Outputs, output => !output.IsError && output.Content.Heads("fare"));
    }

    [Fact]
    public async Task TheOnlyColumnASchemaTakes_CannotBeUnticked_AndSentAnywayIsNotMade_SayingWhy()
    {
        await using var notebook = await NotebookAsync(
            Titanic[0], """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}]}""");
        var declare = notebook.Scaffold.Cells[1];
        var before = declare.Source;

        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.True(declare.Outputs[1].Content.Box(StepRenderer.Include, "survived") is { Ticked: true, Enabled: false });

        var gesture = await notebook.TickAsync(declare, StepRenderer.Include, "survived", ticked: false);

        Assert.False(gesture.StateChanged);
        Assert.Equal(before, declare.Source);
        Assert.Contains(declare.Outputs, output => output.IsError && output.Content.Contains("no column would take part", StringComparison.Ordinal));
    }

    [Fact]
    public async Task TheAnswersBox_IsTicked_AndCannotBeUnticked()
    {
        await using var notebook = await NotebookAsync([.. Titanic, """{"step": "target", "column": "survived"}"""]);
        var declare = notebook.Scaffold.Cells[1];

        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.True(declare.Outputs[1].Content.Box(StepRenderer.Include, "survived") is { Ticked: true, Enabled: false });
        Assert.True(declare.Outputs[1].Content.Box(StepRenderer.Include, "pclass") is { Ticked: true, Enabled: true });
    }

    [Fact]
    public async Task AGestureForAColumnThatIsNotThere_OrOnABlockThatShowsNothing_ChangesNothing()
    {
        await using var notebook = await NotebookAsync(
            Titanic[0], Titanic[1], """{"step": "split.stratified", "column": "survived"}""", Titanic[3]);

        // The source is read, so a column it does not have is known not to be there.
        await notebook.GestureAsync(notebook.Scaffold.Cells[0], StepRenderer.Show);

        var absent = await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "colour", ticked: false);
        var taken = await notebook.TickAsync(notebook.Scaffold.Cells[0], StepRenderer.Include, "colour", ticked: true);
        var below = await notebook.TickAsync(notebook.Scaffold.Cells[3], StepRenderer.Include, "age", ticked: false);
        var unmarkable = await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Category, "colour", ticked: true);

        Assert.False(absent.StateChanged || taken.StateChanged || below.StateChanged || unmarkable.StateChanged);
        Assert.Equal(4, notebook.Scaffold.Cells.Count);
        Assert.Equal(Titanic[1], notebook.Scaffold.Cells[1].Source);
    }

    [Fact]
    public async Task AGestureThatNamesNoColumn_OrSendsNoState_ChangesNothing()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var declare = notebook.Scaffold.Cells[1];
        var before = declare.Source;
        string[] said =
        [
            StepRenderer.Include,
            $"{StepRenderer.Include} ",
            $"{StepRenderer.Include} {{\"name\": \"pclass\"}}",
            $"{StepRenderer.Include} {{\"column\": 3}}",
            $"{StepRenderer.Include} [\"pclass\"]",
            $"{StepRenderer.Include} {{\"column\": ",
            "deepsharp.unknown {\"column\": \"pclass\"}",
        ];

        foreach (var interaction in said)
        {
            Assert.False((await notebook.GestureAsync(declare, interaction, "false")).StateChanged);
        }

        Assert.False((await notebook.GestureAsync(declare, Notebook.BoxAction(StepRenderer.Include, "pclass"), "maybe")).StateChanged);
        Assert.Equal(before, declare.Source);
    }

    [Fact]
    public async Task AColumnWhoseNameHoldsQuotesAndMarkup_IsDrawnAsText_AndTakenOutByItsName()
    {
        const string odd = "he said \"hi\" | <b> & co";

        await File.WriteAllTextAsync(Path.Join(_folder, "odd.csv"), "id,\"he said \"\"hi\"\" | <b> & co\"\n1,a\n2,b\n", TestContext.Current.CancellationToken);
        await using var notebook = await NotebookAsync(
            """{"step": "read.csv", "path": "odd.csv"}""",
            $$"""{"step": "declare", "remainder": "drop", "columns": [{"name": "id", "kind": "integer", "optional": false}, {"name": {{JsonSerializer.Serialize(odd)}}, "kind": "text", "optional": false}]}""");
        var declare = notebook.Scaffold.Cells[1];

        await notebook.GestureAsync(declare, StepRenderer.Show);

        var grid = declare.Outputs[1].Content;
        var box = grid.Box(StepRenderer.Include, odd);

        Assert.Contains($"<th>{WebUtility.HtmlEncode(odd)} <label>", grid, StringComparison.Ordinal);
        Assert.DoesNotContain("<b>", grid, StringComparison.Ordinal);
        Assert.True(box is { Ticked: true, Enabled: true });

        var gesture = await notebook.GestureAsync(declare, box!.Value.Action, "false");

        Assert.True(gesture.StateChanged);
        Assert.True(Declared(notebook).Columns.Single(column => column.Name == odd).Excluded);
    }

    [Fact]
    public async Task AChange_ClearsEveryViewItMadeStale_AndLeavesTheOthers()
    {
        // Unticking the marker the fill made is a drop right after the fill, below the split. The rows at the
        // source are worked out from the steps down to the split and did not change, and nothing their grid draws
        // changed either; the rows at the normalise below the drop did, and its view is gone.
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];
        var normalise = notebook.Scaffold.Cells[4];

        await notebook.GestureAsync(read, StepRenderer.Show);
        await notebook.GestureAsync(normalise, StepRenderer.Show);
        await notebook.TickAsync(notebook.Scaffold.Cells[3], StepRenderer.Include, "age_was_missing", ticked: false);

        Assert.Equal(2, read.Outputs.Count);
        Assert.Empty(normalise.Outputs);
    }

    [Fact]
    public async Task AGridWhoseBoxesAChangeMadeStale_IsCleared_ThoughItsRowsAreTheSame()
    {
        // Unticking age drops it below the split, so the rows at the source are what they were; but their grid
        // drew age as in, and it no longer is. A grid that draws what no longer holds is cleared.
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];

        await notebook.GestureAsync(read, StepRenderer.Show);

        Assert.True(read.Outputs[1].Content.Box(StepRenderer.Include, "age") is { Ticked: true, Enabled: true });

        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "age", ticked: false);

        Assert.Empty(read.Outputs);
    }

    [Fact]
    public async Task AViewOfABlockDeletedSince_IsForgotten_WhenAChangeIsMade()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var normalise = notebook.Scaffold.Cells[4];

        await notebook.GestureAsync(normalise, StepRenderer.Show);
        notebook.Scaffold.RemoveCell(normalise.Id);

        var gesture = await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "pclass", ticked: false);

        Assert.True(gesture.StateChanged);
        Assert.DoesNotContain(normalise.Id, notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session.Shown.Keys);
    }

    [Fact]
    public async Task AChangeBetweenABlockAndTheSplitBelowIt_ClearsThatBlocksView_SinceTheSplitPlacesItsRows()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];

        await notebook.GestureAsync(read, StepRenderer.Show);
        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Category, "pclass", ticked: true);

        Assert.Empty(read.Outputs);
    }

    [Fact]
    public async Task TheGrid_DrawsWhetherEachColumnIsIn_AndWhetherEachColumnTheSchemaTakesIsACategory()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var fill = notebook.Scaffold.Cells[3];

        await notebook.GestureAsync(fill, StepRenderer.Show);

        var grid = fill.Outputs[1].Content;

        Assert.True(grid.Box(StepRenderer.Include, "age_was_missing") is { Ticked: true, Enabled: true });
        Assert.True(grid.Box(StepRenderer.Include, "pclass") is { Ticked: true, Enabled: true });
        Assert.True(grid.Box(StepRenderer.Category, "pclass") is { Ticked: false, Enabled: true });
        Assert.Null(grid.Box(StepRenderer.Category, "age_was_missing"));
    }

    [Fact]
    public async Task TheSourcesGrid_TicksTheColumnsThatAreIn_AndOffersEveryOtherToBeTakenIn()
    {
        // At the source every column of the file is there. The schema takes four and leaves the rest out, and a
        // drop further down takes age away: each of those is unticked, and ticking it takes it in.
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];

        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "age", ticked: false);
        await notebook.GestureAsync(read, StepRenderer.Show);

        var grid = read.Outputs[1].Content;

        Assert.True(grid.Heads("sex"));
        Assert.True(grid.Box(StepRenderer.Include, "sex") is { Ticked: false, Enabled: true });
        Assert.True(grid.Box(StepRenderer.Include, "age") is { Ticked: false, Enabled: true });
        Assert.True(grid.Box(StepRenderer.Include, "pclass") is { Ticked: true, Enabled: true });
        Assert.Null(grid.Box(StepRenderer.Category, "sex"));
        Assert.False((await notebook.TickAsync(read, StepRenderer.Include, "sex", ticked: false)).StateChanged);
    }

    [Fact]
    public async Task TheSourcesGrid_DrawsNoCategoryBox_ForAColumnTheSchemaExcludes()
    {
        // At the source every column of the file is there, an excluded one too. The schema names it and takes no
        // part of it, so there is nothing about it to mark; ticking it takes it in again.
        await using var notebook = await NotebookAsync(
            Titanic[0],
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false, "excluded": true}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
            Titanic[2],
            Titanic[3],
            Titanic[4]);
        var read = notebook.Scaffold.Cells[0];

        await notebook.GestureAsync(read, StepRenderer.Show);

        var grid = read.Outputs[1].Content;

        Assert.True(grid.Heads("pclass"));
        Assert.True(grid.Box(StepRenderer.Include, "pclass") is { Ticked: false, Enabled: true });
        Assert.Null(grid.Box(StepRenderer.Category, "pclass"));
        Assert.NotNull(grid.Box(StepRenderer.Category, "survived"));
    }

    [Fact]
    public async Task UntickingADeclaredColumnOnTheSourcesGrid_ExcludesItInTheSchema_AsFromAnyOther()
    {
        await using var notebook = await NotebookAsync(Titanic);

        var gesture = await notebook.TickAsync(notebook.Scaffold.Cells[0], StepRenderer.Include, "pclass", ticked: false);

        Assert.True(gesture.StateChanged);
        Assert.True(Declared(notebook).Columns[1].Excluded);
    }
}
