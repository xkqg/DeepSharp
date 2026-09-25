// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The schema's block lists every column of the source as one row — its name, its first values, whether it is in and
/// its kind — so nothing a person leaves out ever disappears from sight. A box on a row takes the column in or leaves
/// it out through the same column rules as the grid, and the list is drawn again on the schema's block. A list carries
/// what it was drawn from, so one drawn before the blocks changed is drawn again rather than acted on; and it finds the
/// schema's block by what it carries, not by the block it was drawn on, which a change writes anew.
/// </summary>
public sealed class ColumnListTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-list-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    private static readonly string[] Header =
        ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "class", "who", "adult_male", "deck", "embark_town", "alive", "alone"];

    public ColumnListTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string NotebookPath => Path.Join(_folder, "titanic.verso");

    private string ColumnsFile => Path.Join(_folder, "titanic.columns.json");

    private async Task<Notebook> NotebookAsync(params string[] blocks)
    {
        var notebook = await Notebook.OpenAsync(NotebookPath);

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private static IReadOnlyList<IPipelineStep> Steps(Notebook notebook) =>
        [.. notebook.Scaffold.Cells.Where(cell => cell.Type == StepCellType.StepType).Select(cell => NotebookVerbs.Catalog().ReadStep(cell.Source))];

    private static DeclareStep Declared(Notebook notebook) => Steps(notebook).OfType<DeclareStep>().Single();

    // The schema's block: a change writes it anew, so a test reads the notebook as it is now.
    private static CellModel Schema(Notebook notebook) => notebook.Scaffold.Cells[1];

    // The list the schema's block shows, as its page is written.
    private static string List(Notebook notebook) =>
        Schema(notebook).Outputs.Single(output => output.Content.Contains("<tr data-column=", StringComparison.Ordinal)).Content;

    private static Task<CellInteractionContext> ChooseAsync(Notebook notebook) => notebook.GestureAsync(Schema(notebook), StepRenderer.Columns);

    // A row's box, sent the way the router sends it: its own action, ticked or not.
    private static Task<CellInteractionContext> TickAsync(Notebook notebook, CellModel cell, string column, bool ticked) =>
        notebook.GestureAsync(cell, List(notebook).Row(column).Included.Action, ticked ? "true" : "false");

    [Fact]
    public void TheSchemasCard_OffersToChooseTheColumns_AndNoOtherCardDoes()
    {
        var catalog = NotebookVerbs.Catalog();

        var schema = StepCard.Of(catalog.ReadStep(Titanic[1]), "the schema").Content;
        var source = StepCard.Of(catalog.ReadStep(Titanic[0]), "the source").Content;

        Assert.Contains($"data-action=\"{StepRenderer.Columns}\"", schema, StringComparison.Ordinal);
        Assert.Contains("Choose the columns", schema, StringComparison.Ordinal);
        Assert.DoesNotContain(StepRenderer.Columns, source, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ChoosingTheColumns_ListsEverySourceColumnInItsOrder_WithItsFirstValuesItsKindAndABoxTickedWhenItIsIn()
    {
        await using var notebook = await NotebookAsync(Titanic);

        var chosen = await ChooseAsync(notebook);
        var list = List(notebook);

        Assert.False(chosen.StateChanged);
        Assert.Equal(Header, list.Rows().Select(row => row.Column));
        Assert.Equal("0, 1, 1, 1, 0, 0, 0, 0, 1, 1", list.Row("survived").Values);
        Assert.Equal("22.0, 38.0, 26.0, 35.0, 35.0, ∅, 54.0, 2.0, 27.0, 14.0", list.Row("age").Values);
        Assert.Equal("integer", list.Row("survived").Kind.Value);
        Assert.Equal(string.Empty, list.Row("sex").Kind.Value);
        Assert.Equal("text", list.Row("sex").ShownKind);
        Assert.True(list.Row("survived").Included is { Ticked: true, Enabled: true, CarriesAPayload: false });
        Assert.True(list.Row("sex").Included is { Ticked: false, Enabled: true });
        Assert.Equal([.. Titanic.Select(text => NotebookVerbs.Catalog().ReadStep(text))], Steps(notebook));
    }

    [Fact]
    public async Task AColumnTheSchemaDeclaresAndTheSourceLacks_IsListedAfterTheSourcesColumns_AsNotInTheSource()
    {
        await using var notebook = await NotebookAsync(
            Titanic[0],
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "colour", "kind": "text", "optional": true}]}""");

        await ChooseAsync(notebook);

        Assert.Equal([.. Header, "colour"], List(notebook).Rows().Select(row => row.Column));
        Assert.Equal("not in the source", List(notebook).Row("colour").Mark);
        Assert.Equal(string.Empty, List(notebook).Row("colour").Values);
    }

    [Fact]
    public async Task TickingARow_TakesTheColumnInWithTheKindItShows_WhereTheSourceHasIt_AndListsItAgain()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await ChooseAsync(notebook);
        var taken = await TickAsync(notebook, Schema(notebook), "sex", ticked: true);

        Assert.True(taken.StateChanged);
        Assert.Equal(["survived", "pclass", "sex", "age", "fare"], Declared(notebook).Columns.Select(column => column.Name));
        Assert.Equal(ColumnKind.Text, Declared(notebook).Columns[2].Kind);
        Assert.True(List(notebook).Row("sex").Included.Ticked);
    }

    [Fact]
    public async Task UntickingARow_LeavesTheColumnOut_AsTheGridDoes()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await ChooseAsync(notebook);
        await TickAsync(notebook, Schema(notebook), "pclass", ticked: false);
        await TickAsync(notebook, Schema(notebook), "age", ticked: false);

        Assert.True(Declared(notebook).Columns.Single(column => column.Name == "pclass").Excluded);
        Assert.Equal(["age"], Steps(notebook).OfType<DropColumnsStep>().Single().Columns);
        Assert.False(List(notebook).Row("pclass").Included.Ticked);
        Assert.False(List(notebook).Row("age").Included.Ticked);
    }

    [Fact]
    public async Task AListDrawnBeforeTheBlocksChanged_IsDrawnAgain_AndItsBoxChangesNothing()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await ChooseAsync(notebook);
        var old = List(notebook).Row("sex").Included.Action;

        await notebook.TickAsync(Schema(notebook), StepRenderer.Category, "pclass", ticked: true);
        var changed = Steps(notebook);

        var stale = await notebook.GestureAsync(Schema(notebook), old, "true");

        Assert.False(stale.StateChanged);
        Assert.Equal(changed, Steps(notebook));
        Assert.Equal("category", List(notebook).Row("pclass").Kind.Value);
    }

    [Fact]
    public async Task AListGestureInASessionThatHasNotReadTheSource_IsNotMade_AndTheListIsDrawnAgain()
    {
        await using var drawn = await NotebookAsync(Titanic);

        await ChooseAsync(drawn);
        var action = List(drawn).Row("sex").Included.Action;

        await using var fresh = await NotebookAsync(Titanic);

        var refused = await fresh.GestureAsync(Schema(fresh), action, "true");

        Assert.False(refused.StateChanged);
        Assert.Equal(Steps(drawn), Steps(fresh));
        Assert.Contains(Schema(fresh).Outputs, output => output.IsError
            && WebUtility.HtmlDecode(output.Content).Contains("Choose the columns again", StringComparison.Ordinal));
        Assert.Equal(Header, List(fresh).Rows().Select(row => row.Column));
    }

    [Fact]
    public async Task AListGestureNamingTheSchemaBlockAChangeReplaced_ActsThroughWhatItCarries()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await ChooseAsync(notebook);
        var replaced = Schema(notebook);
        await TickAsync(notebook, replaced, "sex", ticked: true);

        Assert.DoesNotContain(notebook.Scaffold.Cells, cell => cell.Id == replaced.Id);

        // The next key of the same walk still names the block the list was drawn on.
        var acted = await TickAsync(notebook, replaced, "sex", ticked: false);

        Assert.True(acted.StateChanged);
        Assert.DoesNotContain(Declared(notebook).Taking, column => column.Name == "sex");
    }

    [Fact]
    public async Task TheList_MarksNewTheSourcesColumnsTheSavedColumnsNeverShowed_ThenSavesTheHeaderItShowed()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var declaration = new PipelineDeclaration(Steps(notebook));
        string[] shown = ["survived", "pclass", "sex", "age", "fare"];

        await File.WriteAllTextAsync(ColumnsFile, PipelinePreset.Of(declaration, shown).ToJson(), TestContext.Current.CancellationToken);
        await ChooseAsync(notebook);

        var marks = List(notebook).Rows().Where(row => row.Mark == "new").Select(row => row.Column);
        var saved = PipelinePreset.FromJson(await File.ReadAllTextAsync(ColumnsFile, TestContext.Current.CancellationToken), NotebookVerbs.Catalog());

        Assert.Equal(Header.Except(shown), marks);
        Assert.Equal(Header, saved.Source);
        Assert.Equal(declaration.Steps[1], saved.Declare);

        // Shown once, the newness is spent.
        await ChooseAsync(notebook);

        Assert.DoesNotContain(List(notebook).Rows(), row => row.Mark == "new");
    }

    [Fact]
    public async Task SavedColumnsThatNeverSawTheSource_MarkNewEveryColumnTheyDoNotName()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await File.WriteAllTextAsync(ColumnsFile, PipelinePreset.Of(new PipelineDeclaration(Steps(notebook)), header: null).ToJson(), TestContext.Current.CancellationToken);
        await ChooseAsync(notebook);

        Assert.Equal(Header.Except(["survived", "pclass", "age", "fare"]), List(notebook).Rows().Where(row => row.Mark == "new").Select(row => row.Column));
    }

    [Fact]
    public async Task WithoutSavedColumns_TheListMarksNothingAndSavesNothing()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await ChooseAsync(notebook);

        Assert.DoesNotContain(List(notebook).Rows(), row => row.Mark == "new");
        Assert.False(File.Exists(ColumnsFile));
    }

    [Fact]
    public async Task SavedColumnsThatCannotBeRead_AreNotWrittenOver_AndTheListSaysSo()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await File.WriteAllTextAsync(ColumnsFile, "not the saved columns", TestContext.Current.CancellationToken);
        await ChooseAsync(notebook);

        Assert.Equal("not the saved columns", await File.ReadAllTextAsync(ColumnsFile, TestContext.Current.CancellationToken));
        Assert.Contains(Schema(notebook).Outputs, output => output.IsError
            && output.Content.Contains("cannot be read, so they are not written over", StringComparison.Ordinal));
        Assert.Equal(Header, List(notebook).Rows().Select(row => row.Column));
    }

    [Fact]
    public async Task AChangeMadeFromTheList_SavesTheHeaderTheListShowed()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await ChooseAsync(notebook);
        await TickAsync(notebook, Schema(notebook), "pclass", ticked: false);

        var saved = PipelinePreset.FromJson(await File.ReadAllTextAsync(ColumnsFile, TestContext.Current.CancellationToken), NotebookVerbs.Catalog());

        Assert.Equal(Header, saved.Source);
        Assert.True(saved.Declare.Columns.Single(column => column.Name == "pclass").Excluded);
    }

    [Fact]
    public async Task AColumnTheSchemaDoesNotNameButTheSavedColumnsDo_ShowsTheirKind_AndIsTakenInWithIt()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var withSex = new PipelineDeclaration(new PipelineDeclaration(Steps(notebook)).Including("sex", ColumnKind.Category, Header));

        await File.WriteAllTextAsync(ColumnsFile, PipelinePreset.Of(withSex, Header).ToJson(), TestContext.Current.CancellationToken);
        await ChooseAsync(notebook);

        Assert.Equal("category", List(notebook).Row("sex").ShownKind);

        await TickAsync(notebook, Schema(notebook), "sex", ticked: true);

        Assert.Equal(ColumnKind.Category, Declared(notebook).Columns.Single(column => column.Name == "sex").Kind);
    }

    [Fact]
    public async Task TheOnlyColumnASchemaTakes_IsDrawnWithABoxThatCannotBeUnticked_AndSentAnywayIsNotMade()
    {
        await using var notebook = await NotebookAsync(
            Titanic[0], """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}]}""");

        await ChooseAsync(notebook);

        Assert.True(List(notebook).Row("survived").Included is { Ticked: true, Enabled: false });

        var refused = await TickAsync(notebook, Schema(notebook), "survived", ticked: false);

        Assert.False(refused.StateChanged);
        Assert.Contains(Schema(notebook).Outputs, output => output.IsError
            && output.Content.Contains("no column would take part", StringComparison.Ordinal));
        Assert.Equal("survived", Assert.Single(Declared(notebook).Taking).Name);
    }

    [Fact]
    public async Task ARowSentWithAKindTheNotebookDoesNotKnow_TakesTheColumnInAsText()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await ChooseAsync(notebook);
        var carried = ControlAction.Read(List(notebook).Row("sex").Included.Action)!.Value;
        carried.Carried[StepRenderer.KindKey] = "a kind nobody declares";

        await notebook.GestureAsync(Schema(notebook), ControlAction.Of(carried.Gesture, carried.Carried), "true");

        Assert.Equal(ColumnKind.Text, Declared(notebook).Columns.Single(column => column.Name == "sex").Kind);
    }

    [Fact]
    public async Task ARowSentToBlocksWithoutASchema_OrCarryingNoColumn_ChangesNothing()
    {
        await using var bare = await NotebookAsync(Titanic[0]);
        await using var notebook = await NotebookAsync(Titanic);
        var row = ControlAction.Of(StepRenderer.ListInclude, new() { [StepRenderer.ColumnKey] = "sex", [StepRenderer.KindKey] = "text" });
        var noColumn = ControlAction.Of(StepRenderer.ListInclude, new() { [StepRenderer.KindKey] = "text" });

        var withoutSchema = await bare.GestureAsync(bare.Scaffold.Cells[0], row, "true");
        var withoutColumn = await notebook.GestureAsync(Schema(notebook), noColumn, "true");

        Assert.False(withoutSchema.StateChanged);
        Assert.False(withoutColumn.StateChanged);
        Assert.Empty(bare.Scaffold.Cells[0].Outputs);
        Assert.Empty(Schema(notebook).Outputs);
    }

    [Fact]
    public void AListDrawnOverStepsWithoutASchema_ShowsTheSourcesColumns_WithNothingToTick()
    {
        var source = new SourceRows(CsvRowSource.FromText("a,b\n1,\n"), "fingerprint");

        var list = ColumnList.Of(new PipelineDeclaration([new ReadCsvStep("rows.csv")]), source, [], stored: null, "key", ListPicks.None, whole: true).Content;

        Assert.Equal(["a", "b"], list.Rows().Select(row => row.Column));
        Assert.Equal("∅", WebUtility.HtmlDecode(list.Row("b").Values));
        Assert.All(list.Rows(), row => Assert.True(row.Included is { Ticked: false, Enabled: false }));
    }

    [Fact]
    public async Task AListTheBlocksNoLongerMatch_IsClearedWhenAnotherBlockChangesThem()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var read = notebook.Scaffold.Cells[0];

        await ChooseAsync(notebook);
        var schema = Schema(notebook);

        // A drop below the fill: the schema's block stays, and the list on it no longer says what holds.
        await notebook.GestureAsync(read, StepRenderer.Show);
        await notebook.TickAsync(read, StepRenderer.Include, "age", ticked: false);

        Assert.Same(schema, Schema(notebook));
        Assert.DoesNotContain(Schema(notebook).Outputs, output => output.Content.Contains("<tr data-column=", StringComparison.Ordinal));
    }
}
