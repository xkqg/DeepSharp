// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The columns saved beside a notebook are taken over in two presses. The toolbar's button lists, at the schema's
/// block, every column whose decision taking them over would change — and the output, the schema's order, and the
/// source's columns the saved file never showed — and changes nothing. The list's own box applies what it listed, once,
/// and only while the blocks are still the ones it was listed for. A take-over whose blocks would break a rule is
/// listed with every fault and offers nothing to apply.
/// </summary>
public sealed class TakeOverActionTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-takeover-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public TakeOverActionTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

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

    private static TakeOverAction Button(Notebook notebook) => notebook.Host.GetToolbarActions().OfType<TakeOverAction>().Single();

    private static IReadOnlyList<IPipelineStep> Steps(Notebook notebook) =>
        [.. notebook.Scaffold.Cells.Where(cell => cell.Type == StepCellType.StepType).Select(cell => NotebookVerbs.Catalog().ReadStep(cell.Source))];

    private static PipelineDeclaration Blocks(Notebook notebook) => new(Steps(notebook));

    // The decisions saved beside the notebook, as the notebook itself would have saved them.
    private void Saved(IReadOnlyList<IPipelineStep> decided, IReadOnlyList<string>? source = null) =>
        File.WriteAllText(ColumnsFile, PipelinePreset.Of(new PipelineDeclaration(decided), source).ToJson());

    // What the schema's block shows last, as a person reads it.
    private static string Card(Notebook notebook) => WebUtility.HtmlDecode(notebook.Scaffold.Cells[1].Outputs[^1].Content);

    private static DrawnBox? ApplyBox(Notebook notebook) =>
        notebook.Scaffold.Cells[1].Outputs[^1].Content.Boxes().Cast<DrawnBox?>()
            .SingleOrDefault(box => box!.Value.Action.StartsWith($"{StepRenderer.Apply} ", StringComparison.Ordinal));

    private async Task ListAsync(Notebook notebook) => await Button(notebook).ExecuteAsync(new ToolbarGesture(notebook, NotebookPath));

    // The list's box, sent the way the router sends it: ticked or not.
    private static Task<CellInteractionContext> ApplyAsync(Notebook notebook, string action, bool ticked) =>
        notebook.GestureAsync(notebook.Scaffold.Cells[1], action, ticked ? "true" : "false");

    [Fact]
    public async Task TheButton_IsOffered_ForASavedNotebookThatMakesAPipeline_WithColumnsSavedBesideIt()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var button = Button(notebook);

        Assert.False(await button.IsEnabledAsync(new ToolbarGesture(notebook, NotebookPath)));

        Saved(Blocks(notebook).Steps);

        Assert.True(await button.IsEnabledAsync(new ToolbarGesture(notebook, NotebookPath)));
        Assert.False(await button.IsEnabledAsync(new ToolbarGesture(notebook, filePath: null)));

        notebook.AddBlock("""{"step": "normalise", "column": "colour", "scale": "standard", "outOfRange": "pass"}""");

        Assert.False(await button.IsEnabledAsync(new ToolbarGesture(notebook, NotebookPath)));
    }

    [Fact]
    public async Task TheButton_PressedWhereItIsNotOffered_SaysWhy_AndChangesNothing()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var button = Button(notebook);

        var nothingSaved = await Assert.ThrowsAsync<InvalidOperationException>(() => button.ExecuteAsync(new ToolbarGesture(notebook, NotebookPath)));
        var neverSaved = await Assert.ThrowsAsync<InvalidOperationException>(() => button.ExecuteAsync(new ToolbarGesture(notebook, filePath: null)));

        Saved(Blocks(notebook).Steps);
        notebook.AddBlock("""{"step": "normalise", "column": "colour", "scale": "standard", "outOfRange": "pass"}""");

        var noPipeline = await Assert.ThrowsAsync<InvalidOperationException>(() => button.ExecuteAsync(new ToolbarGesture(notebook, NotebookPath)));

        Assert.Equal("There are no columns saved beside the notebook to take over.", nothingSaved.Message);
        Assert.Equal("A notebook never saved has nothing saved beside it to take over.", neverSaved.Message);
        Assert.StartsWith("The blocks do not make a pipeline yet, so there is nothing to take the saved columns into: block 6", noPipeline.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ANotebookWithoutBlocks_HasNothingToTakeTheSavedColumnsInto()
    {
        await using var notebook = await NotebookAsync();

        File.WriteAllText(ColumnsFile, PipelinePreset.Of(
            Pdd.Create().ReadCsv("titanic.csv").Declare(columns => columns.Integer("survived")).Declaration, header: null).ToJson());

        Assert.False(await Button(notebook).IsEnabledAsync(new ToolbarGesture(notebook, NotebookPath)));

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => ListAsync(notebook));

        Assert.Equal("There are no blocks to take the saved columns into.", refused.Message);
    }

    [Fact]
    public async Task TakingOver_ListsEveryColumnWhoseDecisionChanges_AtTheSchema_AndChangesNothingYet()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var before = Blocks(notebook);
        var decided = new PipelineDeclaration(new PipelineDeclaration(before.Excluding("pclass")).WithKind("survived", ColumnKind.Category));

        Saved(decided.Steps);
        var file = await File.ReadAllTextAsync(ColumnsFile, TestContext.Current.CancellationToken);
        await ListAsync(notebook);

        var card = Card(notebook);

        Assert.Equal(before, Blocks(notebook));
        Assert.Equal(file, await File.ReadAllTextAsync(ColumnsFile, TestContext.Current.CancellationToken));
        Assert.Contains("Taking over the columns saved beside the notebook changes:", card, StringComparison.Ordinal);
        Assert.Contains("<code>survived</code>: taken, integer → taken, category (was integer)", card, StringComparison.Ordinal);
        Assert.Contains("<code>pclass</code>: taken, integer → excluded, integer", card, StringComparison.Ordinal);
        Assert.True(card.IndexOf("<code>survived</code>", StringComparison.Ordinal) < card.IndexOf("<code>pclass</code>", StringComparison.Ordinal));
        Assert.True(ApplyBox(notebook) is { Ticked: false, Enabled: true, CarriesAPayload: false });
        Assert.Contains(notebook.Scaffold.Cells[1].Outputs, output => output.Content.Contains("Show the data here", StringComparison.Ordinal));
    }

    [Fact]
    public async Task ApplyingTheList_MakesItsChangesInOneCommit_AndSavesThemBesideTheNotebook_Once()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var decided = new PipelineDeclaration(new PipelineDeclaration(Blocks(notebook).Excluding("pclass")).WithKind("survived", ColumnKind.Category));

        Saved(decided.Steps);
        await ListAsync(notebook);
        var action = ApplyBox(notebook)!.Value.Action;

        var applied = await ApplyAsync(notebook, action, ticked: true);

        Assert.True(applied.StateChanged);
        Assert.Equal(decided, Blocks(notebook));
        Assert.Equal(PipelinePreset.Of(decided, header: null), PipelinePreset.FromJson(await File.ReadAllTextAsync(ColumnsFile, TestContext.Current.CancellationToken), NotebookVerbs.Catalog()));
        Assert.True(notebook.Scaffold.Cells[1].Outputs[^1].Content.Heads("survived"));

        // The router sends the box's state again on the click's echo: the blocks hold the take-over already.
        var echo = await ApplyAsync(notebook, action, ticked: true);

        Assert.False(echo.StateChanged);
        Assert.Equal(decided, Blocks(notebook));
        Assert.DoesNotContain(notebook.Scaffold.Cells[1].Outputs, output => output.IsError);
    }

    [Fact]
    public async Task TheListsBoxUnticked_OrTabbedPast_ChangesNothing()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var before = Blocks(notebook);

        Saved(before.Excluding("pclass"));
        await ListAsync(notebook);

        var untouched = await ApplyAsync(notebook, ApplyBox(notebook)!.Value.Action, ticked: false);

        Assert.False(untouched.StateChanged);
        Assert.Equal(before, Blocks(notebook));
    }

    [Fact]
    public async Task ApplyingAfterTheBlocksChanged_IsNotMade_AndSaysToTakeOverAgain()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var before = Blocks(notebook);

        Saved(before.Excluding("pclass"));
        await ListAsync(notebook);
        var action = ApplyBox(notebook)!.Value.Action;

        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Category, "survived", ticked: true);
        var changed = Blocks(notebook);

        var refused = await ApplyAsync(notebook, action, ticked: true);

        Assert.False(refused.StateChanged);
        Assert.Equal(changed, Blocks(notebook));
        Assert.Contains(notebook.Scaffold.Cells[1].Outputs, output => output.IsError
            && WebUtility.HtmlDecode(output.Content).Contains("the blocks changed since the list was shown — take over again", StringComparison.Ordinal));
    }

    [Fact]
    public async Task ApplyingWhileTheBlocksMakeNoPipeline_IsNotMade_AndSaysWhichBlockStopsThem()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var before = Blocks(notebook);

        Saved(before.Excluding("pclass"));
        await ListAsync(notebook);
        var action = ApplyBox(notebook)!.Value.Action;

        notebook.AddBlock("""{"step": "normalise", "column": "colour", "scale": "standard", "outOfRange": "pass"}""");

        var refused = await ApplyAsync(notebook, action, ticked: true);

        Assert.False(refused.StateChanged);
        Assert.Equal(before.Steps, Steps(notebook).Take(5));
        Assert.Contains(notebook.Scaffold.Cells[1].Outputs, output => output.IsError
            && WebUtility.HtmlDecode(output.Content).Contains("the blocks do not make a pipeline yet: block 6", StringComparison.Ordinal));
    }

    [Fact]
    public async Task ATakeOverWhoseBlocksWouldBreakARule_IsListedWithEveryFault_AndOffersNothingToApply()
    {
        // The saved schema has no age, and the fill reads it.
        await using var notebook = await NotebookAsync(Titanic);

        File.WriteAllText(ColumnsFile, PipelinePreset.Of(
            Pdd.Create().ReadCsv("titanic.csv").Declare(columns => columns.Integer("survived", "pclass").Number("fare")).Declaration, header: null).ToJson());
        await ListAsync(notebook);

        var card = Card(notebook);

        Assert.Contains("cannot be taken over", card, StringComparison.Ordinal);
        Assert.Contains("'age'", card, StringComparison.Ordinal);
        Assert.Null(ApplyBox(notebook));
    }

    [Fact]
    public async Task SavedColumnsTheBlocksAlreadyHold_ListNothingToApply_ButTheSourcesNewColumns()
    {
        await using var notebook = await NotebookAsync(Titanic);
        string[] shown = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "class", "who", "adult_male", "deck", "embark_town", "alive"];

        Saved(Blocks(notebook).Steps, shown);
        await ListAsync(notebook);

        var card = Card(notebook);

        Assert.Contains("The blocks already hold every column decision saved beside the notebook.", card, StringComparison.Ordinal);
        Assert.Contains("New in the source: alone", card, StringComparison.Ordinal);
        Assert.Null(ApplyBox(notebook));
    }

    [Fact]
    public async Task AColumnAStepMakes_IsNamedByThatStep_AndOneTheSourceLacksIsSaidToBeMissingFromIt()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var decided = new PipelineDeclaration(Blocks(notebook).Excluding("age_was_missing"));

        Saved(new PipelineDeclaration(decided.Including("colour", ColumnKind.Text, [])).Steps);
        await ListAsync(notebook);

        var card = Card(notebook);

        Assert.Contains("<code>age_was_missing</code>: made by step 4, 'fill.missing' → dropped", card, StringComparison.Ordinal);
        Assert.Contains("<code>colour</code>: not in the schema → taken, text — not in the source", card, StringComparison.Ordinal);
        Assert.DoesNotContain("<code>age_was_missing</code>: made by step 4, 'fill.missing' → dropped — not in the source", card, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheOutputAndTheSchemasOrder_AreListedAsTheyStandAndAsTheyWouldStand()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var reordered = Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(columns => columns.Integer("survived").Number("fare").Integer("pclass").Optional("age", ColumnKind.Number))
            .SplitStratified("survived", 0.70, 0.15, seed: 20260923)
            .FillMissing("age", With.Median)
            .Normalise("fare")
            .Target("survived")
            .Declaration;

        Saved(reordered.Steps);
        await ListAsync(notebook);

        var card = Card(notebook);

        Assert.Contains("the output: none → <code>{\"step\":\"target\",\"column\":\"survived\"}</code>", card, StringComparison.Ordinal);
        Assert.Contains("the schema's order: survived, pclass, age, fare → survived, fare, pclass, age", card, StringComparison.Ordinal);
    }

    [Fact]
    public async Task SavedColumnsThatCannotBeRead_AreSaidToBeUnreadableAtTheSchema_AndOfferNothingToApply()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await File.WriteAllTextAsync(ColumnsFile, "not the saved columns", TestContext.Current.CancellationToken);
        await ListAsync(notebook);

        Assert.Contains("The columns saved beside the notebook cannot be read, so there is nothing to take over:", Card(notebook), StringComparison.Ordinal);
        Assert.True(notebook.Scaffold.Cells[1].Outputs[^1].IsError);
        Assert.Null(ApplyBox(notebook));
    }

    [Fact]
    public async Task SavedColumnsAnotherProgramHoldsOpen_AreSaidToBeUnreadable_AndOfferNothingToApply()
    {
        await using var notebook = await NotebookAsync(Titanic);

        Saved(Blocks(notebook).Excluding("pclass"));

        await using (new FileStream(ColumnsFile, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
        {
            await ListAsync(notebook);
        }

        Assert.Contains("The columns saved beside the notebook cannot be read, so there is nothing to take over:", Card(notebook), StringComparison.Ordinal);
        Assert.Null(ApplyBox(notebook));
    }

    [Fact]
    public async Task ASourceThatCannotBeRead_OrRowsHandedIn_ListWithoutSayingWhatTheSourceHas()
    {
        // Without the source's first line nothing is known of its columns: none is new, and none is missing from it.
        await using var gone = await NotebookAsync(Titanic);
        await using var handedIn = await NotebookAsync("""{"step": "read.rows", "description": "passengers"}""", Titanic[1]);

        Saved(new PipelineDeclaration(Blocks(gone).Excluding("pclass")).Including("colour", ColumnKind.Text, []));
        File.Delete(Path.Join(_folder, "titanic.csv"));
        await ListAsync(gone);
        var withoutFile = Card(gone);

        Saved(Blocks(handedIn).Excluding("pclass"));
        await ListAsync(handedIn);
        var withoutHeader = Card(handedIn);

        Assert.Contains("<code>colour</code>: not in the schema → taken, text</li>", withoutFile, StringComparison.Ordinal);
        Assert.Contains("<code>pclass</code>: taken, integer → excluded, integer</li>", withoutHeader, StringComparison.Ordinal);
        Assert.All(new[] { withoutFile, withoutHeader }, card =>
        {
            Assert.DoesNotContain("not in the source", card, StringComparison.Ordinal);
            Assert.DoesNotContain("New in the source", card, StringComparison.Ordinal);
        });
    }

    [Fact]
    public async Task BlocksWithoutASchema_AreListedAtTheirSource_AndTakeTheSavedSchemaDirectlyAfterIt()
    {
        await using var notebook = await NotebookAsync(Titanic[0]);
        var decided = new PipelineDeclaration([NotebookVerbs.Catalog().ReadStep(Titanic[0]), NotebookVerbs.Catalog().ReadStep(Titanic[1])]);

        Saved(decided.Steps);
        await ListAsync(notebook);

        var card = WebUtility.HtmlDecode(notebook.Scaffold.Cells[0].Outputs[^1].Content);
        var action = notebook.Scaffold.Cells[0].Outputs[^1].Content.Boxes().Single().Action;

        Assert.Contains("<code>survived</code>: not in the schema → taken, integer", card, StringComparison.Ordinal);

        var applied = await notebook.GestureAsync(notebook.Scaffold.Cells[0], action, "true");

        Assert.True(applied.StateChanged);
        Assert.Equal(decided, Blocks(notebook));
    }
}
