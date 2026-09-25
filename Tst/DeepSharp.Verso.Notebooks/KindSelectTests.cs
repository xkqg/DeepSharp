// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A row's kind select commits the kind it ends on, as one pick of that kind would from the state the list was drawn in —
/// so a category walked away from and back to still remembers the kind it was. Verso's router sends a select's value on
/// every key and on every change, with no end to a walk; each value picks again from the drawn state and replaces what
/// the select wrote before. A send made stale by anything else — the grid, another select, another session — draws the
/// list again and changes nothing, an echo included. A walk is sent here as the router sends it, every value to the
/// block the list was drawn on, which the first change writes anew.
/// </summary>
public sealed class KindSelectTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-kinds-").FullName;

    private const string Read = """{"step": "read.csv", "path": "titanic.csv"}""";

    private const string Below =
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""";

    private const string Fill = """{"step": "fill.missing", "column": "age", "with": "median"}""";

    private const string Scale = """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""";

    public KindSelectTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    // The schema, with pclass written as given.
    private static string Schema(string pclass, string remainder = "drop") =>
        $$"""{"step": "declare", "remainder": "{{remainder}}", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {{pclass}}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""";

    private static readonly string Integer = Schema("""{"name": "pclass", "kind": "integer", "optional": false}""");

    private static readonly string CategoryThatWasInteger = Schema("""{"name": "pclass", "kind": "category", "optional": false, "was": "integer"}""");

    private static readonly string CategoryThatSaysNothing = Schema("""{"name": "pclass", "kind": "category", "optional": false}""");

    private async Task<Notebook> NotebookAsync(params string[] blocks)
    {
        var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private Task<Notebook> TitanicAsync(string schema) => NotebookAsync(Read, schema, Below, Fill, Scale);

    private static CellModel SchemaBlock(Notebook notebook) => notebook.Scaffold.Cells[1];

    private static string List(Notebook notebook) =>
        SchemaBlock(notebook).Outputs.Single(output => output.Content.Contains("<tr data-column=", StringComparison.Ordinal)).Content;

    private static ColumnDeclaration Column(Notebook notebook, string name) =>
        NotebookVerbs.Catalog().ReadStep(SchemaBlock(notebook).Source) is DeclareStep declare
            ? declare.Columns.Single(column => column.Name == name)
            : throw new InvalidOperationException("The second block is not the schema.");

    private static bool Declares(Notebook notebook, string name) =>
        ((DeclareStep)NotebookVerbs.Catalog().ReadStep(SchemaBlock(notebook).Source)).Columns.Any(column => column.Name == name);

    // Lists the columns, and hands back the block the list is drawn on and a row's kind select.
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

    [Fact]
    public async Task AWalkFromIntegerThroughTextToCategory_EndsACategoryThatWasInteger_InTwoCommits()
    {
        await using var notebook = await TitanicAsync(Integer);
        var list = await ChooseAsync(notebook);

        var commits = await WalkAsync(notebook, list.On, list.Html.Row("pclass").Kind.Action, "integer", "text", "text", "category", "category");

        Assert.Equal(2, commits);
        Assert.DoesNotContain(notebook.Scaffold.Cells, cell => cell.Id == list.On.Id);
        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Category, Optional: false) { Was = ColumnKind.Integer }, Column(notebook, "pclass"));
    }

    [Fact]
    public async Task AWalkFromACategoryThatWasIntegerAndBack_EndsAsItWasDrawn()
    {
        await using var notebook = await TitanicAsync(CategoryThatWasInteger);
        var list = await ChooseAsync(notebook);

        var commits = await WalkAsync(notebook, list.On, list.Html.Row("pclass").Kind.Action, "category", "text", "text", "category", "category");

        Assert.Equal(2, commits);
        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Category, Optional: false) { Was = ColumnKind.Integer }, Column(notebook, "pclass"));
    }

    [Theory]
    [MemberData(nameof(WalksAwayFromACategory))]
    public async Task AWalkAwayFromACategory_EndsAsOnePickOfTheLastKindFromTheDrawnState(string schema, string[] values, ColumnKind kind, int expected)
    {
        await using var notebook = await TitanicAsync(schema);
        var list = await ChooseAsync(notebook);

        var commits = await WalkAsync(notebook, list.On, list.Html.Row("pclass").Kind.Action, values);

        Assert.Equal(expected, commits);
        Assert.Equal(new ColumnDeclaration("pclass", kind, Optional: false), Column(notebook, "pclass"));
    }

    public static TheoryData<string, string[], ColumnKind, int> WalksAwayFromACategory() => new()
    {
        // The way back up to integer, and a walk to boolean, from a category that says what it was and from one that does not.
        { CategoryThatWasInteger, ["category", "timestamp", "timestamp", "boolean", "boolean", "integer", "integer"], ColumnKind.Integer, 3 },
        { CategoryThatWasInteger, ["category", "timestamp", "timestamp", "boolean", "boolean"], ColumnKind.Boolean, 2 },
        { CategoryThatSaysNothing, ["category", "timestamp", "timestamp", "boolean", "boolean"], ColumnKind.Boolean, 2 },
        { CategoryThatSaysNothing, ["category", "timestamp", "timestamp", "boolean", "boolean", "integer", "integer"], ColumnKind.Integer, 3 },
    };

    [Fact]
    public async Task AWalkThroughEveryKindAboveIt_EndsACategoryThatWasInteger()
    {
        await using var notebook = await TitanicAsync(Integer);
        var list = await ChooseAsync(notebook);

        var commits = await WalkAsync(notebook, list.On, list.Html.Row("pclass").Kind.Action, "integer", "number", "number", "text", "text", "category", "category");

        Assert.Equal(3, commits);
        Assert.Equal(ColumnKind.Integer, Column(notebook, "pclass").Was);
    }

    [Fact]
    public async Task ACategoryThatSaysNothing_WalkedAwayAndBack_IsStillOne()
    {
        await using var notebook = await TitanicAsync(CategoryThatSaysNothing);
        var list = await ChooseAsync(notebook);

        await WalkAsync(notebook, list.On, list.Html.Row("pclass").Kind.Action, "category", "text", "text", "category");

        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Category, Optional: false), Column(notebook, "pclass"));
    }

    [Fact]
    public async Task AnEchoOrATab_OnACardThatStillHolds_ChangesNothing()
    {
        await using var notebook = await TitanicAsync(Integer);
        var list = await ChooseAsync(notebook);

        var commits = await WalkAsync(notebook, list.On, list.Html.Row("pclass").Kind.Action, "integer", "integer");

        Assert.Equal(0, commits);
        Assert.Same(list.On, SchemaBlock(notebook));
    }

    [Fact]
    public async Task ASendTheGridMadeStale_DrawsTheListAgain_AndCommitsNothing_NotEvenItsEcho()
    {
        await using var notebook = await TitanicAsync(Integer);
        var list = await ChooseAsync(notebook);
        var select = list.Html.Row("pclass").Kind.Action;

        await notebook.TickAsync(SchemaBlock(notebook), StepRenderer.Category, "pclass", ticked: true);

        var commits = await WalkAsync(notebook, list.On, select, "integer", "text", "text");

        Assert.Equal(0, commits);
        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Category, Optional: false) { Was = ColumnKind.Integer }, Column(notebook, "pclass"));
        Assert.Equal("category", List(notebook).Row("pclass").Kind.Value);
    }

    [Fact]
    public async Task ARedrawBetweenKeysEndsTheWalk_AndTheSelectItDrawsIsANewOne()
    {
        await using var notebook = await TitanicAsync(Integer);
        var list = await ChooseAsync(notebook);

        // Home lands and commits; the redraw takes the focus, so End sends nothing on the old select.
        Assert.Equal(1, await WalkAsync(notebook, list.On, list.Html.Row("pclass").Kind.Action, "integer", "text"));
        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Text, Optional: false), Column(notebook, "pclass"));

        // Focused again, the redrawn select starts from text.
        var redrawn = List(notebook).Row("pclass").Kind.Action;

        Assert.Equal(1, await WalkAsync(notebook, SchemaBlock(notebook), redrawn, "text", "category"));
        Assert.Equal(ColumnKind.Text, Column(notebook, "pclass").Was);
    }

    [Fact]
    public async Task AWalkResumedAfterARedraw_RemembersTheKindTheNewSelectWasDrawnAt()
    {
        await using var notebook = await TitanicAsync(Integer);
        var list = await ChooseAsync(notebook);

        await WalkAsync(notebook, list.On, list.Html.Row("pclass").Kind.Action, "integer", "number");
        await WalkAsync(notebook, SchemaBlock(notebook), List(notebook).Row("pclass").Kind.Action, "number", "category");

        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Category, Optional: false) { Was = ColumnKind.Number }, Column(notebook, "pclass"));
    }

    [Fact]
    public async Task AnotherSelectsChangeBetweenTwoSendsOfAWalk_MakesTheLaterSendDrawTheListAgain()
    {
        await using var notebook = await TitanicAsync(Integer);
        var list = await ChooseAsync(notebook);
        var pclass = list.Html.Row("pclass").Kind.Action;

        await WalkAsync(notebook, list.On, pclass, "integer", "text");
        await WalkAsync(notebook, SchemaBlock(notebook), List(notebook).Row("sex").Kind.Action, "category");

        var commits = await WalkAsync(notebook, list.On, pclass, "category");

        Assert.Equal(0, commits);
        Assert.Equal(ColumnKind.Text, Column(notebook, "pclass").Kind);
        Assert.Equal(ColumnKind.Category, Column(notebook, "sex").Kind);
    }

    [Fact]
    public async Task ASendReachingAnotherSessionMidWalk_CommitsNothing()
    {
        await using var notebook = await TitanicAsync(Integer);
        var list = await ChooseAsync(notebook);
        var pclass = list.Html.Row("pclass").Kind.Action;

        await WalkAsync(notebook, list.On, pclass, "integer", "text");

        await using var reopened = await NotebookAsync([.. notebook.Scaffold.Cells.Select(cell => cell.Source)]);

        var commits = await WalkAsync(reopened, SchemaBlock(reopened), pclass, "category");

        Assert.Equal(0, commits);
        Assert.Equal(ColumnKind.Text, Column(reopened, "pclass").Kind);
        Assert.Contains(SchemaBlock(reopened).Outputs, output => output.IsError
            && WebUtility.HtmlDecode(output.Content).Contains("Choose the columns again", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AWalkOverEveryKindTheSelectOffers_IsNeverRefused()
    {
        await using var notebook = await TitanicAsync(Integer);
        var list = await ChooseAsync(notebook);
        var select = list.Html.Row("pclass").Kind;

        await WalkAsync(notebook, list.On, select.Action, [.. select.Options]);

        Assert.DoesNotContain(SchemaBlock(notebook).Outputs, output => output.IsError);
        Assert.Equal(new ColumnDeclaration("pclass", ColumnKind.Category, Optional: false) { Was = ColumnKind.Integer }, Column(notebook, "pclass"));
    }

    [Fact]
    public async Task AnUndeclaredRow_IsTakenInByAKind_AndLeftAsItWasDrawnWhenTheWalkEndsOnNone()
    {
        await using var notebook = await TitanicAsync(Integer);
        var list = await ChooseAsync(notebook);
        var sex = list.Html.Row("sex").Kind.Action;

        Assert.Equal(0, await WalkAsync(notebook, list.On, sex, string.Empty));
        Assert.Equal(1, await WalkAsync(notebook, list.On, sex, "category"));
        Assert.Equal(ColumnKind.Category, Column(notebook, "sex").Kind);

        // The same walk goes on to none: one pick of none from the state it was drawn in.
        Assert.Equal(1, await WalkAsync(notebook, list.On, sex, "category", string.Empty));
        Assert.False(Declares(notebook, "sex"));
    }

    [Fact]
    public async Task ARowsSelect_OffersTheKindsTheRulesKeep_ItsOwnSelected()
    {
        await using var notebook = await TitanicAsync(Integer);
        await using var keeping = await NotebookAsync(Read, Schema("""{"name": "pclass", "kind": "integer", "optional": false}""", "keep"), Below, Fill, Scale);

        var list = (await ChooseAsync(notebook)).Html;
        var kept = (await ChooseAsync(keeping)).Html;

        Assert.True(list.Row("pclass").Kind is { Value: "integer", Enabled: true });
        Assert.Equal(["text", "number", "integer", "boolean", "timestamp", "category"], list.Row("pclass").Kind.Options);
        Assert.Equal(["number", "integer", "boolean"], list.Row("fare").Kind.Options);
        Assert.Equal([string.Empty, "text", "number", "integer", "boolean", "timestamp", "category"], list.Row("sex").Kind.Options);
        Assert.True(kept.Row("sex").Kind is { Value: "", Enabled: false });
    }
}
