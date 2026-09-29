// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Serve;

/// <summary>
/// What the columns' cells propose, and what a profile finds should not be there, reach a page through the server as
/// Verso's editor draws them: the list shows each proposal with its counts and a box that takes the column in so, and an
/// alert's box gives its answer — both clicked through the page's one socket.
/// </summary>
public sealed partial class NotebookEndpointTests
{
    private static DeclareStep Schema(NotebookVersion notebook) =>
        notebook.Cells.Select(cell => StepCatalog.BuiltIn().ReadStep(cell.Source)).OfType<DeclareStep>().Single();

    [Fact]
    public async Task TheListShowsWhatAColumnsCellsPropose_AndItsBoxClickedThroughTheSocketTakesItInSo()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var schema = (await socket.SnapshotAsync()).Version.Cells[1].Id;

        await ClickAsync(socket, schema, "deepsharp.columns", "");

        var listed = (await NotebookAsync(served)).Cells[1];

        Assert.Contains(listed.Outputs, output => output.Content.Contains(
            "<tr data-column=\"embarked\">", StringComparison.Ordinal) && output.Content.Contains("category, proposed, 3 values", StringComparison.Ordinal));

        var box = ActionOf(listed, action => action.StartsWith("deepsharp.list.include ", StringComparison.Ordinal)
                                             && action.Contains("\"column\":\"embarked\"", StringComparison.Ordinal));

        Assert.Contains("\"kind\":\"category\"", box, StringComparison.Ordinal);
        Assert.True((await ClickAsync(socket, schema, box, "true")).StateChanged);
        Assert.Equal(ColumnKind.Category, Schema(await NotebookAsync(served)).Columns.Single(column => column.Name == "embarked").Kind);
    }

    [Fact]
    public async Task AnAlertsBoxClickedThroughTheSocket_LeavesTheColumnThatHandsAModelTheAnswerOut()
    {
        var notebook = new NotebookModel();

        foreach (var step in new[]
        {
            """{"step": "read.csv", "path": "titanic.csv"}""",
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "alive", "kind": "boolean", "optional": false}]}""",
            """{"step": "evidence.profile"}""",
            """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
            """{"step": "target", "column": "survived"}""",
        })
        {
            notebook.Cells.Add(new CellModel { Type = StepCellType.StepType, Language = StepKernel.Language, Source = step });
        }

        await File.WriteAllTextAsync(At("alerts.verso"), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served, "alerts.verso");
        var profile = (await socket.SnapshotAsync()).Version.Cells[2].Id;

        await ClickAsync(socket, profile, "deepsharp.show", "");

        var shown = (await NotebookAsync(served, "alerts.verso")).Cells[2];
        var box = ActionOf(shown, action => action.StartsWith("deepsharp.answer ", StringComparison.Ordinal)
                                            && action.Contains("\"column\":\"alive\"", StringComparison.Ordinal));

        Assert.True((await ClickAsync(socket, profile, box, "true")).StateChanged);
        Assert.True(Schema(await NotebookAsync(served, "alerts.verso")).Columns.Single(column => column.Name == "alive").Excluded);
    }
}
