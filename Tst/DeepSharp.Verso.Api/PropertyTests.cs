// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A cell's properties panel, as an application draws it: a section from every part that has one for the cell —
/// DeepSharp's form for a block, and Verso's own for how any cell is shown — each field with its kind, its value and
/// its choices. Changing a field takes its turn like anything else done to the notebook, and is made by the part the
/// section came from.
/// </summary>
public sealed class PropertyTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-properties-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public PropertyTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    private async Task<NotebookHost> OpenAsync(OpenNotebooks notebooks, params CellModel[] cells)
    {
        var notebook = new NotebookModel();

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        var path = Path.Join(_folder, "titanic.verso");

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task ACellsPanel_HasASectionFromEveryPartThatHasOneForIt()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, [.. Titanic.Select(Block)]);

        var sections = await host.PropertiesAsync(host.Cells[4].Id);
        var form = sections.Single(section => section.Part == StepForm.Id);

        // The notebook's own layout has the panel, as Verso's has it.
        Assert.True(host.Current.Layout.HasPropertiesPanel);
        var scale = form.Fields.Single(field => field.Name == "scale");

        Assert.Equal("Pipeline step", form.Title);
        Assert.Equal(FieldKind.Select, scale.Kind);
        Assert.Equal("standard", scale.Value);
        Assert.Contains(scale.Options, option => option.Value == "minmax");
        Assert.Contains(sections, section => section.Part == "verso.propertyprovider.display");
        Assert.Contains(sections, section => section.Part == "verso.propertyprovider.visibility");
    }

    [Fact]
    public async Task ChangingAField_IsMadeByThePartItsSectionCameFrom()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, [.. Titanic.Select(Block)]);
        var normalise = host.Cells[4].Id;

        await host.SetPropertyAsync(normalise, StepForm.Id, "scale", "minmax");
        await host.SetPropertyAsync(normalise, "verso.propertyprovider.display", "outputVisibility", "hidden");

        Assert.Contains("\"minmax\"", host.Cells[4].Source, StringComparison.Ordinal);
        Assert.Equal(
            "hidden",
            (await host.PropertiesAsync(normalise))
                .Single(section => section.Part == "verso.propertyprovider.display").Fields.Single(field => field.Name == "outputVisibility").Value);
    }

    [Fact]
    public async Task APanelOnACellAChangeTookAway_IsRefused_AsIsAPartThatIsNotThere()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, [.. Titanic.Select(Block)]);
        var gone = Guid.NewGuid();

        Assert.Equal(gone, (await Assert.ThrowsAsync<CellGoneException>(() => host.PropertiesAsync(gone))).Cell);
        await Assert.ThrowsAsync<CellGoneException>(() => host.SetPropertyAsync(gone, StepForm.Id, "scale", "minmax"));

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => host.SetPropertyAsync(host.Cells[4].Id, "no.such.part", "scale", "minmax"));

        Assert.Contains("no.such.part", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ANotebookShownInALayoutWithNoPanel_RefusesThePanelAndItsFields_AndItsBlocksKeepTheirText()
    {
        var notebook = new NotebookModel { ActiveLayout = new LayoutReference("verso.layout.dashboard", "dashboard") };

        foreach (var block in Titanic)
        {
            notebook.Cells.Add(Block(block));
        }

        var path = Path.Join(_folder, "dashboard.verso");

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);
        var normalise = host.Cells[4].Id;

        // As Verso's editors offer the panel only in a layout that has one: the dashboard has none.
        Assert.False(host.Current.Layout.HasPropertiesPanel);
        await Assert.ThrowsAsync<InvalidOperationException>(() => host.PropertiesAsync(normalise));
        await Assert.ThrowsAsync<InvalidOperationException>(() => host.SetPropertyAsync(normalise, StepForm.Id, "scale", "minmax"));
        Assert.Equal(Titanic[4], host.Cells[4].Source);
    }

    [Fact]
    public async Task APanelsContext_IsTheCellItIsFor_InTheNotebookItStandsIn()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, [.. Titanic.Select(Block)]);
        var cell = host.Scaffold.Cells[0];
        var context = new RenderContext(host.Scaffold, cell, new ReadPort(host.Scaffold));

        Assert.Equal(cell.Id, context.CellId);
        Assert.Same(cell.Metadata, context.CellMetadata);
        Assert.Equal((0, 0), context.Dimensions);
        Assert.True(context.IsSelected);
        Assert.Same(host.Scaffold.Variables, context.Variables);
        Assert.Equal(CancellationToken.None, context.CancellationToken);
    }

    [Fact]
    public void TwoLooksAtAPanel_AreEqual_WhileTheyHoldTheSame()
    {
        var field = new HostedField("scale", "scale", FieldKind.Select, "standard", "How to scale.", [new HostedOption("minmax", "minmax")], IsReadOnly: false);
        var section = new HostedSection(StepForm.Id, "Pipeline step", null, [field]);
        var tags = field with { Kind = FieldKind.Tags, Value = new[] { "a", "b" } };

        Assert.Equal(section, section with { Fields = [field with { Options = [new HostedOption("minmax", "minmax")] }] });
        Assert.Equal(section.GetHashCode(), (section with { Fields = [field] }).GetHashCode());
        Assert.NotEqual(section, section with { Part = "other" });
        Assert.NotEqual(section, section with { Title = "Other" });
        Assert.NotEqual(section, section with { Description = "Other." });
        Assert.NotEqual(section, section with { Fields = [] });

        Assert.Equal(field.GetHashCode(), (field with { Options = [new HostedOption("minmax", "minmax")] }).GetHashCode());
        Assert.Equal(tags, tags with { Value = new List<string> { "a", "b" } });
        Assert.NotEqual(tags, tags with { Value = new[] { "a", "c" } });
        Assert.NotEqual(field, field with { Name = "other" });
        Assert.NotEqual(field, field with { Label = "other" });
        Assert.NotEqual(field, field with { Kind = FieldKind.Text });
        Assert.NotEqual(field, field with { Value = "minmax" });
        Assert.NotEqual(field, field with { Description = null });
        Assert.NotEqual(field, field with { Options = [] });
        Assert.NotEqual(field, field with { IsReadOnly = true });
    }
}
