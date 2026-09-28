// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A notebook is shown in a layout and a theme the engine has, as Verso's View panel lists and switches them: by id, in
/// the notebook's turn, each switch a version every view is told, and saved as the notebook's own choice. A theme is the
/// notebook's only once chosen: until then no theme of the engine's is the notebook's, and a view draws in its own look.
/// </summary>
public sealed class ViewTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-view-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private async Task<string> WriteAsync(string name, NotebookModel notebook)
    {
        notebook.Cells.Add(new CellModel { Type = StepCellType.StepType, Language = StepKernel.Language, Source = """{"step": "read.csv", "path": "titanic.csv"}""" });

        var path = Path.Join(_folder, name);

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return path;
    }

    private static async Task<NotebookModel> ReadAsync(string path) =>
        await new VersoSerializer().DeserializeAsync(await File.ReadAllTextAsync(path, TestContext.Current.CancellationToken));

    [Fact]
    public async Task TheEnginesLayoutsAndThemes_AreListed_AndSwitchingByIdIsAVersion_SavedAsTheNotebooksChoice()
    {
        var path = await WriteAsync("view.verso", new NotebookModel());

        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);

        Assert.Equal(["dashboard", "notebook", "presentation"], host.Layouts.Select(layout => layout.Id).Order());
        Assert.Equal("Dashboard", host.Layouts.Single(layout => layout.Id == "dashboard").Name);
        Assert.False(host.Layouts.Single(layout => layout.Id == "dashboard").Allows.HasFlag(LayoutAllows.CellEdit));
        Assert.True(host.Layouts.Single(layout => layout.Id == "notebook").Allows.HasFlag(LayoutAllows.CellEdit));

        var dark = host.Themes.Single(theme => theme.Id == "verso-dark");

        Assert.Equal(["verso-dark", "verso-highcontrast", "verso-light"], host.Themes.Select(theme => theme.Id).Order());
        Assert.Equal("Verso Dark", dark.Name);
        Assert.Equal(ThemeTone.Dark, dark.Tone);
        Assert.Contains("--verso-bg-default: #1E1E1E;", dark.Css, StringComparison.Ordinal);

        // Nothing chosen yet: no theme is the notebook's.
        Assert.Null(host.Current.ThemeId);

        using var view = host.Subscribe();

        await host.SwitchLayoutAsync("dashboard");
        Assert.True(view.TryRead(out var switched));
        Assert.Equal("dashboard", switched.Layout.Id);

        await host.SwitchThemeAsync("verso-dark");
        Assert.True(view.TryRead(out var themed));
        Assert.Equal("verso-dark", themed.ThemeId);

        // Another theme, with nothing else to tell: the switch is told all the same.
        await host.SwitchThemeAsync("verso-highcontrast");
        Assert.True(view.TryRead(out var rethemed));
        Assert.Equal("verso-highcontrast", rethemed.ThemeId);

        await host.SwitchThemeAsync("verso-dark");
        Assert.Equal("verso-dark", host.Current.ThemeId);

        await host.SaveAsync();

        var saved = await ReadAsync(path);

        Assert.Equal("dashboard", saved.ActiveLayout?.LayoutId);
        Assert.Equal("verso-dark", saved.PreferredThemeId);
    }

    [Fact]
    public async Task ALayoutOrThemeTheEngineLacks_IsRefused_AndNothingChanges()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(await WriteAsync("view.verso", new NotebookModel()), TestContext.Current.CancellationToken);
        var before = host.Current;

        await Assert.ThrowsAsync<InvalidOperationException>(() => host.SwitchLayoutAsync("no-such-layout"));
        await Assert.ThrowsAsync<InvalidOperationException>(() => host.SwitchThemeAsync("no-such-theme"));

        Assert.Equal(before, host.Current);
        Assert.False(host.Current.Unsaved);
    }

    [Fact]
    public async Task ANotebookThatChoseATheme_OpensInIt_AndOneThatChoseNone_InNone()
    {
        await using var notebooks = new OpenNotebooks();
        var chose = await notebooks.OpenAsync(await WriteAsync("chose.verso", new NotebookModel { PreferredThemeId = "verso-highcontrast" }), TestContext.Current.CancellationToken);
        var none = await notebooks.OpenAsync(await WriteAsync("none.verso", new NotebookModel()), TestContext.Current.CancellationToken);

        Assert.Equal("verso-highcontrast", chose.Current.ThemeId);
        Assert.Null(none.Current.ThemeId);
    }
}
