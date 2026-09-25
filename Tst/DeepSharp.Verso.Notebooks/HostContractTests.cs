// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using DeepSharp.Verso.Notebooks;
using Verso;
using Verso.Abstractions;
using Verso.Extensions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// What the rest of the suite cannot see, because it enters below Verso's host: how a front end reads a control, and
/// what it does with an answer. A click reaches a part through Verso's router, which sends what a control carries in
/// its <c>data-payload</c>, and a box's own state when it carries none; and a host that is answered replaces the
/// block's outputs with the answer — so a gesture that shows or changes something answers nothing at all.
/// </summary>
public sealed partial class HostContractTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-host-").FullName;

    // Where a package is installed: beside the suite, each run laying it out over the last. A package loaded into a
    // context of its own keeps its files locked until the process ends, so no test can remove it after itself.
    private static readonly string Installs = Path.Join(AppContext.BaseDirectory, "installed");

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public HostContractTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private async Task<Notebook> NotebookAsync()
    {
        var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    [GeneratedRegex("<button[^>]*>")]
    private static partial Regex Buttons();

    [GeneratedRegex("<input[^>]*>")]
    private static partial Regex Inputs();

    [Fact]
    public async Task EveryButtonWhoseGestureTakesAValue_CarriesItWhereVersosRouterReadsIt()
    {
        await using var notebook = await NotebookAsync();
        var declare = notebook.Scaffold.Cells[1];

        await notebook.GestureAsync(declare, StepRenderer.Show);

        var pages = declare.Outputs.SelectMany(output => Buttons().Matches(output.Content).Select(match => match.Value))
            .Where(button => button.Contains($"data-action=\"{StepRenderer.Page}\"", StringComparison.Ordinal))
            .ToArray();

        Assert.NotEmpty(pages);
        Assert.All(pages, button => Assert.Contains("data-payload=\"", button, StringComparison.Ordinal));
        Assert.Contains(pages, button => button.Contains("data-payload=\"1\"", StringComparison.Ordinal));
    }

    [Fact]
    public async Task EveryBox_CarriesItsGestureAndItsColumnInItsAction_AndNoPayload_SoTheRouterSendsTheStateItIsIn()
    {
        // The router sends a control's data-payload when it has one, and a box's state only when it has none.
        await using var notebook = await NotebookAsync();
        var declare = notebook.Scaffold.Cells[1];

        await notebook.GestureAsync(declare, StepRenderer.Show);

        var inputs = declare.Outputs.SelectMany(output => Inputs().Matches(output.Content).Select(match => match.Value)).ToArray();
        var grid = declare.Outputs[1].Content;

        Assert.NotEmpty(inputs);
        Assert.All(inputs, input =>
        {
            Assert.Contains("type=\"checkbox\"", input, StringComparison.Ordinal);
            Assert.Contains($"data-extension-id=\"{StepRenderer.Id}\"", input, StringComparison.Ordinal);
            Assert.DoesNotContain("data-payload", input, StringComparison.Ordinal);
        });
        Assert.All(grid.Boxes(), box => Assert.True(
            box.Action.StartsWith($"{StepRenderer.Include} ", StringComparison.Ordinal) || box.Action.StartsWith($"{StepRenderer.Category} ", StringComparison.Ordinal)));
        Assert.Equal(Notebook.BoxAction(StepRenderer.Include, "pclass"), grid.Box(StepRenderer.Include, "pclass")!.Value.Action);
        Assert.Equal(Notebook.BoxAction(StepRenderer.Category, "pclass"), grid.Box(StepRenderer.Category, "pclass")!.Value.Action);
    }

    // The package as an install lays it out: the notebook's assembly and the ones it brings, in one folder.
    private static string Installed(string folder)
    {
        Directory.CreateDirectory(folder);

        foreach (var file in Directory.GetFiles(AppContext.BaseDirectory, "*.dll")
                     .Where(file => Path.GetFileName(file) is var name
                                    && (name.StartsWith("MatPlotLibNet", StringComparison.Ordinal)
                                        || (name.StartsWith("DeepSharp.", StringComparison.Ordinal) && !name.Contains("Tests", StringComparison.Ordinal)))))
        {
            File.Copy(file, Path.Join(folder, Path.GetFileName(file)), overwrite: true);
        }

        return Path.Join(folder, "DeepSharp.Verso.Notebooks.dll");
    }

    [Fact]
    public async Task ThePackage_LoadedTheWayVersoLoadsAThirdPartyPackage_ShowsTheDataAtABlock()
    {
        // An installed package is loaded into a context of its own, where its types are not this suite's types: the
        // notebook is driven through Verso's own contracts alone, as the host drives it.
        await using var host = new ExtensionHost();

        await host.LoadFromAssemblyAsync(Installed(Path.Join(Installs, "package")));

        var scaffold = new Scaffold(new NotebookModel(), host, Path.Join(_folder, "titanic.verso"));
        scaffold.InitializeSubsystems();

        try
        {
            foreach (var block in Titanic)
            {
                scaffold.AddCell("deepsharp.step", source: block);
            }

            var declare = scaffold.Cells[1];
            var handler = host.GetInteractionHandler("io.github.xkqg.deepsharp.notebooks.renderer")!;

            Assert.NotEqual(typeof(StepRenderer), handler.GetType());

            await handler.OnCellInteractionAsync(new CellInteractionContext
            {
                CellId = declare.Id,
                ExtensionId = "io.github.xkqg.deepsharp.notebooks.renderer",
                InteractionType = "deepsharp.show",
                Payload = string.Empty,
                Region = CellRegion.Output,
                Variables = scaffold.Variables,
                Notebook = scaffold.NotebookOps,
                NotebookModel = scaffold.Notebook,
            });

            Assert.True(declare.Outputs[^1].Content.Heads("fare"));
        }
        finally
        {
            await scaffold.DisposeAsync();
        }
    }

    [Fact]
    public async Task AnInstallTheScanFindsBeforeANotebookOpens_IsTheOnlyOneThatGuardsTheOpening()
    {
        // Opening a Jupyter file, Verso scans the top of its extensions folder, then reads and post-processes the
        // file, then loads the packages the file asks for — and a Jupyter file asks for none. So the guard runs on
        // the way in only for a package at the top of that folder; one installed in its own folder is loaded too
        // late. That is written down as a limit; this pins the order, so a Verso that changes it is noticed.
        var top = Path.Join(Installs, "top");
        var managed = Path.Join(Installs, "managed");
        var version = typeof(StepRenderer).Assembly.GetName().Version!.ToString(3);

        // The panel keeps one install per runtime, in a folder named after the runtime it was made for.
        var runtime = $"net{Environment.Version.Major}.0";

        Installed(top);
        Installed(Path.Join(managed, "DeepSharp.Verso.Notebooks", version, runtime));

        await using var scanningTop = new ExtensionHost();
        await using var scanningManaged = new ExtensionHost();

        await scanningTop.LoadFromDirectoryAsync(top);
        await scanningManaged.LoadFromDirectoryAsync(managed);

        Assert.Contains(scanningTop.GetPostProcessors(), processor => processor.ExtensionId == JupyterGuard.Id);
        Assert.DoesNotContain(scanningManaged.GetPostProcessors(), processor => processor.ExtensionId == JupyterGuard.Id);
    }

    [Fact]
    public async Task APartAHostLoadedWithoutTheBlockType_RefusesToAct_SayingWhy()
    {
        // A host can load a part on its own. The notebook's session is kept by the block type, so a part loaded
        // without it has no notebook to act in — and says so in the words every part uses.
        await using var host = new ExtensionHost();
        var renderer = new StepRenderer();
        var kernel = new StepKernel();

        await host.LoadExtensionAsync(renderer);
        await host.LoadExtensionAsync(kernel);

        await using var notebook = await NotebookAsync();
        var declare = notebook.Scaffold.Cells[1];

        var answer = await renderer.OnCellInteractionAsync(notebook.Gesture(declare, StepRenderer.Show, string.Empty));
        var refused = Assert.Throws<InvalidOperationException>(() => kernel.Session);

        Assert.Contains("beside the notebook's blocks", answer, StringComparison.Ordinal);
        Assert.Contains("beside the notebook's blocks", refused.Message, StringComparison.Ordinal);
        Assert.Empty(declare.Outputs);

        // Nor does it know a column to offer.
        const string column = "{\"step\": \"normalise\", \"column\": \"";

        Assert.Empty(await kernel.GetCompletionsAsync(column, column.Length));
    }

    [Theory]
    [InlineData(StepRenderer.Show, "")]
    [InlineData(StepRenderer.Page, "1")]
    [InlineData(StepRenderer.Include + " {\"column\":\"pclass\"}", "false")]
    [InlineData(StepRenderer.Include + " {\"column\":\"pclass\"}", "true")]
    [InlineData(StepRenderer.Category + " {\"column\":\"pclass\"}", "true")]
    [InlineData(StepRenderer.Category + " {\"column\":\"pclass\"}", "false")]
    [InlineData(StepRenderer.Include + " {\"column\":\"sex\"}", "true")]
    [InlineData(StepRenderer.Apply + " {\"preset\":\"not saved columns\",\"drawn\":\"0\"}", "true")]
    [InlineData(StepRenderer.Apply + " {\"preset\":\"not saved columns\",\"drawn\":\"0\"}", "false")]
    [InlineData("deepsharp.unknown", "")]
    public async Task EveryGesture_AnswersNothing_SoAHostThatAppliesAnswersLeavesTheBlockAsTheRunWroteIt(string interaction, string payload)
    {
        await using var notebook = await NotebookAsync();

        var answer = await notebook.Handler.OnCellInteractionAsync(notebook.Gesture(notebook.Scaffold.Cells[1], interaction, payload));

        Assert.Null(answer);

        // The same gesture again asks for what already is, and answers nothing either.
        Assert.Null(await notebook.Handler.OnCellInteractionAsync(notebook.Gesture(notebook.Scaffold.Cells[1], interaction, payload)));
    }
}
