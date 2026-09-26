// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso;
using Verso.Abstractions;
using Verso.Extensions;
using Verso.Serializers;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A real Verso notebook, in this process, opened the way an application of your own opens one: through
/// <c>DeepSharp.Verso.Api</c>, which registers this package's parts, reads the notebook through its format's serializer,
/// and hands a click, a toolbar button and a properties panel their contexts — one thing at a time, each click settling
/// as long as it does anywhere else.
/// </summary>
/// <remarks>
/// The package is tested inside what runs it rather than against stubs of it, because what matters is how Verso
/// treats a third-party cell type — which kernel runs it, where its output lands, which part an interaction reaches —
/// and a stub would only say what the stub's author believed. Where a test is about a host that orders nothing —
/// VS Code writes a block's text into the notebook whenever it lands — it hands the part Verso's own interaction, as
/// such a host does, through <see cref="Gesture"/> and <see cref="Handler"/>.
/// </remarks>
internal sealed class Notebook : IAsyncDisposable
{
    private readonly OpenNotebooks _notebooks;

    // The folder made for a notebook the test named no file for; nothing else writes there.
    private readonly string? _made;

    private Notebook(OpenNotebooks notebooks, NotebookHost opened, string? made)
    {
        _notebooks = notebooks;
        _made = made;
        Opened = opened;
        Scaffold.OnCellOutputUpdated += Pushed.Add;
    }

    /// <summary>The notebook's host.</summary>
    public NotebookHost Opened { get; }

    /// <summary>Verso's extension host, holding Verso's extensions and this package's.</summary>
    public ExtensionHost Host => Opened.Extensions;

    /// <summary>The notebook itself.</summary>
    public Scaffold Scaffold => Opened.Scaffold;

    /// <summary>Every cell whose output was pushed to the front end while it ran, in the order they were pushed.</summary>
    public List<Guid> Pushed { get; } = [];

    /// <summary>Opens an empty notebook.</summary>
    /// <param name="filePath">Where the notebook is saved; nothing for one that never was.</param>
    /// <returns>The notebook.</returns>
    public static async Task<Notebook> OpenAsync(string? filePath = null)
    {
        var made = filePath is null ? Directory.CreateTempSubdirectory("deepsharp-notebook-").FullName : null;
        var path = filePath ?? Path.Join(made, "untitled.verso");

        if (!File.Exists(path))
        {
            await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(new NotebookModel()));
        }

        var notebooks = new OpenNotebooks();
        var opened = await notebooks.OpenAsync(path);

        // A notebook never saved has no file, as a new one in Verso's editors has none until it is saved.
        if (filePath is null)
        {
            opened.Scaffold.SetFilePath(null);
        }

        return new Notebook(notebooks, opened, made);
    }

    /// <summary>Adds a block holding one step, written as the given text.</summary>
    /// <param name="source">The step's JSON.</param>
    /// <returns>The cell.</returns>
    public CellModel AddBlock(string source) => Scaffold.AddCell(StepCellType.StepType, source: source);

    /// <summary>Runs a cell and hands back what it shows.</summary>
    /// <param name="cell">The cell.</param>
    /// <returns>Its outputs after the run.</returns>
    public async Task<IReadOnlyList<CellOutput>> RunAsync(CellModel cell)
    {
        var ran = await Opened.RunAsync(cell.Id);

        Assert.True(ran.LastStatus == "Success", string.Concat(ran.Outputs.Select(output => output.Content)));

        return [.. cell.Outputs];
    }

    /// <summary>The part an interaction on a block reaches, as Verso finds it.</summary>
    public ICellInteractionHandler Handler => Host.GetInteractionHandler(StepRenderer.Id)!;

    /// <summary>A gesture made on a block, as a host that hands it straight to the part builds it.</summary>
    /// <param name="cell">The block.</param>
    /// <param name="interaction">What the gesture asks for.</param>
    /// <param name="payload">What the gesture carries.</param>
    /// <returns>The interaction, as Verso hands it to the part it reaches.</returns>
    public CellInteractionContext Gesture(CellModel cell, string interaction, string payload = "") => new()
    {
        CellId = cell.Id,
        ExtensionId = StepRenderer.Id,
        InteractionType = interaction,
        Payload = payload,
        Region = CellRegion.Output,
        Variables = Scaffold.Variables,
        Notebook = Scaffold.NotebookOps,
        NotebookModel = Scaffold.Notebook,
    };

    /// <summary>Makes a gesture on a block through the host, and hands back what came of it.</summary>
    /// <param name="cell">The block.</param>
    /// <param name="interaction">What the gesture asks for.</param>
    /// <param name="payload">What the gesture carries.</param>
    /// <returns>Whether it changed the blocks, and what the part answered.</returns>
    public Task<GestureResult> GestureAsync(CellModel cell, string interaction, string payload = "") =>
        Opened.GestureAsync(new HostedGesture(cell.Id, StepRenderer.Id, interaction, payload));

    /// <summary>What a grid's box for a column carries in its <c>data-action</c>, as Verso's router hands it on.</summary>
    /// <param name="gesture">The gesture the box makes.</param>
    /// <param name="column">The column the box is about.</param>
    /// <returns>The gesture's name, a space, and the column as JSON.</returns>
    public static string BoxAction(string gesture, string column) =>
        $"{gesture} {JsonSerializer.Serialize(new Dictionary<string, string> { ["column"] = column })}";

    /// <summary>A grid's box for a column sent the way the router sends one: ticked or not, as it is after the key or click.</summary>
    /// <param name="cell">The block the grid is under.</param>
    /// <param name="gesture">The gesture the box makes.</param>
    /// <param name="column">The column the box is about.</param>
    /// <param name="ticked">Whether the box is ticked.</param>
    /// <returns>What came of it.</returns>
    public Task<GestureResult> TickAsync(CellModel cell, string gesture, string column, bool ticked) =>
        GestureAsync(cell, BoxAction(gesture, column), ticked ? "true" : "false");

    /// <summary>Presses a toolbar button through the host.</summary>
    /// <param name="button">The button's id.</param>
    /// <returns>The file it handed over; nothing when it handed none.</returns>
    public Task<HostedFile?> PressAsync(string button) => Opened.RunToolbarAsync(button);

    /// <summary>Whether a toolbar button can be pressed now, as the host's toolbar says.</summary>
    /// <param name="button">The button's id.</param>
    /// <returns>Whether it can.</returns>
    public async Task<bool> EnabledAsync(string button) => (await Opened.ToolbarAsync()).Single(each => each.Id == button).IsEnabled;

    /// <summary>What the host hands a toolbar button, for a test that hands it to a button itself.</summary>
    /// <returns>The context; the file a button hands over is kept in it.</returns>
    public ToolbarContext ToolbarContext() => new(Scaffold, []);

    /// <summary>What the host hands a part drawing or changing a cell's properties panel.</summary>
    /// <param name="cell">The cell.</param>
    /// <returns>The context.</returns>
    public RenderContext RenderContext(CellModel cell) => new(Scaffold, cell);

    public async ValueTask DisposeAsync()
    {
        await _notebooks.DisposeAsync();

        if (_made is not null)
        {
            Directory.Delete(_made, recursive: true);
        }
    }
}
