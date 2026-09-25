// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Verso.Notebooks;
using Verso;
using Verso.Abstractions;
using Verso.Execution;
using Verso.Extensions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A real Verso notebook, in this process: Verso's own extension host with Verso's extensions and this
/// package's parts loaded the way an installed package is found, and Verso's own scaffold running the cells.
/// </summary>
/// <remarks>
/// The package is tested inside what runs it rather than against stubs of it, because what matters is how
/// Verso treats a third-party cell type — which kernel runs it, where its output lands, which part an
/// interaction reaches — and a stub would only say what the stub's author believed.
/// </remarks>
internal sealed class Notebook : IAsyncDisposable
{
    private Notebook(ExtensionHost host, Scaffold scaffold)
    {
        Host = host;
        Scaffold = scaffold;
        Scaffold.OnCellOutputUpdated += Pushed.Add;
    }

    /// <summary>Verso's extension host, holding Verso's extensions and this package's.</summary>
    public ExtensionHost Host { get; }

    /// <summary>The notebook itself.</summary>
    public Scaffold Scaffold { get; }

    /// <summary>Every cell whose output was pushed to the front end while it ran, in the order they were pushed.</summary>
    public List<Guid> Pushed { get; } = [];

    /// <summary>Opens an empty notebook.</summary>
    /// <param name="filePath">Where the notebook is saved; nothing for one that never was.</param>
    /// <returns>The notebook.</returns>
    public static async Task<Notebook> OpenAsync(string? filePath = null)
    {
        var host = new ExtensionHost();

        // Verso's own discovery: its built-in extensions and every assembly beside it that references the
        // abstractions, which is where this package's parts are found.
        await host.LoadBuiltInExtensionsAsync();

        var scaffold = new Scaffold(new NotebookModel(), host, filePath);
        scaffold.InitializeSubsystems();

        return new Notebook(host, scaffold);
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
        var result = await Scaffold.ExecuteCellAsync(cell.Id);

        Assert.True(result.Status == ExecutionResult.ExecutionStatus.Success, result.Error?.ToString());

        return [.. cell.Outputs];
    }

    /// <summary>The part an interaction on a block reaches, as Verso finds it.</summary>
    public ICellInteractionHandler Handler => Host.GetInteractionHandler(StepRenderer.Id)!;

    /// <summary>A gesture made on a block, as the front end sends it.</summary>
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

    /// <summary>Makes a gesture on a block and hands back what the part it reached answered.</summary>
    /// <param name="cell">The block.</param>
    /// <param name="interaction">What the gesture asks for.</param>
    /// <param name="payload">What the gesture carries.</param>
    /// <returns>The interaction after it was handled, and the answer.</returns>
    public async Task<CellInteractionContext> GestureAsync(CellModel cell, string interaction, string payload = "")
    {
        var gesture = Gesture(cell, interaction, payload);

        await Handler.OnCellInteractionAsync(gesture);

        return gesture;
    }

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
    /// <returns>The interaction after it was handled.</returns>
    public Task<CellInteractionContext> TickAsync(CellModel cell, string gesture, string column, bool ticked) =>
        GestureAsync(cell, BoxAction(gesture, column), ticked ? "true" : "false");

    public async ValueTask DisposeAsync()
    {
        await Scaffold.DisposeAsync();
        await Host.DisposeAsync();
    }
}
