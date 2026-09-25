// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// The button that lists what taking over the columns saved beside the notebook would change, before anything does.
/// </summary>
/// <remarks>
/// The first of two presses. This one reads the saved file and the source's header, works out the take-over against
/// the blocks as they stand, and lists it at the schema's block — or at the source's, for blocks without a schema,
/// where the saved schema would go — changing nothing. The list carries its own box, which applies what it listed.
/// There is something to take over only in a saved notebook whose blocks make one pipeline, with columns saved beside
/// it.
/// </remarks>
[VersoExtension]
public sealed class TakeOverAction : NotebookExtension, IToolbarAction
{
    /// <summary>The button's id.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.takeover";

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp take over the saved columns";

    /// <inheritdoc />
    public override string Description => "Lists what taking over the columns saved beside the notebook would change, with a box that applies it.";

    /// <inheritdoc />
    public string ActionId => Id;

    /// <inheritdoc />
    public string DisplayName => "Take over the saved columns";

    /// <inheritdoc />
    public string Icon =>
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" width=\"16\" height=\"16\" fill=\"currentColor\">"
        + "<path d=\"M4 3h10l6 6v12H4zM13 4v6h6M7 13h10v2H7zM7 17h7v2H7z\"/></svg>";

    /// <inheritdoc />
    public bool IconOnly => false;

    /// <inheritdoc />
    public bool IsPrimary => false;

    /// <inheritdoc />
    public string? ConfirmationPrompt => null;

    /// <inheritdoc />
    public ToolbarPlacement Placement => ToolbarPlacement.MainToolbar;

    /// <inheritdoc />
    public int Order => 1;

    /// <inheritdoc />
    /// <remarks>When the notebook is saved, its blocks make one pipeline, and columns are saved beside it.</remarks>
    public Task<bool> IsEnabledAsync(IToolbarActionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        var assembled = NotebookPipeline.Of(context.NotebookCells);

        return Task.FromResult(
            assembled.Whole && assembled.Blocks.Count > 0 && context.NotebookMetadata.ColumnsFilePath() is { } path && File.Exists(path));
    }

    /// <inheritdoc />
    public async Task ExecuteAsync(IToolbarActionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        var session = RequiredSession;
        var path = context.NotebookMetadata.ColumnsFilePath()
            ?? throw new InvalidOperationException("A notebook never saved has nothing saved beside it to take over.");

        await session.OneAtATimeAsync(async () =>
        {
            var assembled = NotebookPipeline.Of(context.NotebookCells);

            if (!File.Exists(path))
            {
                throw new InvalidOperationException("There are no columns saved beside the notebook to take over.");
            }

            if (assembled.Blocks.Count == 0)
            {
                throw new InvalidOperationException("There are no blocks to take the saved columns into.");
            }

            if (!assembled.Whole)
            {
                throw new InvalidOperationException(
                    $"The blocks do not make a pipeline yet, so there is nothing to take the saved columns into: {string.Join("; ", assembled.Stopping)}");
            }

            var at = ListedAt(assembled);

            session.Request(at, assembled.RequestFor(at, ViewTrigger.TakeOver, page: 0) with { Card = Listed(assembled.Readable, path, context.NotebookMetadata.SourceFolder()) });
            await context.Notebook.ExecuteCellAsync(at);

            return true;
        });
    }

    // What taking the saved columns over would change, as its list shows it; or why the saved file cannot be read.
    private static CellOutput Listed(PipelineDeclaration blocks, string path, SourceFolder folder)
    {
        string saved;
        PipelinePreset preset;

        try
        {
            saved = File.ReadAllText(path);
            preset = PipelinePreset.FromJson(saved, NotebookVerbs.Catalog());
        }
        catch (PipelineFileException unreadable)
        {
            return TakeOverCard.Unreadable(unreadable.Faults.Select(fault => fault.ToString()));
        }
        catch (Exception unreadable) when (unreadable is IOException or UnauthorizedAccessException)
        {
            return TakeOverCard.Unreadable([unreadable.Message]);
        }

        return TakeOverCard.Of(blocks, preset.TakeOver(blocks, HeaderOf(blocks, folder)), saved);
    }

    // The source's columns, read from its first line alone; nothing when the rows are handed in or the file will not read.
    private static IReadOnlyList<string>? HeaderOf(PipelineDeclaration blocks, SourceFolder folder)
    {
        if (blocks.Steps is not [ReadCsvStep read, ..])
        {
            return null;
        }

        try
        {
            return CsvRowSource.HeaderOf(folder.Resolve(read.Path));
        }
        catch (Exception unreadable) when (unreadable is IOException or UnauthorizedAccessException or FormatException)
        {
            return null;
        }
    }

    // The schema's block, where a take-over is listed; blocks without a schema list it at their source, which the saved
    // schema would follow.
    private static Guid ListedAt(NotebookPipeline assembled) => assembled.SchemaBlock ?? assembled.Blocks[0].Cell;
}
