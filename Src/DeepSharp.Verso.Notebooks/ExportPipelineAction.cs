// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// The button that saves the pipeline the blocks declare as a pipeline file, for the file door to read.
/// </summary>
/// <remarks>
/// The file holds the declaration, and what a run of the whole pipeline learned only while it is the fit of the steps
/// the blocks declare now: a fit of other steps would be served under steps it never saw. It is named after the
/// notebook, and handed to the person to save through the host, which knows where files go; a notebook never saved
/// gives one of the usual name. There is nothing to export while the blocks make no pipeline.
/// </remarks>
[VersoExtension]
public sealed class ExportPipelineAction : NotebookExtension, IToolbarAction
{
    /// <summary>The button's id.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.export";

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp export the pipeline";

    /// <inheritdoc />
    public override string Description => "Saves the pipeline the blocks declare as a pipeline file, with what a run learned while it is the run of those steps.";

    /// <inheritdoc />
    public string ActionId => Id;

    /// <inheritdoc />
    public string DisplayName => "Export the pipeline";

    /// <inheritdoc />
    public string Icon =>
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" width=\"16\" height=\"16\" fill=\"currentColor\">"
        + "<path d=\"M11 3h2v10l3.5-3.5 1.4 1.4L12 16.8l-5.9-5.9 1.4-1.4L11 13zM4 19h16v2H4z\"/></svg>";

    /// <inheritdoc />
    public bool IconOnly => false;

    /// <inheritdoc />
    public bool IsPrimary => false;

    /// <inheritdoc />
    public string? ConfirmationPrompt => null;

    /// <inheritdoc />
    public ToolbarPlacement Placement => ToolbarPlacement.ExportMenu;

    /// <inheritdoc />
    public int Order => 0;

    /// <inheritdoc />
    /// <remarks>When the blocks make one pipeline.</remarks>
    public Task<bool> IsEnabledAsync(IToolbarActionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        var assembled = NotebookPipeline.Of(context.NotebookCells);

        return Task.FromResult(assembled.Whole && assembled.Blocks.Count > 0);
    }

    /// <inheritdoc />
    public Task ExecuteAsync(IToolbarActionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        var assembled = NotebookPipeline.Of(context.NotebookCells);

        if (!assembled.Whole)
        {
            throw new InvalidOperationException(
                $"The blocks do not make a pipeline yet, so there is none to export: {string.Join("; ", assembled.Stopping)}");
        }

        var bytes = SourceCache.BytesOf(assembled.Readable, context.NotebookMetadata.SourceFolder());
        var file = RequiredSession.EnvelopeFor(context.Variables, assembled, bytes)!;

        return context.RequestFileDownloadAsync(FileName(context.NotebookMetadata.FilePath), "application/json", Encoding.UTF8.GetBytes(file));
    }

    // Named after the notebook; a relative source path in it is read from wherever the file is saved.
    private static string FileName(string? notebook) =>
        string.IsNullOrWhiteSpace(notebook) ? "pipeline.json" : $"{Path.GetFileNameWithoutExtension(notebook)}.pipeline.json";
}
