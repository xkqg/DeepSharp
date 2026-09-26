// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Notebooks;
using Verso;
using Verso.Abstractions;
using Verso.Extensions;

namespace DeepSharp.Verso.Api;

/// <summary>One open notebook: Verso's engine with DeepSharp's parts, holding the notebook one file saves.</summary>
/// <remarks>
/// It is opened the way Verso's own editors open a notebook — through the serializer for its format, past the guards
/// that run after reading, with the cells that are only ever shown rendered drawn — and DeepSharp's parts are
/// registered before Verso looks for any beside the application, so an application published as a single file, which
/// has no assemblies beside it, still has them. An extension a notebook asks for is refused: the engine runs what the
/// application carries, and fetches nothing. A host is opened and closed by the <see cref="OpenNotebooks"/> that
/// holds it, never by whoever it was handed to.
/// </remarks>
public sealed class NotebookHost
{
    private NotebookHost(string filePath, ExtensionHost extensions, Scaffold scaffold)
    {
        FilePath = filePath;
        Extensions = extensions;
        Scaffold = scaffold;
    }

    /// <summary>The file the notebook is saved in, as a full path.</summary>
    public string FilePath { get; }

    /// <summary>The notebook's cells as they stand, in order.</summary>
    public IReadOnlyList<HostedCell> Cells => [.. Scaffold.Cells.Select(cell => cell.Hosted())];

    /// <summary>The engine's extensions: Verso's own and DeepSharp's.</summary>
    internal ExtensionHost Extensions { get; }

    /// <summary>The notebook as Verso's engine holds and runs it.</summary>
    internal Scaffold Scaffold { get; }

    /// <summary>Every part DeepSharp's notebook has, made anew for one engine.</summary>
    /// <returns>The parts.</returns>
    internal static IExtension[] Parts() =>
    [
        new StepCellType(),
        new StepKernel(),
        new StepRenderer(),
        new StepForm(),
        new RunPipelineAction(),
        new ExportPipelineAction(),
        new TakeOverAction(),
        new JupyterGuard(),
    ];

    /// <summary>Registers DeepSharp's parts with an engine, before it looks for any beside the application.</summary>
    /// <param name="extensions">The engine.</param>
    /// <returns>When they are registered.</returns>
    /// <remarks>
    /// First, because Verso's own look beside the application passes over a part already registered, while one
    /// registered after it would be refused as a second.
    /// </remarks>
    internal static async Task RegisterAsync(ExtensionHost extensions)
    {
        foreach (var part in Parts())
        {
            await extensions.LoadExtensionAsync(part);
        }
    }

    /// <summary>Opens the notebook a file holds on an engine, and closes the engine when it cannot be opened.</summary>
    /// <param name="filePath">The notebook's file, as a full path.</param>
    /// <param name="extensions">The engine to open it on; closed when the notebook cannot be opened.</param>
    /// <param name="cancellationToken">Stops the open.</param>
    /// <returns>The host.</returns>
    /// <exception cref="NotSupportedException">No format Verso knows reads the file.</exception>
    internal static async Task<NotebookHost> OpenAsync(string filePath, ExtensionHost extensions, CancellationToken cancellationToken)
    {
        Scaffold? scaffold = null;

        try
        {
            var content = await File.ReadAllTextAsync(filePath, cancellationToken);

            extensions.ConsentHandler = static (_, _) => Task.FromResult(false);
            await RegisterAsync(extensions);
            await extensions.LoadBuiltInExtensionsAsync();

            var serializer = extensions.GetSerializers().FirstOrDefault(each => each.CanImport(filePath))
                ?? throw new NotSupportedException($"No format Verso knows reads '{Path.GetFileName(filePath)}'.");
            var notebook = await serializer.DeserializeAsync(content);

            foreach (var guard in extensions.GetPostProcessors().Where(each => each.CanProcess(filePath, serializer.FormatId)).OrderBy(each => each.Priority))
            {
                notebook = await guard.PostDeserializeAsync(notebook, filePath);
            }

            // As Verso's own editors leave a notebook that names neither.
            notebook.DefaultKernelId ??= "csharp";
            notebook.ActiveLayout ??= LayoutDefaults.Reference;

            scaffold = new Scaffold(notebook, extensions, filePath);
            scaffold.InitializeSubsystems();
            await scaffold.RenderTransientCellsAsync(cancellationToken);

            return new NotebookHost(filePath, extensions, scaffold);
        }
        catch
        {
            if (scaffold is not null)
            {
                await scaffold.DisposeAsync();
            }

            await extensions.DisposeAsync();

            throw;
        }
    }

    /// <summary>Closes the notebook, and the engine it ran on.</summary>
    /// <returns>When both are closed.</returns>
    internal async ValueTask CloseAsync()
    {
        await Scaffold.DisposeAsync();
        await Extensions.DisposeAsync();
    }
}
