// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// What every part of this package tells Verso about itself.
/// </summary>
/// <remarks>
/// Verso finds each part on its own and makes one of each, and every part answers the same questions: who it is,
/// which version of the package it came in, who wrote it. They are answered here once. The parts keep nothing
/// between calls — Verso may render with one instance and handle a click with another — except the notebook's
/// session, which the block type keeps and every other part reaches through the host that loaded it, the same way:
/// among everything the host loaded, so a block type switched off still keeps the one session. A part no host
/// loaded beside the block type has no notebook, and says so.
/// </remarks>
public abstract class NotebookExtension : IExtension
{
    private IExtensionHostContext? _host;

    /// <summary>Only this package's own parts are notebook extensions.</summary>
    private protected NotebookExtension()
    {
    }

    /// <inheritdoc />
    public abstract string ExtensionId { get; }

    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    /// <remarks>The version the build stamped, without the commit the SDK appends after a <c>+</c>.</remarks>
    public string Version =>
        typeof(NotebookExtension).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()!
            .InformationalVersion.Split('+')[0];

    /// <inheritdoc />
    public string Author => "H.P. Gansevoort";

    /// <inheritdoc />
    public abstract string Description { get; }

    /// <inheritdoc />
    /// <remarks>Remembers the host: the way to the notebook's session, never state of the part's own.</remarks>
    public Task OnLoadedAsync(IExtensionHostContext context)
    {
        _host = context;

        return Task.CompletedTask;
    }

    /// <summary>
    /// The notebook's session, kept by the block type Verso loaded beside this part; nothing when Verso did not load
    /// this part, or loaded it without the block type.
    /// </summary>
    /// <remarks>
    /// Found among everything Verso loaded rather than among what is switched on: a block type switched off is left
    /// out of that list, and the notebook would split into two sessions that each know half of it.
    /// </remarks>
    private protected NotebookSession? LoadedSession => _host is null ? null : SessionLoadedBy(_host);

    /// <summary>The notebook's session, for a part about to act on the notebook.</summary>
    /// <exception cref="InvalidOperationException">
    /// Verso did not load this part beside the block type, so there is no notebook to act on.
    /// </exception>
    private protected NotebookSession RequiredSession =>
        SessionLoadedBy(Host) ?? throw new InvalidOperationException(NotLoadedBeside);

    /// <summary>Why a part not loaded beside the notebook's blocks does nothing, in the words every part uses.</summary>
    private protected string NotLoadedBeside =>
        $"'{Name}' was not loaded by Verso beside the notebook's blocks, so there is no notebook for it to act on.";

    /// <summary>Whether the layout a notebook is shown in lets a part do something to its blocks.</summary>
    /// <param name="notebook">What can be done to the notebook, which names its layout.</param>
    /// <param name="needed">What the part is about to do.</param>
    /// <returns><see langword="true"/> unless a layout Verso loaded is shown and does not allow it.</returns>
    /// <exception cref="InvalidOperationException">Verso did not load this part, so it knows no layout.</exception>
    private protected bool LayoutLets(INotebookOperations notebook, LayoutCapabilities needed) =>
        Host.GetLayouts().FirstOrDefault(layout => layout.LayoutId == notebook.ActiveLayoutId) is not { } shown
        || shown.Capabilities.HasFlag(needed);

    // The host that loaded this part: every question a part asks Verso goes through it, and one no host loaded has
    // nobody to ask.
    private IExtensionHostContext Host => _host ?? throw new InvalidOperationException(NotLoadedBeside);

    private static NotebookSession? SessionLoadedBy(IExtensionHostContext host) =>
        host.GetLoadedExtensions().OfType<StepCellType>().FirstOrDefault()?.Session;

    /// <inheritdoc />
    public Task OnUnloadedAsync() => Task.CompletedTask;
}
