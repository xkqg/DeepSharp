// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Notebooks;
using Verso;
using Verso.Abstractions;
using Verso.Diffing;
using Verso.Extensions;

namespace DeepSharp.Verso.Api;

/// <summary>
/// What the engine is asked about itself, each question asked of the thing it is about.
/// </summary>
/// <remarks>
/// None of these touches an open notebook's own state: they are what the engine and the notebook it holds answer, as
/// Verso's own editors ask them. Kept apart from the host for that reason — a host holds one notebook's turn, and a
/// question about which kinds of cell the engine has, or which kernel runs a cell, is no part of that.
/// </remarks>
internal static class EngineExtensions
{
    /// <summary>The language the engine's C# kernel is named by.</summary>
    internal const string CSharp = "csharp";

    extension(string? name)
    {
        /// <summary>Whether two of the engine's names name the same thing, as the engine compares them.</summary>
        /// <param name="other">The other name.</param>
        /// <returns>Whether they name the same thing.</returns>
        internal bool IsNamed(string? other) => string.Equals(name, other, StringComparison.OrdinalIgnoreCase);
    }

    extension(ExtensionHost extensions)
    {
        /// <summary>
        /// The kinds Verso's editors offer, with each language folded in: code in every language the engine runs but the
        /// blocks' own, then Markdown, then every other kind of cell in the order the engine has them.
        /// </summary>
        /// <returns>The kinds.</returns>
        /// <remarks>The languages are the engine's kernels, which is what its list of registered languages holds in a notebook opened here.</remarks>
        internal HostedKind[] KindsOf() =>
        [
            .. extensions.GetKernels()
                .Where(kernel => !kernel.LanguageId.IsNamed(StepKernel.Language))
                .Select(kernel => new HostedKind("code", kernel.LanguageId, kernel.DisplayName, Editable: true, Rendered: false)),
            .. extensions.GetCellTypes()
                .Where(type => !type.CellTypeId.IsNamed("code"))
                .OrderBy(type => type.CellTypeId.IsNamed("markdown") ? 0 : 1)
                .Select(type => new HostedKind(type.CellTypeId, type.Kernel?.LanguageId, type.DisplayName, type.IsEditable, extensions.Rendered(type))),
        ];

        /// <summary>The layouts the engine has, as Verso's View panel lists them.</summary>
        /// <returns>The layouts.</returns>
        internal HostedLayoutChoice[] LayoutsOf() =>
        [
            .. extensions.GetLayouts().Select(layout =>
                new HostedLayoutChoice(layout.LayoutId, layout.DisplayName, Enum.Parse<LayoutAllows>(layout.Capabilities.ToString()))),
        ];

        /// <summary>The themes the engine has, as Verso's View panel lists them, each with the block Verso's engine writes for it.</summary>
        /// <returns>The themes.</returns>
        internal HostedTheme[] ThemesOf() =>
        [
            .. extensions.GetThemes().Select(theme =>
                new HostedTheme(theme.ThemeId, theme.DisplayName, Enum.Parse<ThemeTone>(theme.ThemeKind.ToString()), ThemeCss.BuildRootBlock(theme))),
        ];

        /// <summary>
        /// Whether a kind is shown rendered once it has run, as Verso's editors decide it: by the renderer the engine has for
        /// its type, else by the one its cell type brings.
        /// </summary>
        /// <param name="type">The kind of cell.</param>
        /// <returns>Whether it is shown rendered.</returns>
        internal bool Rendered(ICellType type) =>
            (extensions.GetRenderers().FirstOrDefault(renderer => renderer.CellTypeId.IsNamed(type.CellTypeId)) ?? type.Renderer)?.CollapsesInputOnExecute ?? false;

        /// <summary>The language Verso's editors give a cell of a type named with no language.</summary>
        /// <param name="scaffold">The notebook as the engine holds it.</param>
        /// <param name="type">The kind of cell.</param>
        /// <returns>The language; none for a type a renderer draws.</returns>
        internal string? LanguageOf(Scaffold scaffold, string type)
        {
            if (extensions.GetCellTypes().FirstOrDefault(each => each.CellTypeId.IsNamed(type)) is { } cellType)
            {
                return cellType.Kernel?.LanguageId;
            }

            return extensions.GetRenderers().Any(renderer => renderer.CellTypeId.IsNamed(type)) ? null : scaffold.DefaultKernelId ?? CSharp;
        }

        /// <summary>The kernel the engine finds for a language: one it was given, one a part brings, or one a cell type brings.</summary>
        /// <param name="scaffold">The notebook as the engine holds it.</param>
        /// <param name="language">The language.</param>
        /// <returns>The kernel; none when no kernel reads that language.</returns>
        internal ILanguageKernel? KernelNamed(Scaffold scaffold, string language) =>
            scaffold.GetKernel(language)
            ?? extensions.GetCellTypes().Select(type => type.Kernel).FirstOrDefault(kernel => kernel is not null && kernel.LanguageId.IsNamed(language));

        /// <summary>
        /// Every toolbar button the engine has — Verso's own and DeepSharp's — each saying whether it can be pressed now, by
        /// place and then in their order.
        /// </summary>
        /// <param name="scaffold">The notebook as the engine holds it.</param>
        /// <returns>The buttons.</returns>
        /// <remarks>
        /// A button of the notebook as a whole is asked once, with no cell chosen; one on a cell's toolbar or in its menu is
        /// asked for every cell, as Verso's editors ask it for the cell it is drawn on, so a page draws it pressable where it
        /// is — running a cell wherever the layout lets cells run, clearing one once it shows something. Asking is a look,
        /// which does nothing to the notebook.
        /// </remarks>
        internal async Task<IReadOnlyList<HostedToolbarAction>> ButtonsAsync(Scaffold scaffold)
        {
            var look = new ReadPort(scaffold);
            var buttons = new List<HostedToolbarAction>();

            foreach (var action in extensions.GetToolbarActions())
            {
                var place = Enum.Parse<ToolbarPlace>(action.Placement.ToString());
                var cells = new List<Guid>();
                string? fault = null;

                if (place is ToolbarPlace.CellToolbar or ToolbarPlace.ContextMenu)
                {
                    foreach (var cell in scaffold.Cells)
                    {
                        var forCell = await action.AnswerAsync(new ToolbarContext(scaffold, [cell.Id], look));

                        if (forCell.Pressable)
                        {
                            cells.Add(cell.Id);
                        }

                        fault ??= forCell.Fault;
                    }
                }

                var whole = await action.AnswerAsync(new ToolbarContext(scaffold, [], look));

                buttons.Add(new HostedToolbarAction(
                    action.ActionId,
                    action.DisplayName,
                    action.Icon,
                    action.IconOnly,
                    action.IsPrimary,
                    action.ConfirmationPrompt,
                    place,
                    action.Order,
                    whole.Pressable,
                    cells,
                    whole.Fault ?? fault));
            }

            return [.. buttons.OrderBy(button => button.Place).ThenBy(button => button.Order)];
        }

        /// <summary>
        /// What the engine falls back on when the notebook names none, or names one it does not have — the notebook's own
        /// layout, a light theme — set on the engine and never written into the notebook, as Verso's browser editor sets them.
        /// </summary>
        /// <param name="scaffold">The notebook as the engine holds it.</param>
        internal void EnsureDefaults(Scaffold scaffold)
        {
            if (scaffold.LayoutManager is { ActiveLayout: null } layouts)
            {
                layouts.TryActivate(LayoutDefaults.LayoutId);
            }

            if (scaffold.ThemeEngine is { ActiveTheme: null } themes && extensions.GetThemes().FirstOrDefault(theme => theme.ThemeKind == ThemeKind.Light) is { } light)
            {
                themes.SetActiveTheme(light.ThemeId);
            }
        }
    }

    extension(Scaffold scaffold)
    {
        /// <summary>The layout the notebook is shown in, as the engine holds it now.</summary>
        internal HostedLayout Layout => new(
            scaffold.NotebookOps.ActiveLayoutId,
            Enum.Parse<LayoutAllows>(scaffold.LayoutCapabilities.ToString()),
            scaffold.LayoutManager?.ActiveLayout?.SupportsPropertiesPanel ?? true);

        /// <summary>
        /// The theme the notebook chose, as the engine resolved it; none while it chose none.
        /// </summary>
        /// <remarks>
        /// The engine's own default is drawn by nobody but the engine's parts, and a view draws the notebook in its own look.
        /// </remarks>
        internal string? ThemeId => scaffold.Notebook.PreferredThemeId is null ? null : scaffold.ThemeEngine?.ActiveTheme?.ThemeId;

        /// <summary>What the notebook says of itself, as the engine holds it now.</summary>
        internal HostedMetadata SaysOfItself =>
            new(scaffold.Title, scaffold.DefaultKernelId, scaffold.Notebook.Created, scaffold.Notebook.Modified, scaffold.Notebook.FormatVersion);

        /// <summary>
        /// What the layout the notebook is shown in draws of its own, as Verso's editors ask it to draw.
        /// </summary>
        /// <returns>What it drew; nothing when its layout draws nothing of its own.</returns>
        /// <remarks>
        /// Only a layout that draws an arrangement in the page itself rather than the notebook's list, or in a frame of its
        /// own — and never the engine's own notebook layout, the list of the cells, whose arrangement says nothing a version
        /// does not. Drawing is a look, and a look does nothing to the notebook. A part that fails to draw holds up no
        /// version, and says why instead.
        /// </remarks>
        internal async Task<HostedArrangement> ArrangementAsync()
        {
            if (scaffold.LayoutManager?.ActiveLayout is not { RequiresCustomRenderer: true, RendererIsolation: LayoutRendererIsolation.Inline } layout
                || (layout.LayoutId.IsNamed(LayoutDefaults.LayoutId) && layout is IExtension { ExtensionId: var extension } && extension.IsNamed(LayoutDefaults.ExtensionId)))
            {
                return default;
            }

            try
            {
                var drawn = await layout.RenderLayoutAsync(scaffold.Notebook.Cells, new ToolbarContext(scaffold, [], new ReadPort(scaffold)));

                return drawn.MimeType.IsNamed("text/html") ? new HostedArrangement(drawn.Content, null) : default;
            }
            catch (Exception failed)
            {
                return new HostedArrangement(null, failed.Message);
            }
        }

        /// <summary>
        /// What the file holds for the notebook's layouts and for the parts' settings, handed back to them as the notebook
        /// opens, as Verso's editors hand them back.
        /// </summary>
        /// <returns>When they have it.</returns>
        /// <remarks>
        /// Before anything draws the notebook, which would otherwise arrange the layouts afresh, and before any save takes
        /// them back, which would otherwise take the parts' defaults in their place.
        /// </remarks>
        internal async Task RestoreAsync()
        {
            if (scaffold.LayoutManager is { } layouts)
            {
                // Restoring an arrangement is a look at the notebook, and a look does nothing to it.
                await layouts.RestoreMetadataAsync(scaffold.Notebook, new ToolbarContext(scaffold, [], new ReadPort(scaffold)));
            }

            if (scaffold.SettingsManager is { } settings)
            {
                await settings.RestoreSettingsAsync(scaffold.Notebook);
            }
        }

        /// <summary>What the layouts and the parts' settings hold now, taken back into the notebook.</summary>
        /// <returns>When the notebook holds them.</returns>
        /// <remarks>
        /// Before every save and every look at what is unsaved, as Verso's editors take them — since each keeps what a person
        /// changed in it until then.
        /// </remarks>
        internal async Task FlushAsync()
        {
            if (scaffold.LayoutManager is { } layouts)
            {
                await layouts.SaveMetadataAsync(scaffold.Notebook);
            }

            if (scaffold.SettingsManager is { } settings)
            {
                await settings.SaveSettingsAsync(scaffold.Notebook);
            }
        }

        /// <summary>
        /// Refuses a change a layout with no properties panel does not let a person make, as Verso's editors offer the panel
        /// only in a layout that has one.
        /// </summary>
        /// <exception cref="InvalidOperationException">The layout has no properties panel.</exception>
        /// <remarks>A form's field rewrites a block, which such a layout does not let a person do.</remarks>
        internal void ThrowIfNoPanel()
        {
            if (!scaffold.Layout.HasPropertiesPanel)
            {
                throw new InvalidOperationException("The layout the notebook is shown in has no properties panel.");
            }
        }

        /// <summary>
        /// Refuses a change to a cell's text or kind in a layout that does not allow it, as the engine's own port refuses what
        /// its layout does not allow.
        /// </summary>
        /// <exception cref="LayoutCapabilityException">The layout does not let a cell be edited.</exception>
        internal void ThrowIfNoEdit()
        {
            if (!scaffold.LayoutCapabilities.HasFlag(LayoutCapabilities.CellEdit))
            {
                throw new LayoutCapabilityException(LayoutCapabilities.CellEdit);
            }
        }

        /// <summary>
        /// The kernel a cell's text is written for, started first as Verso's editors start it before they ask it anything.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <returns>The kernel; none for a cell whose language no kernel the engine has reads.</returns>
        internal async Task<ILanguageKernel?> KernelForAsync(CellModel cell)
        {
            if (cell.Language is not { } language || scaffold.GetKernel(language) is not { } kernel)
            {
                return null;
            }

            await scaffold.WarmUpKernelAsync(language);

            return kernel;
        }
    }

    extension(CellModel cell)
    {
        /// <summary>Which kernel runs a cell, in the order the engine asks when it runs one.</summary>
        /// <param name="scaffold">The notebook as the engine holds it.</param>
        /// <param name="extensions">The engine.</param>
        /// <returns>The kernel's language; none for a cell no kernel runs.</returns>
        /// <remarks>
        /// A cell type the engine has answers first — with its own kernel, or with none when it only draws; a cell of no such
        /// type runs in the kernel its language names; a type a renderer claims is drawn and runs none; a cell that names no
        /// language runs in the notebook's default kernel. None is a cell no kernel runs, so a run takes the C# turn, and a
        /// stop starts a kernel afresh, only when a kernel really runs the cell.
        /// </remarks>
        internal string? KernelIn(Scaffold scaffold, ExtensionHost extensions)
        {
            if (extensions.GetCellTypes().FirstOrDefault(type => type.CellTypeId.IsNamed(cell.Type)) is { } type)
            {
                return type.Kernel?.LanguageId;
            }

            if (!string.IsNullOrEmpty(cell.Language) && extensions.KernelNamed(scaffold, cell.Language) is { } named)
            {
                return named.LanguageId;
            }

            if (extensions.GetRenderers().Any(renderer => renderer.CellTypeId.IsNamed(cell.Type)))
            {
                return null;
            }

            // A language no kernel reads runs in none: the engine takes it for the default kernel's name, and finds no kernel.
            return cell.Language is null ? scaffold.DefaultKernelId : null;
        }
    }

    extension(IReadOnlyList<HostedKind> kinds)
    {
        /// <summary>A kind as the notebook lists it; one it does not list is refused.</summary>
        /// <param name="kind">The kind asked for.</param>
        /// <param name="language">The language it is taken to be in when it names none.</param>
        /// <returns>The kind the notebook lists.</returns>
        /// <exception cref="InvalidOperationException">The notebook lists no such kind.</exception>
        internal HostedKind Listed(HostedKind kind, string? language)
        {
            foreach (var each in kinds.Where(each => each.Type.IsNamed(kind.Type) && each.Language.IsNamed(language)))
            {
                return each;
            }

            throw new InvalidOperationException($"The notebook has no kind of cell '{kind.Type}' in '{kind.Language}' to add or turn a cell into.");
        }
    }

    extension(INotebookSerializer serializer)
    {
        /// <summary>Reads a notebook the way Verso's own editors read one: through its format's serializer, and past the guards that run after reading.</summary>
        /// <param name="extensions">The engine whose guards run.</param>
        /// <param name="content">The file's text.</param>
        /// <param name="path">The file, as a full path.</param>
        /// <returns>The notebook.</returns>
        /// <remarks>
        /// Nothing the engine falls back on is written into it, as Verso's browser editor writes nothing: a file that names no
        /// kernel or layout is saved naming none.
        /// </remarks>
        internal async Task<NotebookModel> ReadAsync(ExtensionHost extensions, string content, string path)
        {
            var notebook = await serializer.DeserializeAsync(content);

            foreach (var guard in extensions.GetPostProcessors().Where(each => each.CanProcess(path, serializer.FormatId)).OrderBy(each => each.Priority))
            {
                notebook = await guard.PostDeserializeAsync(notebook, path);
            }

            return notebook;
        }
    }

    extension(NotebookModel saved)
    {
        /// <summary>
        /// Whether a notebook differs from this one, as Verso's own comparison of two notebooks finds it, told which cells'
        /// outputs are never saved.
        /// </summary>
        /// <param name="scaffold">The notebook as the engine holds it now.</param>
        /// <param name="path">The file, as a full path.</param>
        /// <param name="extensions">The engine, which says which cells' outputs are never saved.</param>
        /// <returns>Whether a save would change the file.</returns>
        internal bool DiffersFrom(Scaffold scaffold, string path, ExtensionHost extensions)
        {
            var diff = NotebookDiffEngine.Compute(saved, scaffold.Notebook, path, extensions.GetCellTypes());

            return diff.Summary.Added + diff.Summary.Removed + diff.Summary.Modified + diff.Summary.Moved + diff.MetadataChanges.Count > 0;
        }
    }

    extension(IToolbarAction action)
    {
        /// <summary>Whether a button can be pressed, as its part says.</summary>
        /// <param name="context">What the part is told about the notebook.</param>
        /// <returns>Whether it can, or why the part could not say.</returns>
        /// <remarks>
        /// A part that fails to say is not pressed: every version asks every button, and one part that fails holds up no
        /// version, so what failed is told with the button instead.
        /// </remarks>
        internal async Task<Answer> AnswerAsync(ToolbarContext context)
        {
            try
            {
                return new Answer(await action.IsEnabledAsync(context), null);
            }
            catch (Exception failed)
            {
                return new Answer(false, failed.Message);
            }
        }
    }
}

/// <summary>What a button's part answered when asked whether it can be pressed: whether it can, or why it could not say.</summary>
/// <param name="Pressable">Whether it can be pressed.</param>
/// <param name="Fault">Why its part could not say; none when it said.</param>
internal readonly record struct Answer(bool Pressable, string? Fault);
