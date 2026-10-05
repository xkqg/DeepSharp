// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// A button that writes the steps of a course a notebook does not hold yet, each as its skeleton, in one turn.
/// </summary>
/// <remarks>
/// A pipeline written from nothing starts with an empty notebook and a person who has to know which step comes where. The
/// button writes every step of the course as a block that waits for what only its person knows — the file, the columns,
/// the bounds — and starts the rest as the step starts it, so the person fills in instead of remembering. A block that waits
/// does not read as a step, so it shows its own refusal naming the key it waits for and takes no part in the pipeline the
/// blocks above it make; a notebook shows no rule broken by a block nobody has touched.
/// It goes on from where the blocks stand: it never writes over a block, never moves one, puts the steps right after the
/// last block that is a step, and offers nothing to a notebook whose blocks take the course's steps in another order or that
/// holds all of it already. Pressing it twice writes the course once.
/// </remarks>
public abstract class CourseAction : NotebookExtension, IToolbarAction
{
    private readonly PipelineCourse _course;

    private protected CourseAction(PipelineCourse course) => _course = course;

    /// <inheritdoc />
    public string ActionId => ExtensionId;

    /// <summary>The button's label.</summary>
    public abstract string DisplayName { get; }

    /// <inheritdoc />
    public string Icon =>
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" width=\"16\" height=\"16\" fill=\"currentColor\">"
        + "<path d=\"M4 4h10v3H4zM4 9h10v3H4zM4 14h10v3H4zM16 4h4v3h-4zM16 9h4v3h-4zM16 14h4v3h-4zM4 19h16v2H4z\"/></svg>";

    /// <inheritdoc />
    public bool IconOnly => false;

    /// <inheritdoc />
    public bool IsPrimary => false;

    /// <inheritdoc />
    public string? ConfirmationPrompt => null;

    /// <inheritdoc />
    public ToolbarPlacement Placement => ToolbarPlacement.MainToolbar;

    /// <summary>Where the button stands among the others.</summary>
    public abstract int Order { get; }

    /// <inheritdoc />
    /// <remarks>
    /// When the layout the notebook is shown in lets a block be added, its blocks follow the course, and the course has a
    /// step they do not say.
    /// </remarks>
    public Task<bool> IsEnabledAsync(IToolbarActionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        return Task.FromResult(
            LayoutLets(context.Notebook, LayoutCapabilities.CellInsert)
            && CourseProgress.IsOfferedFor(_course, context.NotebookCells, NotebookVerbs.Catalog()));
    }

    /// <inheritdoc />
    public async Task ExecuteAsync(IToolbarActionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        var session = RequiredSession;
        var catalog = NotebookVerbs.Catalog();

        await session.OneAtATimeAsync(context.CancellationToken, async turn =>
        {
            if (!LayoutLets(context.Notebook, LayoutCapabilities.CellInsert)
                || !CourseProgress.IsOfferedFor(_course, context.NotebookCells, catalog))
            {
                return false;
            }

            var progress = CourseProgress.Of(_course, context.NotebookCells, catalog);

            // Every step is written, and what the blocks no longer make is taken back once, as one write for the turn.
            await session.LetThroughAsync(turn, async () =>
            {
                var at = progress.InsertAt;

                foreach (var step in progress.Missing)
                {
                    var made = await context.InsertedAsync(
                        at, new CellModel { Type = StepCellType.StepType, Language = StepKernel.Language, Source = step.AsBlockText(catalog) });

                    at = context.IndexOf(made) + 1;
                }

                var now = NotebookPipeline.Of(context.NotebookCells);

                foreach (var stale in session.StaleIn(now, except: null))
                {
                    await context.Notebook.ClearOutputAsync(stale);
                }

                session.CaughtUp(now, context.Variables, except: null);
            });

            return true;
        });
    }
}

/// <summary>The button that writes the course for the rows of a table into the notebook.</summary>
/// <remarks>For rows that do not depend on one another: see <see cref="PipelineCourse.Table"/>.</remarks>
[VersoExtension]
public sealed class TableCourseAction : CourseAction
{
    /// <summary>The button's id.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.course.table";

    /// <summary>The button for the course of a table.</summary>
    public TableCourseAction()
        : base(PipelineCourse.Table)
    {
    }

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp the course for a table";

    /// <inheritdoc />
    public override string Description =>
        "Writes the steps of the course for the rows of a table that the notebook does not hold yet, each as a block that waits for what only you know.";

    /// <inheritdoc />
    public override string DisplayName => "Course for a table";

    /// <inheritdoc />
    public override int Order => 3;
}

/// <summary>The button that writes the course for a series in time into the notebook.</summary>
/// <remarks>For rows that follow one another in time: see <see cref="PipelineCourse.SeriesInTime"/>.</remarks>
[VersoExtension]
public sealed class SeriesCourseAction : CourseAction
{
    /// <summary>The button's id.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.course.series";

    /// <summary>The button for the course of a series in time.</summary>
    public SeriesCourseAction()
        : base(PipelineCourse.SeriesInTime)
    {
    }

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp the course for a series in time";

    /// <inheritdoc />
    public override string Description =>
        "Writes the steps of the course for a series in time that the notebook does not hold yet, each as a block that waits for what only you know.";

    /// <inheritdoc />
    public override string DisplayName => "Course for a series";

    /// <inheritdoc />
    public override int Order => 4;
}
