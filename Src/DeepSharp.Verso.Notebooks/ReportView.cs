// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>What a report's block draws is measured from: the steps, the bytes of the rows, and the predictions handed back.</summary>
/// <param name="Key">The key of the whole declaration, which covers every step of it.</param>
/// <param name="Fingerprint">A SHA-256 of the bytes the rows were read from.</param>
/// <param name="Predictions">The predictions, as the text they were handed back as.</param>
internal readonly record struct ReportFrom(string Key, string Fingerprint, string Predictions);

/// <summary>
/// What a report's block draws under its grid: the measures of the predictions a C# cell handed back, taken by the report on
/// the notebook's own run of its blocks — or, while none are handed back, where they come from.
/// </summary>
/// <param name="run">
/// The notebook's whole pipeline over the rows the grid was read from, run once for the show — the one the fit handed over
/// comes from too — with the session that keeps the report drawn last.
/// </param>
/// <remarks>
/// The notebook trains nothing, so what a model predicted reaches it as text, under <see cref="StepKernel.HandedBack"/>, and
/// is measured here by the pipeline's own measuring on the notebook's own run, only beside the very fit it was made behind:
/// predictions handed back before a block was edited are made behind another fit and refused, as are those of another
/// output, by the answers' names, and of other rows, by their keys — each said at the block, in the words that refused it,
/// with what to do.
/// What is drawn is DeepSharp.Charts' rendering of the report, through <see cref="EvidenceView"/>, so the block shows what a
/// cell that ends with the report shows. It is kept under what it was measured from, so another page of the grid, or the
/// block shown again, measures nothing again.
/// </remarks>
internal sealed class ReportView(WholeRun run)
{
    /// <summary>What the block of a report draws under its grid.</summary>
    /// <param name="report">The report.</param>
    /// <param name="variables">The notebook's variables, where a C# cell hands predictions back.</param>
    /// <returns>The measures drawn, why the predictions handed back were refused, or where the measures come from.</returns>
    public CellOutput Of(INamesTheMeasures report, IVariableStore variables)
    {
        // The store keeps no gaps, so a name it holds text under holds some; anything else under it is no text to measure.
        if (!variables.TryGet<string>(StepKernel.HandedBack, out var predictions))
        {
            return StepCard.NotMeasured(report);
        }

        var from = new ReportFrom(NotebookSession.KeyOf(run.Declaration), run.Fingerprint, predictions!);

        return run.Session.ReportFor(from) ?? run.Session.KeepReport(from, Measured(predictions!));
    }

    // The report's measures of the predictions, on the run of the blocks over the rows the grid was read from; or what refused
    // them — the text, the answers' names, the fit they were made behind, the rows' keys, or a whole run that cannot be made.
    private CellOutput Measured(string predictions)
    {
        try
        {
            return run.Prepared.MeasureAgain(predictions).Accept(new EvidenceView());
        }
        catch (ArgumentException refused)
        {
            return StepCard.PredictionsRefused(refused.Message);
        }
        catch (InvalidOperationException refused)
        {
            return StepCard.PredictionsRefused(refused.Message);
        }
    }
}
