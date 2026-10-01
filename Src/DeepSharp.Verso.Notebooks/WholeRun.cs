// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// The notebook's whole pipeline over the rows a block's show read, run at most once for that show, and only when something
/// asks for it: what the run learned, handed to C# cells after the toolbar's run, and a report's measures of the predictions
/// a C# cell handed back.
/// </summary>
/// <param name="session">The notebook's session, which counts every run of the whole pipeline.</param>
/// <param name="pipeline">The notebook's pipeline, over the rows the show read.</param>
/// <param name="fingerprint">A SHA-256 of the bytes those rows were read from.</param>
/// <remarks>
/// A show at the last block of the toolbar's run hands the fit over and, when that block is a report, measures the
/// predictions handed back: both from this one run, so the pipeline is not fitted twice for one show.
/// </remarks>
internal sealed class WholeRun(NotebookSession session, Pipeline pipeline, string fingerprint)
{
    private PreparedData? _prepared;

    /// <summary>The notebook's session.</summary>
    public NotebookSession Session => session;

    /// <summary>The steps the pipeline runs.</summary>
    public PipelineDeclaration Declaration => pipeline.Declaration;

    /// <summary>A SHA-256 of the bytes the rows were read from.</summary>
    public string Fingerprint => fingerprint;

    /// <summary>The run: made the first time it is asked for, and the same run every time after.</summary>
    /// <exception cref="InvalidOperationException">The whole pipeline cannot run over these rows, as <see cref="Pipeline.Run()"/> says.</exception>
    public PreparedData Prepared => _prepared ??= session.Ran(pipeline.Run());
}
