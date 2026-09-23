// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// One thing wrong with a pipeline file, and where in the file it is.
/// </summary>
/// <param name="Line">The line, counting from one.</param>
/// <param name="Column">The column, counting from one, in characters as an editor counts them rather than in bytes.</param>
/// <param name="Message">What is wrong, in the words a person reads it in.</param>
/// <remarks>
/// The place is the start of the thing the fault is about: the step, the fitted entry, the key at the top of
/// the file, or the character where the text stopped being JSON. A file refuses to load with every fault it
/// has at once, because somebody handed one fault at a time, five times over, stops using the thing.
/// </remarks>
public readonly record struct PipelineFileFault(int Line, int Column, string Message)
{
    /// <summary>The fault as an editor prints it: where it is, then what is wrong.</summary>
    /// <returns>The line and column in brackets, and the message.</returns>
    public override string ToString() => $"({Line},{Column}): {Message}";
}

/// <summary>
/// A pipeline file that cannot be read, with every fault it has, each where it is in the file.
/// </summary>
/// <remarks>
/// A <see cref="FormatException"/>, because whatever is wrong is wrong with the file: a step nobody can read, a
/// verb nobody registered, a rule the steps break, a fit that does not belong to them. The faults come with it
/// as data, so a caller can put each one at its line without reading the message apart.
/// </remarks>
public sealed class PipelineFileException : FormatException
{
    /// <summary>A refusal carrying every fault the file has.</summary>
    /// <param name="faults">The faults, in the order they stand in the file.</param>
    public PipelineFileException(IReadOnlyList<PipelineFileFault> faults)
        : base(string.Join(Environment.NewLine, faults ?? throw new ArgumentNullException(nameof(faults)))) =>
        Faults = faults;

    /// <summary>Every fault, in the order they stand in the file.</summary>
    public IReadOnlyList<PipelineFileFault> Faults { get; }
}
