// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>A network file that cannot be read, with every fault found in it, each at its line and column.</summary>
/// <remarks>
/// Every fault at once, so a file is mended in one sitting rather than one error at a time; each at the line and column of
/// the file it was read from, whichever part of a larger file the network stood in.
/// </remarks>
public sealed class NetworkFileException : FormatException
{
    /// <summary>A refusal of a network file.</summary>
    /// <param name="faults">Every fault found in it, in the order they stand in the file.</param>
    public NetworkFileException(IReadOnlyList<NetworkFileFault> faults)
        : base(string.Join(Environment.NewLine, faults ?? throw new ArgumentNullException(nameof(faults))))
    {
        Faults = faults;
    }

    /// <summary>Every fault found, in the order they stand in the file.</summary>
    public IReadOnlyList<NetworkFileFault> Faults { get; }
}

/// <summary>One fault in a network file.</summary>
/// <param name="Line">The line it is on, counted from one.</param>
/// <param name="Column">The column, counted from one, in characters.</param>
/// <param name="Message">What is wrong there.</param>
public readonly record struct NetworkFileFault(int Line, int Column, string Message)
{
    /// <summary>The fault as an editor would place it: <c>(12,5): …</c>.</summary>
    public override string ToString() => $"({Line},{Column}): {Message}";
}
