// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>A column whose decision a take-over changes.</summary>
/// <param name="Column">The column.</param>
/// <param name="Before">How it stood.</param>
/// <param name="After">How it stands after.</param>
/// <param name="InSource">Whether the source has it; nothing when the source's columns are not known, or a step makes it.</param>
public readonly record struct ColumnChange(string Column, ColumnChoice Before, ColumnChoice After, bool? InSource);

/// <summary>The output a take-over changes.</summary>
/// <param name="Before">The output that stood, or nothing.</param>
/// <param name="After">The output after, or nothing.</param>
public readonly record struct OutputChange(INamesTheAnswer? Before, INamesTheAnswer? After);

/// <summary>The order of the columns both schemas declare, when a take-over changes it.</summary>
/// <param name="Before">The order they stood in.</param>
/// <param name="After">The order after.</param>
public readonly record struct OrderChange(IReadOnlyList<string> Before, IReadOnlyList<string> After);

/// <summary>What the schema does with the columns it does not name, when a take-over changes it.</summary>
/// <param name="Before">What it did, or nothing when there was no schema.</param>
/// <param name="After">What it does after.</param>
public readonly record struct RemainderChange(Remainder? Before, Remainder After);

/// <summary>A saved drop a take-over cannot make, because its column does not reach the end of the pipeline.</summary>
/// <param name="Column">The column the saved decisions drop.</param>
/// <param name="After">How the column stands after the take-over: why there is nothing to drop.</param>
/// <param name="InSource">Whether the source has it; nothing when the source's columns are not known, or a step makes it.</param>
/// <remarks>
/// Listed, never made: the steps stay as they are, and the columns saved beside a notebook forget the drop the next time
/// they are written.
/// </remarks>
public readonly record struct DropNotMade(string Column, ColumnChoice After, bool? InSource);

/// <summary>What taking saved column decisions over into a pipeline makes, and what it changes.</summary>
/// <param name="Steps">The steps after the take-over; none when it is refused.</param>
/// <param name="Faults">Every rule the result would break; none when it keeps them.</param>
/// <param name="NewColumns">The source's columns the saved decisions never showed; nothing when the source's columns are not known.</param>
/// <param name="Changes">Every column whose decision changes.</param>
/// <param name="Output">The output, when it changes.</param>
/// <param name="DeclareOrder">The schema's order, when it changes.</param>
/// <param name="DeclareRemainder">What the schema does with the columns it does not name, when that changes.</param>
/// <param name="DropsNotMade">Every saved drop the take-over cannot make; none when it is refused.</param>
public readonly record struct PresetTakeOver(
    IReadOnlyList<IPipelineStep> Steps, IReadOnlyList<DeclarationFault> Faults, IReadOnlyList<string>? NewColumns,
    IReadOnlyList<ColumnChange> Changes, OutputChange? Output, OrderChange? DeclareOrder, RemainderChange? DeclareRemainder,
    IReadOnlyList<DropNotMade> DropsNotMade);
