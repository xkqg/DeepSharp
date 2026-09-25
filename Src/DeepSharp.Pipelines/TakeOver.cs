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

/// <summary>What taking saved column decisions over into a pipeline makes, and what it changes.</summary>
/// <param name="Steps">The steps after the take-over; none when it is refused.</param>
/// <param name="Faults">Every rule the result would break; none when it keeps them.</param>
/// <param name="NewColumns">The source's columns the saved decisions never showed; nothing when the source's columns are not known.</param>
/// <param name="Changes">Every column whose decision changes.</param>
/// <param name="Output">The output, when it changes.</param>
/// <param name="DeclareOrder">The schema's order, when it changes.</param>
public readonly record struct PresetTakeOver(
    IReadOnlyList<IPipelineStep> Steps, IReadOnlyList<DeclarationFault> Faults, IReadOnlyList<string>? NewColumns,
    IReadOnlyList<ColumnChange> Changes, OutputChange? Output, OrderChange? DeclareOrder);
