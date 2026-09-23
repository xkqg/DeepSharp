// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Reading a declared value out of a file, where absent and wrongly typed are both faults of the file.
/// </summary>
/// <remarks>
/// The BCL throws a different exception for each of those, and neither says which step or which parameter
/// was at fault. A person editing a pipeline by hand has to be told where to look, so both become one
/// <see cref="FormatException"/> that names the parameter.
/// </remarks>
internal static class JsonElementExtensions
{
    internal static string RequiredString(this JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.String)
        {
            throw new FormatException($"The step is missing a text value for '{name}'.");
        }

        return value.GetString()!;
    }

    internal static double RequiredNumber(this JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Number)
        {
            throw new FormatException($"The step is missing a number for '{name}'.");
        }

        return value.GetDouble();
    }
}
