// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>A list of names, as a message writes it.</summary>
/// <remarks>
/// Said once, for every message of the pipeline and of the learners behind it that names several columns, features, answers or
/// settings: each in single quotes, a comma between them.
/// </remarks>
internal static class NameListExtensions
{
    extension(IEnumerable<string> names)
    {
        /// <summary>The names as a message writes them: <c>'x', 'y', 'z'</c>.</summary>
        /// <returns>Each name in single quotes, a comma and a space between them; nothing, for no names.</returns>
        internal string Quoted() => string.Join(", ", names.Select(name => $"'{name}'"));
    }
}
