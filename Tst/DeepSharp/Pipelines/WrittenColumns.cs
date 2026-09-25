// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// An output as another package might write it: a class, so two of the same columns are two outputs to its equality,
/// though both write the same.
/// </summary>
/// <param name="columns">The answer columns, in their order.</param>
internal sealed class WrittenColumns(params string[] columns) : INamesTheAnswer, IDescribesColumns
{
    public string Verb => "test.columns";

    public IReadOnlyList<string> Answers => columns;

    public ColumnState After(ColumnState before) => before;

    public void WriteTo(Utf8JsonWriter writer)
    {
        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteStartArray("columns");

        foreach (var column in columns)
        {
            writer.WriteStringValue(column);
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
    }
}
