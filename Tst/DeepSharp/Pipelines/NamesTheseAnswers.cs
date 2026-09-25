// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>An output of another kind than the target: it names several answer columns and acts on nothing.</summary>
/// <param name="answers">The answer columns, in their order.</param>
internal sealed class NamesTheseAnswers(params string[] answers) : INamesTheAnswer, IDescribesColumns
{
    public string Verb => "test.answers";

    public IReadOnlyList<string> Answers => answers;

    public ColumnState After(ColumnState before) => before;

    public void WriteTo(Utf8JsonWriter writer) => throw new NotSupportedException("A test step is never written down.");
}
