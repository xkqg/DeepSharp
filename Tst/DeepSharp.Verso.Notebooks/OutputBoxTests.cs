// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A row's output box takes a column the schema does not take in first, and that taking in is kept to the rules
/// before the output is placed: a rule it breaks refuses the tick in the rule's own words.
/// </summary>
public sealed class OutputBoxTests
{
    [Fact]
    public void ATickWhoseTakingInBreaksARuleOfAStepFromElsewhere_IsRefusedInThatRulesWords()
    {
        var catalog = NotebookVerbs.Catalog();
        var declaration = new PipelineDeclaration(
        [
            catalog.ReadStep("""{"step": "read.csv", "path": "titanic.csv"}"""),
            catalog.ReadStep("""{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "fare", "kind": "number", "optional": false}]}"""),
            catalog.ReadStep("""{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}"""),
            new SumStep(),
        ]);

        var change = OutputBox.Change(catalog, declaration, "target", "name", ColumnKind.Text, ["survived", "name", "fare"], ticked: true);

        Assert.Null(change.Steps);
        Assert.Equal(["Step 4, 'sum.all': adds every column up, and 'name' is text."], change.NotMade);
    }

    // A verb another package brings: every column added up into one, which a column of text cannot be.
    private sealed record SumStep : IPipelineStep<SumStep>, IAddsColumns, IDescribesColumns
    {
        public static string Name => "sum.all";

        public static string Purpose => "Adds every column up into one.";

        public static StepParameters<SumStep> Parameters { get; } = new();

        public string Verb => Name;

        public static SumStep ReadFrom(JsonElement element) => new();

        public void AddTo(Table table) => throw new NotSupportedException("Only declared here, never run.");

        public ColumnState After(ColumnState before) => before.With("sum", ColumnKind.Number);

        public string? Refusal(ColumnState before) =>
            before.Columns.Where(column => column.Kind == ColumnKind.Text).Select(column => column.Name).FirstOrDefault() is { } text
                ? $"adds every column up, and '{text}' is text."
                : null;
    }
}
