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
    private static PipelineDeclaration WithTheSum(StepCatalog catalog) => new(
    [
        catalog.ReadStep("""{"step": "read.csv", "path": "titanic.csv"}"""),
        catalog.ReadStep("""{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "fare", "kind": "number", "optional": false}]}"""),
        catalog.ReadStep("""{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}"""),
        new SumStep(),
    ]);

    [Fact]
    public void ATickWhoseTakingInBreaksARuleOfAStepFromElsewhere_IsRefusedInThatRulesWords()
    {
        var catalog = NotebookVerbs.Catalog();

        var change = new OutputBox(catalog, WithTheSum(catalog), "target", ["survived", "name", "fare"]).Change("name", ColumnKind.Text, ticked: true);

        Assert.Null(change.Steps);
        Assert.Equal(["Step 4, 'sum.all': adds every column up, and 'name' is text."], change.NotMade);
    }

    [Fact]
    public void ARangeWhoseTakingInBreaksARuleOfAStepFromElsewhere_IsRefusedInThatRulesWords()
    {
        // The range gives the columns it takes in the kind the output reads, one after another, and stops at the rule that is broken.
        var catalog = NotebookVerbs.Catalog();

        var change = new OutputBox(catalog, WithTheSum(catalog), "target.labels", ["survived", "name", "fare"]).Ranged(["survived", "name"], ColumnKind.Text);

        Assert.Null(change.Steps);
        Assert.Equal(["Step 4, 'sum.all': adds every column up, and 'name' is text."], change.NotMade);
    }

    [Fact]
    public void ARangeOverStepsThatHoldNoSchema_TakesNothingIn_AndPlacesTheOutputAsTheRulesAllow()
    {
        // The range takes its columns in through the schema and asks the kind of a column the schema takes; steps with no schema
        // have neither, and the range is made as any other change is: kept to the rules, with nothing thrown for the missing schema.
        var catalog = NotebookVerbs.Catalog();
        var declaration = new PipelineDeclaration([catalog.ReadStep("""{"step": "read.csv", "path": "titanic.csv"}""")]);

        var change = new OutputBox(catalog, declaration, "target.labels", ["survived", "pclass"]).Ranged(["survived", "pclass"], ColumnKind.Integer);

        Assert.Empty(change.NotMade);
        Assert.Equal(["read.csv", "target.labels"], change.Steps!.Select(step => step.Verb));
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
