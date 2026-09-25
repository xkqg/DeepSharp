// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Saying which columns take part, and what they are. This is the step that decides what the rest of the
/// pipeline is even allowed to see: a column nobody declared is dropped, because a column nobody thought
/// about is a column nobody checked — and a published dataset can hold the answer in a column nobody
/// thought about, which no amount of splitting discipline would catch.
/// </summary>
public class SchemaTests
{
    [Fact]
    public void ADeclaredSchema_KnowsItsColumnsAndTheirKinds()
    {
        var builder = Pdd.Create()
            .ReadCsv("btceur-1d.csv")
            .Declare(schema => schema
                .Timestamp("timestamp")
                .Number("open", "high", "low", "close", "volume")
                .Optional("trades", ColumnKind.Integer));

        var declared = Assert.IsType<DeclareStep>(builder.Declaration.Steps[1]);

        Assert.Equal(7, declared.Columns.Count);
        Assert.Equal(ColumnKind.Timestamp, declared.Columns[0].Kind);
        Assert.Equal("volume", declared.Columns[5].Name);
        Assert.False(declared.Columns[5].Optional);
        Assert.True(declared.Columns[6].Optional);
        Assert.Equal(ColumnKind.Integer, declared.Columns[6].Kind);
    }

    [Fact]
    public void WhatHappensToTheRest_IsSaidRatherThanAssumed()
    {
        var dropped = Pdd.Create().ReadCsv("x.csv").Declare(schema => schema.Text("a"));
        var kept = Pdd.Create().ReadCsv("x.csv").Declare(schema => schema.Text("a"), Remainder.Keep);

        Assert.Equal(Remainder.Drop, ((DeclareStep)dropped.Declaration.Steps[1]).Remainder);
        Assert.Equal(Remainder.Keep, ((DeclareStep)kept.Declaration.Steps[1]).Remainder);
    }

    [Fact]
    public void EveryKindHasAWordForIt()
    {
        var builder = Pdd.Create().ReadCsv("x.csv").Declare(schema => schema
            .Text("name")
            .Number("fare")
            .Integer("siblings")
            .Boolean("alone")
            .Timestamp("when"));

        var declared = (DeclareStep)builder.Declaration.Steps[1];

        Assert.Equal(
            [ColumnKind.Text, ColumnKind.Number, ColumnKind.Integer, ColumnKind.Boolean, ColumnKind.Timestamp],
            declared.Columns.Select(column => column.Kind));
    }

    [Fact]
    public void ASchemaSurvivesTheFile()
    {
        var declaration = Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema
                .Integer("survived")
                .Text("sex")
                .Optional("age", ColumnKind.Number), Remainder.Keep)
            .Declaration;

        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn()));
    }

    [Fact]
    public void TheSameColumnTwice_IsRefused()
    {
        // Two declarations of one column cannot both be right, and the second silently winning is how a
        // column ends up typed one way in the schema and another way in everybody's head.
        var refused = Assert.Throws<ArgumentException>(
            () => Pdd.Create().ReadCsv("x.csv").Declare(schema => schema.Text("age").Number("age")));

        Assert.Contains("age", refused.Message);
    }

    [Fact]
    public void ASchemaWithoutColumns_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => Pdd.Create().ReadCsv("x.csv").Declare(_ => { }));
    }

    [Fact]
    public void AColumnWithoutAName_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => Pdd.Create().ReadCsv("x.csv").Declare(schema => schema.Text(" ")));
    }

    [Fact]
    public void ADeclarationWithNoSchemaAtAll_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => Pdd.Create().ReadCsv("x.csv").Declare(null!));
    }

    [Fact]
    public void AKindNobodyDefined_DoesNotSurviveTheFile()
    {
        const string json = """
            {"declaration":[{"step":"declare","remainder":"drop",
                             "columns":[{"name":"age","kind":"colour","optional":false}]}]}
            """;

        Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    [Fact]
    public void ARemainderPolicyNobodyDefined_DoesNotSurviveTheFileEither()
    {
        const string json = """
            {"declaration":[{"step":"declare","remainder":"maybe",
                             "columns":[{"name":"age","kind":"number","optional":false}]}]}
            """;

        Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    [Fact]
    public void AColumnDeclaredTwice_IsRefusedWithItsName()
    {
        // Two declarations of one column is a person having meant two different things, and whichever the
        // binding happened to keep would be right half the time.
        var refused = Assert.Throws<ArgumentException>(() => new DeclareStep([
            new ColumnDeclaration("age", ColumnKind.Number, true),
            new ColumnDeclaration("age", ColumnKind.Integer, false),
        ]));

        Assert.Contains("'age'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("twice", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoSchemasThatDisagreeAboutTheRest_AreNotTheSameSchema()
    {
        // What happens to the columns nobody named is part of what the schema says, so two schemas with
        // the same columns and different answers to that are two schemas.
        var columns = new[] { new ColumnDeclaration("age", ColumnKind.Number, true) };

        var drops = new DeclareStep(columns);
        var keeps = new DeclareStep(columns, Remainder.Keep);

        Assert.NotEqual(drops, keeps);
        Assert.False(drops.Equals(null));
        Assert.Equal(drops, new DeclareStep(columns));
    }

    // A schema that names a column and excludes it: its kind is kept, so taking it in again brings it back as it was.
    private static DeclareStep WithAnExcludedColumn(Remainder remainder = Remainder.Drop) => new(
    [
        new ColumnDeclaration("a", ColumnKind.Number, false),
        new ColumnDeclaration("b", ColumnKind.Integer, false) { Excluded = true },
    ], remainder);

    private static string Written(IPipelineStep step)
    {
        var buffer = new MemoryStream();

        using (var writer = new System.Text.Json.Utf8JsonWriter(buffer))
        {
            step.WriteTo(writer);
        }

        return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
    }

    [Fact]
    public void AnExcludedColumn_IsNamedByTheSchema_WithItsKind_AndTakesNoPart()
    {
        var schema = WithAnExcludedColumn();

        Assert.Equal(["a", "b"], schema.Columns.Select(column => column.Name));
        Assert.Equal(["a"], schema.Taking.Select(column => column.Name));
        Assert.Equal(ColumnKind.Integer, schema.Columns[1].Kind);
        Assert.True(schema.Columns[1].Excluded);
    }

    [Theory]
    [InlineData(Remainder.Drop)]
    [InlineData(Remainder.Keep)]
    [InlineData(Remainder.Refuse)]
    public void AnExcludedColumn_IsNotRead_NotKeptWithTheRest_AndNotUnexpected(Remainder remainder)
    {
        // Named, so what becomes of the rest of the file does not reach it; excluded, so nothing reads it.
        var table = WithAnExcludedColumn(remainder).Bind(new InMemoryRowSource(["a", "b"], [["1.5", "7"], ["2", "8"]]));

        Assert.Equal(["a"], table.Columns.Select(column => column.Name));
    }

    [Fact]
    public void AnExcludedColumn_MayBeAbsentFromTheSource()
    {
        var table = WithAnExcludedColumn().Bind(new InMemoryRowSource(["a"], [["1.5"]]));

        Assert.Equal(["a"], table.Columns.Select(column => column.Name));
    }

    [Theory]
    [InlineData(Remainder.Drop)]
    [InlineData(Remainder.Refuse)]
    public void ASchemaThatTakesNoColumn_IsRefused(Remainder remainder)
    {
        var refused = Assert.Throws<ArgumentException>(
            () => new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false) { Excluded = true }], remainder));

        Assert.Contains("excludes every column", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASchemaThatExcludesEveryColumnItNames_AndKeepsTheRest_TakesTheRest()
    {
        var schema = new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false) { Excluded = true }], Remainder.Keep);

        Assert.Empty(schema.Taking);
        Assert.Equal(["c"], schema.Bind(new InMemoryRowSource(["a", "c"], [["1", "x"]])).Columns.Select(column => column.Name));
    }

    [Fact]
    public void AnExcludedCategory_IsNoCategoryToEncode()
    {
        var schema = new DeclareStep([
            new ColumnDeclaration("sex", ColumnKind.Category, false),
            new ColumnDeclaration("embarked", ColumnKind.Category, false) { Excluded = true },
        ]);

        Assert.Equal(["sex"], schema.Categories);
    }

    [Fact]
    public void ACategory_RemembersTheKindItWas()
    {
        var schema = new DeclareStep([new ColumnDeclaration("pclass", ColumnKind.Category, false) { Was = ColumnKind.Integer }]);

        Assert.Equal(ColumnKind.Integer, schema.Columns[0].Was);
    }

    [Theory]
    // Only a category was something else before; and a category never was one before it became one.
    [InlineData(ColumnKind.Number, ColumnKind.Integer)]
    [InlineData(ColumnKind.Category, ColumnKind.Category)]
    public void AKindItWas_IsRefused_UnlessACategorySaysWhatItWasBefore(ColumnKind kind, ColumnKind was)
    {
        var refused = Assert.Throws<ArgumentException>(
            () => new DeclareStep([new ColumnDeclaration("pclass", kind, false) { Was = was }]));

        Assert.Contains("'pclass'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AKindItWas_ThatIsNoKind_IsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new DeclareStep([new ColumnDeclaration("pclass", ColumnKind.Category, false) { Was = (ColumnKind)7 }]));
    }

    [Fact]
    public void ExcludedAndWhatACategoryWas_SurviveTheFile_WrittenOnlyWhereTheySaySomething()
    {
        var declaration = new PipelineDeclaration([
            new ReadCsvStep("x.csv"),
            new DeclareStep([
                new ColumnDeclaration("a", ColumnKind.Number, false),
                new ColumnDeclaration("b", ColumnKind.Integer, true) { Excluded = true },
                new ColumnDeclaration("c", ColumnKind.Category, false) { Was = ColumnKind.Integer },
            ]),
        ]);

        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn()));
        Assert.Equal(
            """{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false},"""
            + """{"name":"b","kind":"integer","optional":true,"excluded":true},{"name":"c","kind":"category","optional":false,"was":"integer"}]}""",
            Written(declaration.Steps[1]));
    }

    [Fact]
    public void AColumnThatSaysItIsNotExcluded_IsWrittenBackAsAnyColumnIs()
    {
        var read = (DeclareStep)StepCatalog.BuiltIn().ReadStep(
            """{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false,"excluded":false}]}""");

        Assert.False(read.Columns[0].Excluded);
        Assert.Null(read.Columns[0].Was);
        Assert.Equal("""{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false}]}""", Written(read));
    }

    [Fact]
    public void TwoSchemasThatDifferOnlyInWhatTheyExclude_OrInWhatACategoryWas_AreTwoSchemas()
    {
        ColumnDeclaration a = new("a", ColumnKind.Number, false);
        ColumnDeclaration b = new("b", ColumnKind.Category, false);

        Assert.NotEqual(new DeclareStep([a, b]), new DeclareStep([a, b with { Excluded = true }]));
        Assert.NotEqual(new DeclareStep([a, b]), new DeclareStep([a, b with { Was = ColumnKind.Text }]));
        Assert.NotEqual(new DeclareStep([a, b]).GetHashCode(), new DeclareStep([a, b with { Excluded = true }]).GetHashCode());
        Assert.Equal(new DeclareStep([a, b with { Was = ColumnKind.Text }]), new DeclareStep([a, b with { Was = ColumnKind.Text }]));
    }
}
