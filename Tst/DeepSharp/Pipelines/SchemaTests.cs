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

        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson()));
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

        Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json));
    }

    [Fact]
    public void ARemainderPolicyNobodyDefined_DoesNotSurviveTheFileEither()
    {
        const string json = """
            {"declaration":[{"step":"declare","remainder":"maybe",
                             "columns":[{"name":"age","kind":"number","optional":false}]}]}
            """;

        Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json));
    }
}
