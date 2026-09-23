// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Which columns stand for a group rather than for themselves is a fact about the data, so it is said where
/// the data is declared and not again further down. Everything after reads it from there: encoding takes
/// them by name, and a category that never became numbers is refused at the handover rather than dropped.
/// </summary>
public class CategoryTests
{
    private static string Titanic => Path.Join(RepoRoot(), "Samples", "data", "titanic.csv");

    private static PipelineBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema
                .Integer("survived", "pclass")
                .Number("fare")
                .Category("sex", "embarked"));

    [Fact]
    public void AColumnSaidToBeACategory_IsOneFromTheMomentItIsRead()
    {
        var table = Passengers().Build().Prepare();

        Assert.Equal(ColumnKind.Category, table["sex"].Kind);
        Assert.Equal(ColumnKind.Category, table["embarked"].Kind);
        Assert.Equal(ColumnKind.Number, table["fare"].Kind);
        Assert.Equal("male", ((TextColumn)table["sex"])[0]);
    }

    [Fact]
    public void TheSchemaKnowsWhichOnesTheyWere()
    {
        var declared = (DeclareStep)Passengers().Declaration.Steps[1];

        Assert.Equal(["sex", "embarked"], declared.Categories);
    }

    [Fact]
    public void EncodingTakesThemByNameRatherThanAskingAgain()
    {
        var prepared = Passengers()
            .SplitStratified("survived", 0.70, 0.15, 0.15)
            .EncodeCategories()
            .Build()
            .Run();

        Assert.False(prepared.Table.Has("sex"));
        Assert.False(prepared.Table.Has("embarked"));
        Assert.True(prepared.Table.Has("sex_female"));
        Assert.True(prepared.Table.Has("embarked_S"));

        // Two steps written from one word, in the order the schema declared them.
        Assert.Equal(["encode", "encode"], prepared.Declaration.Steps.TakeLast(2).Select(step => step.Verb));
    }

    [Fact]
    public void AndTheResultIsHandedOverAsNumbers()
    {
        var batch = Passengers()
            .SplitStratified("survived", 0.70, 0.15, 0.15)
            .EncodeCategories(As.Ordinal)
            .Target("survived")
            .Build()
            .Run()
            .Batch(Split.Train);

        Assert.Contains("sex", batch.FeatureNames);
        Assert.Equal(batch.Width, batch.Features[0].Length);
    }

    [Fact]
    public void ACategoryNobodyEncoded_IsRefusedAtTheHandoverRatherThanDropped()
    {
        var refused = Assert.Throws<InvalidOperationException>(
            () => Passengers()
                .SplitStratified("survived", 0.70, 0.15, 0.15)
                .Build()
                .Run()
                .Batch(Split.Train));

        Assert.Contains("sex", refused.Message);
        Assert.Contains("Encode it", refused.Message);
    }

    [Fact]
    public void ASchemaWithoutCategories_SaysSoRatherThanQuietlyDoingNothing()
    {
        var refused = Assert.Throws<InvalidOperationException>(
            () => Pdd.Create()
                .ReadCsv(Titanic)
                .Declare(schema => schema.Integer("survived"))
                .SplitAtRandom(0.70, 0.15, 0.15)
                .EncodeCategories());

        Assert.Contains("no categories", refused.Message);
    }

    [Fact]
    public void ACategorySurvivesTheFile()
    {
        var declaration = Passengers().Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson());

        Assert.Equal(declaration, returned);
        Assert.Equal(["sex", "embarked"], ((DeclareStep)returned.Steps[1]).Categories);
    }

    [Fact]
    public void AColumnOfWordsIsTextOrACategory_AndNothingElse()
    {
        Assert.Throws<ArgumentException>(() => new TextColumn("a", ColumnKind.Number, ["x"]));
        Assert.Equal(ColumnKind.Text, new TextColumn("a", ["x"]).Kind);
    }

    private static string RepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);

        while (directory is not null && !File.Exists(Path.Join(directory.FullName, "DeepSharp.slnx")))
        {
            directory = directory.Parent;
        }

        Assert.NotNull(directory);
        return directory!.FullName;
    }
}
