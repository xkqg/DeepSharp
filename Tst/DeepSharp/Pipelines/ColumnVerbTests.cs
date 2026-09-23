// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The two things a person does to columns while looking at them — leave one out, and say one stands for a
/// group — as steps a file can hold. Neither was: a column nobody wanted could only be left out of the
/// schema, which fails the moment something reads it or made it, and encoding the categories was a word in
/// the chain that became two other steps and could not be written down as itself.
/// </summary>
public class ColumnVerbTests
{
    private static string Titanic => Repository.Data("titanic.csv");

    private static PipelineBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema
                .Integer("survived")
                .Category("pclass", "sex")
                .Optional("age", ColumnKind.Number));

    [Fact]
    public void DroppingTheMarkerColumn_KeepsItOutOfTheBatch()
    {
        // The column that says where a gap was is written always, and a caller who does not want it drops
        // it "like any other column, which is a verb they already have" — which they did not have.
        var batch = Passengers()
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Drop("age_was_missing")
            .EncodeCategories()
            .Target("survived")
            .Build()
            .Run()
            .Batch(Part.Train);

        Assert.DoesNotContain("age_was_missing", batch.FeatureNames);
        Assert.Contains("age", batch.FeatureNames);
    }

    [Fact]
    public void AColumnCanBeDroppedAboveTheSplitToo()
    {
        var prepared = Passengers().Drop("age").Build().Prepare();
        var run = Passengers().Drop("age").Build().Run();

        Assert.True(prepared.Has("age"), "Prepare reads the declared columns and stops there");
        Assert.False(run.Table.Has("age"));
    }

    [Fact]
    public void DroppingAColumnThatIsNotThere_IsRefusedWithItsName()
    {
        var refused = Assert.Throws<DeclarationException>(() => Passengers().Drop("deck"));

        Assert.Contains("'deck'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DroppingAColumnThatTurnsOutNotToBeThere_IsRefusedWhenTheRowsAreRead()
    {
        // A schema that keeps the rest leaves any column possible, and the rows then say which are there:
        // refused before any step runs.
        var refused = Assert.Throws<DeclarationException>(
            () => Pdd.Create()
                .ReadCsv(Titanic)
                .Declare(schema => schema.Integer("survived"), Remainder.Keep)
                .Drop("cabin")
                .Build()
                .Run());

        Assert.Contains("'cabin'", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DroppingAColumnNothingCouldSayIsThere_IsRefusedWhenTheStepRuns()
    {
        // A step from elsewhere that does not say what it leaves behind leaves any column possible, so neither
        // the declaration nor the rows can tell; the drop itself still says what it did not find.
        var refused = Assert.Throws<InvalidOperationException>(
            () => Pdd.Create()
                .ReadCsv(Titanic)
                .Declare(schema => schema.Integer("survived"))
                .Add(new ScaleByStep("survived", 1))
                .Drop("cabin")
                .Build()
                .Run());

        Assert.Contains("'drop.columns', reads 'cabin', which is not here", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DroppingAColumnATableDoesNotHold_SaysWhatItHolds()
    {
        var table = new Table([new Column<double>("a", ColumnKind.Number, [1.0])]);

        var refused = Assert.Throws<InvalidOperationException>(() => new DropColumnsStep(["b"]).DropFrom(table));

        Assert.Contains("'b' is not here to be left out", refused.Message, StringComparison.Ordinal);
        Assert.Contains("a", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DropColumns_SurvivesTheFile()
    {
        var declaration = Passengers().Drop("age", "sex").Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(declaration, returned);
        Assert.Equal(["age", "sex"], ((DropColumnsStep)returned.Steps[2]).Columns);
        Assert.Throws<ArgumentException>(() => new DropColumnsStep([]));
        Assert.Throws<ArgumentException>(() => new DropColumnsStep([" "]));
    }

    [Fact]
    public void TwoDropsThatDiffer_AreNotTheSameStep()
    {
        var one = new DropColumnsStep(["a", "b"]);

        Assert.Equal(one, new DropColumnsStep(["a", "b"]));
        Assert.Equal(one.GetHashCode(), new DropColumnsStep(["a", "b"]).GetHashCode());
        Assert.NotEqual(one, new DropColumnsStep(["b", "a"]));
        Assert.False(one.Equals(null));
        Assert.Throws<ArgumentNullException>(() => one.DropFrom(null!));
        Assert.Throws<ArgumentNullException>(() => new DropColumnsStep(null!));
    }

    [Fact]
    public void EncodeCategories_IsOneStepAndSurvivesTheFileAsOne()
    {
        var declaration = Passengers()
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories(As.Ordinal, Unseen.Refuse)
            .Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(declaration, returned);
        Assert.Equal("encode.categories", returned.Steps[^1].Verb);

        var step = (EncodeCategoriesStep)returned.Steps[^1];

        Assert.Equal(As.Ordinal, step.How);
        Assert.Equal(Unseen.Refuse, step.Unseen);
    }

    [Fact]
    public void MarkingAColumnACategory_IsEncodedWithoutAnotherLine()
    {
        // pclass was a number read as a number; said to be a category in the schema, the one step that
        // encodes categories now finds it there, and nothing further down had to change.
        var prepared = Passengers()
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Target("survived")
            .Build()
            .Run();

        var batch = prepared.Batch(Part.Train);

        Assert.Contains("pclass_1", batch.FeatureNames);
        Assert.Contains("sex_female", batch.FeatureNames);
        Assert.False(prepared.Table.Has("pclass"));
        Assert.Equal(["1", "2", "3"], prepared.Fitted.Values.Single(values => values.Lists.ContainsKey("pclass")).List("pclass"));
    }

    [Fact]
    public void EveryCategoryIsLearnedFromTheTrainingRowsAlone_AndReplayedAsItWasLearned()
    {
        var trained = Passengers()
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Build()
            .Run();

        var served = trained.Replay(new InMemoryRowSource(
            ["survived", "pclass", "sex", "age"], [["0", "3", "female", null]]));

        Assert.Equal(1, ((Column<double>)served["pclass_3"])[0]);
        Assert.Equal(1, ((Column<double>)served["sex_female"])[0]);
        Assert.Equal(0, ((Column<double>)served["sex_male"])[0]);
    }

    [Fact]
    public void EncodeCategoriesWithNothingDeclaredACategory_IsRefusedWhereItIsWritten()
    {
        var refused = Assert.Throws<DeclarationException>(
            () => Pdd.Create()
                .ReadCsv(Titanic)
                .Declare(schema => schema.Integer("survived"))
                .SplitAtRandom(0.70, 0.15)
                .EncodeCategories());

        Assert.Contains("no categories", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EncodeCategoriesThatFindsNoneWhenItRuns_SaysSo()
    {
        // After a step from elsewhere that does not say what it leaves behind, a category is possible, so the
        // declaration cannot say there is none; the encoder can, when it runs.
        var refused = Assert.Throws<InvalidOperationException>(
            () => Pdd.Create()
                .ReadCsv(Titanic)
                .Declare(schema => schema.Integer("survived"))
                .Add(new ScaleByStep("survived", 1))
                .SplitAtRandom(0.70, 0.15)
                .EncodeCategories()
                .Build()
                .Run());

        Assert.Contains("no categories", refused.Message, StringComparison.Ordinal);
    }
}
