// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// An answer that is a set of labels: several columns, each nought or one on every row. A row may hold one of them —
/// which of several things it is — or any number of them, and the output says which it means.
/// </summary>
public class LabelsTests
{
    private static readonly string[] Animals = ["cat", "dog", "bird"];

    private static PreparedData Labelled(IReadOnlyList<IReadOnlyList<string?>> rows, int ones = 0) =>
        Pdd.Create()
            .Read(new InMemoryRowSource(["weight", "cat", "dog", "bird"], rows), "four animals")
            .Declare(schema => schema.Number("weight").Integer(Animals))
            .SplitAtRandom(0.50, seed: 3)
            .Labels(Animals, ones)
            .Build()
            .Run();

    private static Batch[] Handed(PreparedData prepared) => [.. new[] { Part.Train, Part.Test }.Select(part => prepared.Batch(part))];

    [Fact]
    public void SeveralLabels_AreOneAnswer_HandedOverAsNoughtsAndOnes()
    {
        var prepared = Labelled([["4", "1", "0", "0"], ["30", "0", "1", "0"], ["0.1", "0", "0", "1"], ["5", "1", "0", "0"]], ones: 1);

        var batch = prepared.Batch(Part.Train);

        Assert.Equal(Animals, batch.AnswerNames);
        Assert.Equal(["weight"], batch.FeatureNames);
        Assert.All(batch.Answers!, labels => Assert.Equal(1, labels.Sum()));
        Assert.All(batch.Answers!, labels => Assert.All(labels, label => Assert.True(label is 0 or 1)));
    }

    [Fact]
    public void ALabelThatIsNeitherNoughtNorOne_IsRefused()
    {
        var prepared = Labelled([["4", "2", "0", "0"], ["30", "0", "2", "0"], ["0.1", "0", "0", "2"], ["5", "2", "0", "0"]]);

        var refused = Assert.Throws<InvalidOperationException>(() => Handed(prepared));

        Assert.Contains("'target.labels'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("neither nought nor one", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARowHoldingMoreOnesThanTheOutputSays_IsRefused()
    {
        var prepared = Labelled([["4", "1", "1", "0"], ["30", "0", "1", "1"], ["0.1", "1", "0", "1"], ["5", "1", "1", "0"]], ones: 1);

        var refused = Assert.Throws<InvalidOperationException>(() => Handed(prepared));

        Assert.Contains("holds 2 ones", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WithoutSayingHowMany_ARowMayHoldAnyNumberOfOnes()
    {
        var prepared = Labelled([["4", "1", "1", "0"], ["30", "0", "0", "0"], ["0.1", "1", "1", "1"], ["5", "0", "1", "0"]]);

        Assert.Equal(4, Handed(prepared).Sum(batch => batch.RowCount));
    }

    [Fact]
    public void TrueAndFalse_AreLabelsToo()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["weight", "tame", "wild"], [["4", "true", "false"], ["30", "false", "true"]]), "two animals")
            .Declare(schema => schema.Number("weight").Boolean("tame", "wild"))
            .SplitAtRandom(0.50, seed: 3)
            .Labels(["tame", "wild"], ones: 1)
            .Build()
            .Run();

        Assert.All(Handed(prepared).SelectMany(batch => batch.Answers!), labels => Assert.Equal(1, labels.Sum()));
    }

    [Fact]
    public void LabelsAreHeldInAtLeastTwoColumns()
    {
        Assert.Throws<ArgumentException>(() => new LabelsStep(["cat"]));
        Assert.Throws<ArgumentNullException>(() => new LabelsStep(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LabelsStep(Animals, -1));
    }

    [Fact]
    public void LabelsAreWrittenDown_WithHowManyOnesARowHolds_OnlyWhenThatIsSaid()
    {
        var any = new LabelsStep(Animals);
        var one = new LabelsStep(Animals, 1);
        var declaration = new PipelineDeclaration(
        [
            new ReadCsvStep("animals.csv"),
            new DeclareStep([.. Animals.Select(name => new ColumnDeclaration(name, ColumnKind.Integer, Optional: false))]),
            new SplitAtRandomStep(new SplitShares(0.50, 0, 0.50), 3),
            one,
        ]);

        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn()));
        Assert.DoesNotContain("ones", new PipelineDeclaration([.. declaration.Steps.Take(3), any]).ToJson(), StringComparison.Ordinal);
        Assert.NotEqual(any, one);
        Assert.Equal(one, new LabelsStep(Animals, 1));
        Assert.Equal(one.GetHashCode(), new LabelsStep(Animals, 1).GetHashCode());
        Assert.Equal(Animals, one.Answers);
        Assert.Equal(1, one.Ones);
    }
}
