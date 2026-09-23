// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A pipeline written in C# and a pipeline written as a file have to reach equally far, and the only way to
/// keep that true is to prove it: build one, write it out, read it back, and require the two to be the same
/// declaration. Without this test the promise decays silently, one step at a time, as soon as a verb learns
/// something the file cannot say.
/// </summary>
public class DeclarationRoundTripTests
{
    private static PipelineDeclaration ADeclaration() =>
        Pdd.Create()
            .ReadCsv("btceur-1d.csv")
            .SplitByTime("timestamp", train: 0.70, validation: 0.15, test: 0.15)
            .FillMissing("trades", With.Mean)
            .Declaration;

    [Fact]
    public void ADeclarationWrittenOutAndReadBack_IsTheSameDeclaration()
    {
        var original = ADeclaration();

        var returned = PipelineDeclaration.FromJson(original.ToJson());

        Assert.Equal(original, returned);
    }

    [Fact]
    public void AnEmptyDeclaration_SurvivesTheTripToo()
    {
        var empty = Pdd.Create().Declaration;

        Assert.Equal(empty, PipelineDeclaration.FromJson(empty.ToJson()));
    }

    [Fact]
    public void EveryStepIsWrittenUnderTheKeyThatNamesIt()
    {
        // The verb is one key, always the same key. A shape that puts the verb in the key name instead
        // ({"read": "csv"}) forces a reader to guess which of the keys is the verb, and guessing is what a
        // declaration exists to remove.
        using var document = JsonDocument.Parse(ADeclaration().ToJson());

        var steps = document.RootElement.GetProperty("declaration");

        Assert.Equal(3, steps.GetArrayLength());
        Assert.Equal("read.csv", steps[0].GetProperty("step").GetString());
        Assert.Equal("btceur-1d.csv", steps[0].GetProperty("path").GetString());
        Assert.Equal("split.byTime", steps[1].GetProperty("step").GetString());
        Assert.Equal(0.70, steps[1].GetProperty("train").GetDouble());
        Assert.Equal("fill.missing", steps[2].GetProperty("step").GetString());
        Assert.Equal("mean", steps[2].GetProperty("with").GetString());
    }

    [Fact]
    public void AStepNobodyRegistered_RefusesToLoadAndSaysWhichOne()
    {
        // Skipping it quietly would produce a pipeline that runs, reports nothing wrong, and is not the
        // pipeline in the file.
        const string json = """{"declaration":[{"step":"read.avro","path":"x.avro"}]}""";

        var refused = Assert.Throws<NotSupportedException>(() => PipelineDeclaration.FromJson(json));

        Assert.Contains("read.avro", refused.Message);
    }

    [Fact]
    public void AStepWithoutAVerb_IsRefused()
    {
        const string json = """{"declaration":[{"path":"x.csv"}]}""";

        Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json));
    }

    [Fact]
    public void AFileWithoutADeclaration_IsRefused()
    {
        Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson("""{"fitted":{}}"""));
    }

    [Fact]
    public void AStepMissingOneOfItsParameters_IsRefusedByTheStepThatNeedsIt()
    {
        const string json = """{"declaration":[{"step":"split.byTime","column":"t","train":0.7}]}""";

        Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json));
    }

    [Fact]
    public void TwoDeclarationsThatSayDifferentThings_AreNotEqual()
    {
        var one = Pdd.Create().ReadCsv("a.csv").Declaration;
        var other = Pdd.Create().ReadCsv("b.csv").Declaration;

        Assert.NotEqual(one, other);
        Assert.NotEqual(one.GetHashCode(), other.GetHashCode());
    }

    [Fact]
    public void ADeclarationIsNotEqualToSomethingThatIsNotOne()
    {
        var declaration = ADeclaration();

        Assert.False(declaration.Equals(null));
        Assert.False(declaration.Equals("read.csv"));
    }

    [Fact]
    public void TheCatalogRefusesToLearnTheSameVerbTwice()
    {
        // One verb, one reader. A second registration that silently wins would make a file mean different
        // things depending on which packages happened to be present.
        var catalog = StepCatalog.BuiltIn();

        var refused = Assert.Throws<InvalidOperationException>(
            () => catalog.Register("read.csv", element => ReadCsvStep.ReadFrom(element)));

        Assert.Contains("read.csv", refused.Message);
    }

    [Fact]
    public void ACatalogWithAnExtraVerb_ReadsAFileTheBuiltInOneCannot()
    {
        const string json = """{"declaration":[{"step":"read.avro","path":"x.avro"}]}""";
        var catalog = StepCatalog.BuiltIn();
        catalog.Register("read.avro", element => new ReadCsvStep(element.GetProperty("path").GetString()!));

        var declaration = PipelineDeclaration.FromJson(json, catalog);

        Assert.Single(declaration.Steps);
    }
}
