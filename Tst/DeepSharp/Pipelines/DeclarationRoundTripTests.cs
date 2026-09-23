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
            .Declare(schema => schema.Timestamp("timestamp").Number("trades"))
            .SplitByTime("timestamp", train: 0.70, validation: 0.15)
            .FillMissing("trades", With.Mean)
            .Declaration;

    [Fact]
    public void ADeclarationWrittenOutAndReadBack_IsTheSameDeclaration()
    {
        var original = ADeclaration();

        var returned = PipelineDeclaration.FromJson(original.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(original, returned);
    }

    [Fact]
    public void AnEmptyDeclaration_SurvivesTheTripToo()
    {
        var empty = Pdd.Create().Declaration;

        Assert.Equal(empty, PipelineDeclaration.FromJson(empty.ToJson(), StepCatalog.BuiltIn()));
    }

    [Fact]
    public void EveryStepIsWrittenUnderTheKeyThatNamesIt()
    {
        // The verb is one key, always the same key. A shape that puts the verb in the key name instead
        // ({"read": "csv"}) forces a reader to guess which of the keys is the verb, and guessing is what a
        // declaration exists to remove.
        using var document = JsonDocument.Parse(ADeclaration().ToJson());

        var steps = document.RootElement.GetProperty("declaration");

        Assert.Equal(4, steps.GetArrayLength());
        Assert.Equal("read.csv", steps[0].GetProperty("step").GetString());
        Assert.Equal("btceur-1d.csv", steps[0].GetProperty("path").GetString());
        Assert.Equal("declare", steps[1].GetProperty("step").GetString());
        Assert.Equal("split.byTime", steps[2].GetProperty("step").GetString());
        Assert.Equal(0.70, steps[2].GetProperty("train").GetDouble());
        Assert.Equal("fill.missing", steps[3].GetProperty("step").GetString());
        Assert.Equal("mean", steps[3].GetProperty("with").GetString());
    }

    [Fact]
    public void AStepNobodyRegistered_RefusesToLoadAndSaysWhichOne()
    {
        // Skipping it quietly would produce a pipeline that runs, reports nothing wrong, and is not the
        // pipeline in the file.
        const string json = """{"declaration":[{"step":"read.avro","path":"x.avro"}]}""";

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Contains("read.avro", refused.Message);
    }

    [Fact]
    public void AMisspelledStepAndAStepFromAPackageNotRegistered_AreTwoDifferentFaults()
    {
        // "I do not know this step" and "I know it, but its package is not here" are two different problems
        // for whoever reads them, and the one message that said "either" left them to guess which.
        var misspelled = Assert.Throws<PipelineFileException>(
            () => PipelineDeclaration.FromJson("""{"declaration":[{"step":"read.cvs","path":"x.csv"}]}""", StepCatalog.BuiltIn()));

        var elsewhere = Assert.Throws<PipelineFileException>(
            () => PipelineDeclaration.FromJson(
                """{"declaration":[{"step":"feature.indicator","column":"x","indicator":"sma","period":5,"columns":["a"]}]}""", StepCatalog.BuiltIn()));

        Assert.Contains("'read.cvs'", misspelled.Message, StringComparison.Ordinal);
        Assert.Contains("'read.csv'", misspelled.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("package", misspelled.Message, StringComparison.Ordinal);

        Assert.Contains("DeepSharp.Pipelines.Indicators", elsewhere.Message, StringComparison.Ordinal);
        Assert.Equal("DeepSharp.Pipelines.Indicators", StepCatalog.PackageThatBrings("feature.indicator"));
        Assert.Null(StepCatalog.PackageThatBrings("read.cvs"));

        // A catalog that knows nothing has nothing to suggest.
        using var document = System.Text.Json.JsonDocument.Parse("""{"step":"read.cvs","path":"x.csv"}""");
        var empty = Assert.Throws<NotSupportedException>(() => new StepCatalog().Read(document.RootElement));

        Assert.DoesNotContain("nearest", empty.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryVerbAnotherPackageBrings_IsKnownToComeFromThatPackage()
    {
        // Both ways round: a satellite's verb that the core does not know comes from it would be reported as
        // a misspelling, and a verb the core says comes from a package that no longer brings it would send
        // somebody to install the wrong thing.
        var core = StepCatalog.BuiltIn();
        var everything = StepCatalog.BuiltIn();
        new IndicatorSteps().AddTo(everything);

        var brought = everything.Descriptions.Select(description => description.Verb)
            .Where(verb => !core.Knows(verb))
            .ToArray();

        Assert.NotEmpty(brought);
        Assert.All(brought, verb => Assert.Equal(typeof(IndicatorSteps).Assembly.GetName().Name, StepCatalog.PackageThatBrings(verb)));
        Assert.Equal(brought.Order(StringComparer.Ordinal), StepCatalog.VerbsOtherPackagesBring.Keys.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void AStepWithoutAVerb_IsRefused()
    {
        const string json = """{"declaration":[{"path":"x.csv"}]}""";

        Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    [Fact]
    public void AFileWithoutADeclaration_IsRefused()
    {
        Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson("""{"fitted":{}}""", StepCatalog.BuiltIn()));
    }

    [Fact]
    public void AStepMissingOneOfItsParameters_IsRefusedByTheStepThatNeedsIt()
    {
        const string json = """{"version":2,"declaration":[{"step":"split.byTime","column":"t","train":0.7}]}""";

        Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    [Fact]
    public void TwoDeclarationsThatSayDifferentThings_AreNotEqual()
    {
        var one = Pdd.Create().ReadCsv("a.csv").Declaration;
        var other = Pdd.Create().ReadCsv("b.csv").Declaration;

        Assert.NotEqual(one, other);
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

        var refused = Assert.Throws<InvalidOperationException>(() => catalog.Register<ReadCsvStep>());

        Assert.Contains("read.csv", refused.Message);
    }

    [Fact]
    public void AStepTypeWhoseStepsAnswerToAnotherVerb_IsRefusedWhenItIsRead()
    {
        // The verb is named by the type and answered by the step, so the two can disagree. When they do, a
        // file loads under one name and writes itself back under a different one, and the round trip still
        // reports the declarations equal because it compares declarations, never documents.
        const string json = """{"declaration":[{"step":"read.parquet"}]}""";
        var catalog = StepCatalog.BuiltIn();
        catalog.Register<MisnamedStep>();

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, catalog));

        Assert.Contains("read.parquet", refused.Message);
        Assert.Contains("read.csv", refused.Message);
    }
}
