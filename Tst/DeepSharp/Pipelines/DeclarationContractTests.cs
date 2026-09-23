// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a step has to be, and what a declaration has to promise, for the round trip to keep meaning what it
/// says once verbs start arriving from other packages. Each of these was a hole a council walked through:
/// a step that compares by reference, a verb written under one name and read under another, a fault from a
/// file arriving as the wrong kind of exception.
/// </summary>
public class DeclarationContractTests
{
    /// <summary>A step written the way a package author would write one, not the way the core does.</summary>
    private sealed record ScaleStep(string Column, double By) : IPipelineStep
    {
        public string Verb => "scale.by";

        public void WriteTo(Utf8JsonWriter writer)
        {
            writer.WriteStartObject();
            writer.WriteString("step", Verb);
            writer.WriteString("column", Column);
            writer.WriteNumber("by", By);
            writer.WriteEndObject();
        }

        public static ScaleStep ReadFrom(JsonElement element) =>
            new(element.GetProperty("column").GetString()!, element.GetProperty("by").GetDouble());
    }

    [Fact]
    public void AStepFromAnotherPackage_SurvivesTheRoundTripAndComparesEqual()
    {
        // This test is also the shape a package author copies, which is why it uses nothing internal.
        var catalog = StepCatalog.BuiltIn();
        catalog.Register("scale.by", ScaleStep.ReadFrom);

        var original = new PipelineDeclaration([new ReadCsvStep("x.csv"), new ScaleStep("fare", 0.5)]);

        var returned = PipelineDeclaration.FromJson(original.ToJson(), catalog);

        Assert.Equal(original, returned);
    }

    [Fact]
    public void EqualDeclarations_HaveEqualHashCodes()
    {
        // The contractual direction, and the one a dictionary depends on. The opposite — two unequal
        // declarations having unequal hashes — is not promised by anything and was asserted here by
        // mistake, passing only because these two happened not to collide.
        var one = Pdd.Create().ReadCsv("a.csv").SplitByTime("t", 0.70, 0.15, 0.15).Declaration;
        var other = Pdd.Create().ReadCsv("a.csv").SplitByTime("t", 0.70, 0.15, 0.15).Declaration;

        Assert.Equal(one, other);
        Assert.Equal(one.GetHashCode(), other.GetHashCode());
    }

    [Fact]
    public void TwoDeclarationsCompareTheSameWayWithEitherSpelling()
    {
        // Every step is a record and so has a working `==`. A declaration that answered differently to the
        // operator than to the method is the kind of trap nobody looks for twice.
        var one = Pdd.Create().ReadCsv("a.csv").Declaration;
        var same = Pdd.Create().ReadCsv("a.csv").Declaration;
        var other = Pdd.Create().ReadCsv("b.csv").Declaration;

        Assert.True(one == same);
        Assert.False(one != same);
        Assert.True(one != other);
        Assert.False(null! == one);
        Assert.True((PipelineDeclaration?)null == (PipelineDeclaration?)null);
    }

    [Fact]
    public void EveryStepThisLibraryShips_IsOneTheBuiltInCatalogCanReadBack()
    {
        // A verb is written in the step and read in the catalog, so the two can drift, and a forgotten
        // registration breaks the round trip for that verb alone while every existing test stays green.
        var verbs = typeof(Pdd).Assembly.GetTypes()
            .Where(type => type.IsClass && !type.IsAbstract)
            .Where(type => type.GetInterfaces().Any(
                face => face.IsGenericType && face.GetGenericTypeDefinition() == typeof(IPipelineStep<>)))
            .Select(type => (Verb: (string)type.GetProperty(
                "Name", BindingFlags.Public | BindingFlags.Static)!.GetValue(null)!, Type: type))
            .ToArray();

        var catalog = StepCatalog.BuiltIn();
        var unknown = verbs.Where(step => !catalog.Knows(step.Verb))
                           .Select(step => $"{step.Type.Name} ({step.Verb})").ToArray();

        Assert.NotEmpty(verbs);
        Assert.True(unknown.Length == 0, $"The built-in catalog cannot read back: {string.Join(", ", unknown)}");
    }

    [Fact]
    public void AFaultInsideOneStep_ReachesTheCallerAsAFaultInTheFile()
    {
        // The three shares adding to 0.95 is the flagship message of the validator, and it used to arrive
        // as an ArgumentException naming a C# parameter — which a file-loading boundary does not catch.
        const string json = """
            {"declaration":[{"step":"split.byTime","column":"t","train":0.7,"validation":0.15,"test":0.10}]}
            """;

        var refused = Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json));

        Assert.Contains("split.byTime", refused.Message);
        Assert.Contains("0.95", refused.InnerException!.Message);
    }

    [Fact]
    public void AStrategyNobodyDefined_DoesNotSurviveTheFile()
    {
        // An unknown verb refuses loudly; an unknown strategy used to load and mean whatever the eventual
        // executor's default branch means.
        const string json = """
            {"declaration":[{"step":"split.byTime","column":"t","train":0.7,"validation":0.15,"test":0.15},
                            {"step":"fill.missing","column":"age","with":"next"}]}
            """;

        var refused = Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json));

        Assert.Contains("next", refused.InnerException!.Message);
    }

    [Fact]
    public void AStrategyThatWasNeverGivenAName_IsRefusedWhereItIsWritten()
    {
        var after = Pdd.Create().ReadCsv("x.csv").SplitByTime("t", 0.70, 0.15, 0.15);

        Assert.Throws<ArgumentException>(() => after.FillMissing("age", default));
    }

    [Theory]
    // A strategy that takes a number, written without one — and one that takes none, written with one.
    // Both are reachable from a hand-written file, and both would otherwise reach an executor that has to
    // guess what was meant.
    [InlineData("constant", null)]
    [InlineData("mean", 3.0)]
    public void AStrategyWrittenWithTheWrongNumberOfNumbers_IsRefused(string name, double? value)
    {
        var after = Pdd.Create().ReadCsv("x.csv").SplitByTime("t", 0.70, 0.15, 0.15);

        var refused = Assert.Throws<ArgumentException>(
            () => after.FillMissing("age", new FillStrategy(name, value)));

        Assert.Contains(name, refused.Message);
    }

    [Theory]
    [InlineData("""{"step":"fill.missing","column":"age"}""")]
    [InlineData("""{"step":"fill.missing","column":"age","with":7}""")]
    public void AFillWhoseStrategyIsMissingOrIsNotEvenAName_IsRefused(string step)
    {
        var json = $$"""
            {"declaration":[{"step":"split.byTime","column":"t","train":0.7,"validation":0.15,"test":0.15},
                            {{step}}]}
            """;

        Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json));
    }

    [Fact]
    public void AStrategySaysWhatItIs()
    {
        // It ends up in refusal messages, where "constant" without its number would leave the reader
        // guessing which constant.
        Assert.Equal("mean", With.Mean.ToString());
        Assert.Equal("constant(-1)", With.Constant(-1).ToString());
        Assert.Equal(string.Empty, default(FillStrategy).ToString());
    }

    [Fact]
    public void AConstantToFillWith_CarriesItsValueThroughTheFile()
    {
        // A strategy with a number in it is the case that would have broken the file format later: the
        // written form has room for it now, while nothing has shipped.
        var declaration = Pdd.Create()
            .ReadCsv("x.csv")
            .SplitByTime("t", 0.70, 0.15, 0.15)
            .FillMissing("age", With.Constant(-1))
            .Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson());

        Assert.Equal(declaration, returned);
        Assert.Equal(-1, ((FillMissingStep)returned.Steps[2]).Strategy.Value);
    }
}
