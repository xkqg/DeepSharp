// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The decisions a pipeline made about its columns, saved as a file of their own: the schema, the columns dropped
/// after the steps that read them, the output, and the source's columns as they were last shown. It is read through
/// the same door as a pipeline file, with the same care: every fault at once, each at its line and column, and a
/// file from a newer version refused whole.
/// </summary>
public class PipelinePresetTests
{
    private static readonly string[] Header =
        ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "class", "who", "adult_male", "deck", "embark_town", "alive", "alone"];

    private const string Schema = """{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false}]}""";

    // A pipeline that decided something of every kind: a category that remembers what it was, an excluded column, a
    // drop after the step that made its column, and an output.
    private static PipelineDeclaration Decided()
    {
        var steps = Pdd.Create()
            .ReadCsv("titanic.csv")
            .Declare(schema => schema.Integer("survived", "pclass").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Drop("age_was_missing")
            .Target("survived")
            .Declaration.Steps.ToList();

        steps[1] = new DeclareStep([
            new ColumnDeclaration("survived", ColumnKind.Integer, false),
            new ColumnDeclaration("pclass", ColumnKind.Category, false) { Was = ColumnKind.Integer },
            new ColumnDeclaration("age", ColumnKind.Number, true),
            new ColumnDeclaration("fare", ColumnKind.Number, false) { Excluded = true },
        ]);

        return new PipelineDeclaration(steps);
    }

    private static IReadOnlyList<PipelineFileFault> Refused(string json) =>
        Assert.Throws<PipelineFileException>(() => PipelinePreset.FromJson(json, StepCatalog.BuiltIn())).Faults;

    private static IEnumerable<string> RootKeys(string json)
    {
        using var document = JsonDocument.Parse(json);

        return [.. document.RootElement.EnumerateObject().Select(property => property.Name)];
    }

    [Fact]
    public void APreset_SurvivesTheFile_WithEverythingItHolds()
    {
        var preset = PipelinePreset.Of(Decided(), Header);

        var read = PipelinePreset.FromJson(preset.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(preset, read);
        Assert.Equal(Header, read.Source);
        Assert.Equal(["age_was_missing"], read.Drop);
        Assert.Equal(new TargetStep("survived"), read.Output);
        Assert.True(read.Declare.Columns[3].Excluded);
        Assert.Equal(ColumnKind.Integer, read.Declare.Columns[1].Was);
    }

    [Fact]
    public void APreset_IsWhatTheBlocksSay()
    {
        var declaration = Decided();

        var preset = PipelinePreset.Of(declaration, header: null);

        Assert.Equal(declaration.Steps[1], preset.Declare);
        Assert.Equal(["age_was_missing"], preset.Drop);
        Assert.Equal(declaration.Output, preset.Output);
        Assert.Null(preset.Source);
    }

    [Fact]
    public void APreset_WritesOnlyWhatItHolds_WithALineFeedEveryWhere()
    {
        var bare = new PipelinePreset(new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]));
        var full = PipelinePreset.Of(Decided(), Header);

        Assert.Equal(["version", "declare"], RootKeys(bare.ToJson()));
        Assert.Equal(["version", "source", "declare", "drop", "output"], RootKeys(full.ToJson()));
        Assert.Contains("\"version\": 2", bare.ToJson(), StringComparison.Ordinal);
        Assert.DoesNotContain("\r", full.ToJson(), StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineWithoutASchema_HasNoPreset()
    {
        Assert.Throws<ArgumentException>(() => PipelinePreset.Of(new PipelineDeclaration([new ReadCsvStep("x.csv")]), null));
    }

    [Fact]
    public void ADropThatNamesAColumnTwice_IsRefused()
    {
        var refused = Assert.Throws<ArgumentException>(
            () => new PipelinePreset(new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]), drop: ["b", "b"]));

        Assert.Contains("'b'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADroppedColumnWithoutAName_IsRefused()
    {
        var refused = Assert.Throws<ArgumentException>(
            () => new PipelinePreset(new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]), drop: [" "]));

        Assert.Contains("needs a name", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoPresets_AreEqual_WhenTheySayTheSameThing()
    {
        var one = PipelinePreset.Of(Decided(), Header);

        Assert.Equal(one, PipelinePreset.Of(Decided(), Header));
        Assert.Equal(one.GetHashCode(), PipelinePreset.Of(Decided(), Header).GetHashCode());
        Assert.NotEqual(one, PipelinePreset.Of(Decided(), null));
        Assert.NotEqual(PipelinePreset.Of(Decided(), null), one);
        Assert.Equal(PipelinePreset.Of(Decided(), null).GetHashCode(), PipelinePreset.Of(Decided(), null).GetHashCode());
        Assert.NotEqual(one, new PipelinePreset(one.Declare, one.Drop, output: null, one.Source));
        Assert.NotEqual(one, new PipelinePreset(one.Declare, drop: [], one.Output, one.Source));
        Assert.NotEqual(one, new PipelinePreset(one.Declare, one.Drop, one.Output, source: ["survived"]));
        Assert.False(one.Equals(null));
    }

    [Fact]
    public void APresetThatNamesNoVersion_IsRefused()
    {
        var fault = Assert.Single(Refused($$"""{"declare":{{Schema}}}"""));

        Assert.Contains("names the version", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APresetFromANewerVersion_IsRefusedWhole()
    {
        var fault = Assert.Single(Refused($$"""{"version":3,"colour":"red","declare":{{Schema}}}"""));

        Assert.Contains("newer DeepSharp", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AVersionThatIsNoWholeNumber_IsRefused()
    {
        Assert.Contains("is not one", Assert.Single(Refused($$"""{"version":"two","declare":{{Schema}}}""")).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AKeyAPresetDoesNotHave_IsRefused_NamingTheOnesItHas()
    {
        var fault = Assert.Single(Refused($$"""{"version":2,"declare":{{Schema}},"colour":"red"}"""));

        Assert.Contains("'colour'", fault.Message, StringComparison.Ordinal);
        Assert.Contains("version, source, declare, drop, output", fault.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("[1, 2]", "one JSON object")]
    [InlineData("""{"version": 2,""", "stops being JSON")]
    [InlineData("""{"version":2,"version":2,"declare":{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false}]}}""", "written twice")]
    [InlineData("""{"version":2}""", "under 'declare'")]
    [InlineData("""{"version":2,"declare":{"step":"drop.columns","columns":["a"]}}""", "'drop.columns'")]
    [InlineData("""{"version":2,"declare":{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"colour","optional":false}]}}""", "'colour'")]
    [InlineData("""{"version":2,"declare":"declare"}""", "under 'declare'")]
    [InlineData("""{"version":2,"declare":{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false}]},"output":{"step":"drop.columns","columns":["a"]}}""", "names no answer")]
    [InlineData("""{"version":2,"declare":{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false}]},"output":{"step":"feature.indicator","name":"x"}}""", "DeepSharp.Pipelines.Indicators")]
    [InlineData("""{"version":2,"declare":{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false}]},"drop":[1]}""", "'drop' is a list of column names")]
    [InlineData("""{"version":2,"declare":{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false}]},"drop":["b","b"]}""", "'b' twice")]
    [InlineData("""{"version":2,"declare":{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false}]},"source":"a"}""", "'source' is a list of column names")]
    public void AFileThatIsNoPreset_IsRefused_SayingWhy(string json, string why)
    {
        Assert.Contains(Refused(json), fault => fault.Message.Contains(why, StringComparison.Ordinal));
    }

    [Fact]
    public void EveryFault_IsNamedAtOnce_AtItsLineAndColumn()
    {
        var faults = Refused($$"""
            {
              "version": 2,
              "colour": "red",
              "declare": {{Schema}},
              "drop": [1]
            }
            """);

        Assert.Equal([3, 5], faults.Select(fault => fault.Line));
        Assert.All(faults, fault => Assert.Equal(3, fault.Column));
    }
}
