// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using Corvus.Json;
using DeepSharp.Pipelines;
using Schema = Corvus.Json.Validator.JsonSchema;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The steps are the description, and everything a person meets a pipeline through is projected from them:
/// the JSON Schema an editor checks a file against as it is typed, and the reference page that says what
/// every verb takes. Both are written by the catalog and committed beside the code, and these tests fail the
/// moment either stops being what the steps say — which is the only way a document about code stays true.
/// </summary>
public class ProjectionTests
{
    // Set to 1 to write the committed copies from what the steps describe now, after a change to a step.
    private const string Regenerate = "DEEPSHARP_REGEN_GOLDEN";

    private static readonly Lazy<Schema> TheSchema = new(() =>
        Schema.FromText(Everything().JsonSchema(), "https://github.com/xkqg/DeepSharp/pipeline.schema.json"));

    private static StepCatalog Everything() => StepCatalog.BuiltIn().WithIndicators();

    private static bool Valid(string json)
    {
        using var document = JsonDocument.Parse(json);

        return TheSchema.Value.Validate(document.RootElement, ValidationLevel.Flag).IsValid;
    }

    public static TheoryData<string> EveryVerb()
    {
        var verbs = new TheoryData<string>();

        foreach (var description in Everything().Descriptions)
        {
            verbs.Add(description.Verb);
        }

        return verbs;
    }

    [Theory]
    [MemberData(nameof(EveryVerb))]
    public void TheSchema_AcceptsTheStepEveryNewBlockStartsWith(string verb)
    {
        Assert.True(Valid($$"""{"version":2,"declaration":[{{Everything().Describe(verb).Template}}]}"""), verb);
    }

    [Theory]
    [MemberData(nameof(EveryVerb))]
    public void TheSchema_HoldsAFileThatNamesNoVersionToTheFirst_AsTheReaderDoes(string verb)
    {
        // A verb whose meaning changed after the first version is refused from a file written before, by the
        // schema an editor checks it against as much as by the reader.
        var description = Everything().Describe(verb);
        using var step = JsonDocument.Parse(description.Template);
        var fromTheFirst = Record.Exception(() => Everything().Read(step.RootElement, writtenAgainst: 1));

        Assert.Equal(description.Since == 1, Valid($$"""{"declaration":[{{description.Template}}]}"""));
        Assert.Equal(description.Since == 1, Valid($$"""{"version":1,"declaration":[{{description.Template}}]}"""));
        Assert.Equal(description.Since == 1, fromTheFirst is null);
    }

    [Fact]
    public void TheSchema_AcceptsAWholePipelineTheLibraryWrote_WithWhatItLearned()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass").Category("sex").Optional("age", ColumnKind.Number))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Constant(-1))
            .Normalise("age", Scale.Quantile)
            .EncodeCategories()
            .Target("survived")
            .Build()
            .Run();

        Assert.True(Valid(prepared.Declaration.ToJson()));
        Assert.True(Valid(prepared.ToJson()));
    }

    [Theory]
    // Each of these is refused by the reader, and an editor checking the file against the schema has to say
    // so too: a schema more forgiving than the reader lets a file look right that will never load.
    [InlineData("""{"declaration":[{"step":"read.csv","path":"x.csv","colour":"red"}]}""")]
    [InlineData("""{"declaration":[{"step":"read.cvs","path":"x.csv"}]}""")]
    [InlineData("""{"declaration":[{"step":"read.csv"}]}""")]
    [InlineData("""{"declaration":[{"step":"read.csv","path":"  "}]}""")]
    [InlineData("""{"declaration":[{"path":"x.csv"}]}""")]
    [InlineData("""{"version":2,"declaration":[{"step":"split.byTime","column":"t","train":"0.7","validation":0.15,"test":0.15}]}""")]
    [InlineData("""{"version":2,"declaration":[{"step":"split.atRandom","train":0.7,"validation":0.15,"test":0.15,"seed":3.5}]}""")]
    [InlineData("""{"version":2,"declaration":[{"step":"split.atRandom","train":0.7,"validation":0.15,"test":0.15,"seed":1e10}]}""")]
    [InlineData("""{"version":2,"declaration":[{"step":"split.atRandom","train":0,"validation":0.15,"test":0.85,"seed":1}]}""")]
    [InlineData("""{"version":2,"declaration":[{"step":"drop.warmup","atMost":-1}]}""")]
    [InlineData("""{"declaration":[{"step":"outliers.clip","column":"a","bounds":"iqr","at":0,"outlier":"clip"}]}""")]
    [InlineData("""{"declaration":[{"step":"normalise","column":"a","scale":"sideways","outOfRange":"pass"}]}""")]
    [InlineData("""{"declaration":[{"step":"normalise","column":"a","scale":"1","outOfRange":"pass"}]}""")]
    [InlineData("""{"declaration":[{"step":"fill.nan","column":"a","with":"previous"}]}""")]
    [InlineData("""{"declaration":[{"step":"fill.missing","column":"a","with":"constant"}]}""")]
    [InlineData("""{"declaration":[{"step":"fill.missing","column":"a","with":{"kind":"mean","value":1}}]}""")]
    [InlineData("""{"declaration":[{"step":"fill.missing","column":"a","with":{"kind":"constant","value":1,"colour":"red"}}]}""")]
    [InlineData("""{"declaration":[{"step":"declare","remainder":"drop","columns":[]}]}""")]
    [InlineData("""{"declaration":[{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number"}]}]}""")]
    [InlineData("""{"declaration":[{"step":"feature.timeParts","column":"t","asCategories":true,"parts":[]}]}""")]
    [InlineData("""{"version":2,"declaration":[{"step":"drop.columns","columns":["a", 3]}]}""")]
    [InlineData("""{"version":2,"declaration":[{"step":"drop.columns","columns":["a","b","a"]}]}""")]
    [InlineData("""{"declaration":[],"colour":"red"}""")]
    [InlineData("""{"fitted":{}}""")]
    [InlineData("""{"version":3,"declaration":[]}""")]
    [InlineData("""{"version":0,"declaration":[]}""")]
    [InlineData("""{"version":"2","declaration":[]}""")]
    [InlineData("""{"version":1.5,"declaration":[]}""")]
    [InlineData("""{"declaration":[{"step":"drop.columns","columns":["a"]}]}""")]
    [InlineData("""{"version":1,"declaration":[{"step":"order.by","columns":["a"]}]}""")]
    [InlineData("""{"version":2,"declaration":[],"fitted":{"3":{"value":1}}}""")]
    [InlineData("""{"version":2,"declaration":[],"fitted":[{"step":"normalise","prefix":"ab","learned":{}}]}""")]
    [InlineData("""{"version":2,"declaration":[],"fitted":[{"step":"normalise","prefix":"0000000000000000000000000000000000000000000000000000000000000000"}]}""")]
    [InlineData("""{"version":2,"declaration":[],"fitted":[{"step":"normalise","prefix":"0000000000000000000000000000000000000000000000000000000000000000","learned":{"a":true}}]}""")]
    [InlineData("""{"version":2,"declaration":[],"fitted":[{"step":"normalise","prefix":"0000000000000000000000000000000000000000000000000000000000000000","learned":{"a":[1,"b"]}}]}""")]
    public void TheSchema_RefusesWhatTheReaderRefuses(string json)
    {
        Assert.False(Valid(json), json);

        // Read through the door that reads a whole file, the fitted half included.
        Assert.IsType<PipelineFileException>(Record.Exception(() => PreparedData.FromJson(json, Everything())));
    }

    [Fact]
    public void AnIndicator_MayReadOneColumnInEveryPlace_ForTheSchemaAsForTheReader()
    {
        // Rows with one price have it as their high, their low and their close alike.
        const string step = """{"step":"feature.indicator","column":"range","indicator":"atr","columns":["close","close","close"],"period":14}""";

        Assert.True(Valid($$"""{"version":2,"declaration":[{{step}}]}"""));
        Assert.Equal(["close", "close", "close"], ((AddIndicatorStep)Everything().ReadStep(step)).Columns);
    }

    [Theory]
    // A word is read in whatever case a person typed it, and the schema says exactly that rather than
    // squiggling a file the reader takes.
    [InlineData("MinMax")]
    [InlineData("MINMAX")]
    public void TheSchema_TakesAWordInAnyCase_AsTheReaderDoes(string scale)
    {
        var json = $$"""{"declaration":[{"step":"normalise","column":"a","scale":"{{scale}}","outOfRange":"pass"}]}""";

        using var document = JsonDocument.Parse(json);
        var step = Everything().Read(document.RootElement.GetProperty("declaration")[0]);

        Assert.True(Valid(json));
        Assert.Equal(Scale.MinMax, ((NormaliseStep)step).Scale);
    }

    [Fact]
    public void ACatalogTaughtAnotherVerb_WritesItIntoItsSchemaAndItsReference()
    {
        var catalog = StepCatalog.BuiltIn();
        catalog.Register<ScaleByStep>();

        using var schema = JsonDocument.Parse(catalog.JsonSchema());

        Assert.Contains("scale.by", schema.RootElement.GetRawText(), StringComparison.Ordinal);
        Assert.DoesNotContain("feature.indicator", schema.RootElement.GetRawText(), StringComparison.Ordinal);
        Assert.Contains("## `scale.by`", catalog.VerbReference(), StringComparison.Ordinal);
        Assert.Contains(ScaleByStep.Purpose, catalog.VerbReference(), StringComparison.Ordinal);

        // A catalog whose verbs all mean what they meant in the first version holds no file to an earlier one.
        var first = new StepCatalog();
        first.Register<ScaleByStep>();

        using var firstSchema = JsonDocument.Parse(first.JsonSchema());

        Assert.False(firstSchema.RootElement.TryGetProperty("allOf", out _));
        Assert.True(schema.RootElement.TryGetProperty("allOf", out _));
    }

    [Fact]
    public void AVerbWithNothingToSet_IsListedWithoutATableOfKeys()
    {
        var catalog = new StepCatalog();
        catalog.Register<MisnamedStep>();

        var reference = catalog.VerbReference();

        Assert.Contains("## `read.parquet`", reference, StringComparison.Ordinal);
        Assert.DoesNotContain("| key |", reference, StringComparison.Ordinal);
    }

    [Fact]
    public void TheVerbReference_NamesEveryVerb_WhatItDoes_AndEveryKeyItTakes()
    {
        var catalog = Everything();
        var reference = catalog.VerbReference();

        foreach (var description in catalog.Descriptions)
        {
            Assert.Contains($"## `{description.Verb}`", reference, StringComparison.Ordinal);
            Assert.Contains(description.Purpose, reference, StringComparison.Ordinal);

            foreach (var key in description.Parameters.SelectMany(parameter => parameter.Keys))
            {
                Assert.Contains($"`{key}`", reference, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void TheCommittedSchema_IsTheOneTheStepsDescribe() =>
        AssertCommitted("pipeline.schema.json", Everything().JsonSchema());

    [Fact]
    public void TheCommittedVerbReference_IsTheOneTheStepsDescribe() =>
        AssertCommitted("VERBS.md", Everything().VerbReference());

    private static void AssertCommitted(string file, string written)
    {
        var path = Path.Join(Repository.Root, file);

        if (Environment.GetEnvironmentVariable(Regenerate) == "1")
        {
            File.WriteAllText(path, written);

            return;
        }

        Assert.True(File.Exists(path), $"{file} is not committed. Write it with {Regenerate}=1.");

        // Compared line by line, so a checkout that turned line endings round is not a difference.
        Assert.Equal(
            written.ReplaceLineEndings("\n"),
            File.ReadAllText(path).ReplaceLineEndings("\n"));
    }
}
