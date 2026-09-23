// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A pipeline travels as one file: the version of the file it was written against, the steps a person wrote,
/// and what the fit learned. Every fitted entry is tied by a key to the steps it was fitted behind, so a fit
/// is never served under steps that changed after it — a declaration without its fit used to serve every
/// value unscaled, and a fit spliced under another declaration served a price of 135.7 where 0.9048 was
/// meant. And a file with something wrong in it says everything that is wrong at once, each at its line and
/// column, rather than one fault at a time.
/// </summary>
public class PipelineFileTests
{
    private static PipelineDeclaration Passengers(double train = 0.70, string target = "survived") =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", train, 0.15)
            .FillMissing("age", With.Median)
            .Normalise("age")
            .Normalise("fare")
            .Target(target)
            .Declaration;

    private static PreparedData Trained() => new Pipeline(Passengers()).Run();

    private static IReadOnlyList<int> WhereEntriesBelong(PipelineDeclaration declaration) =>
        [.. Enumerable.Range(0, declaration.Steps.Count).Where(at => declaration.Steps[at] is IFittedStep or ISplitStep)];

    private static PipelineFileException Refused(string json) =>
        Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(json, StepCatalog.BuiltIn()));

    /// <summary>A line and a column, counted from one, as an editor counts them.</summary>
    private readonly record struct Place(int Line, int Column);

    private static Place At(PipelineFileFault fault) => new(fault.Line, fault.Column);

    // Where a piece of text stands in a file, found by reading the file as lines of characters.
    private static Place Where(string json, string marker, int occurrence = 1)
    {
        var lines = json.ReplaceLineEndings("\n").Split('\n');
        var seen = 0;

        for (var line = 0; line < lines.Length; line++)
        {
            for (var column = lines[line].IndexOf(marker, StringComparison.Ordinal);
                 column >= 0;
                 column = lines[line].IndexOf(marker, column + 1, StringComparison.Ordinal))
            {
                if (++seen == occurrence)
                {
                    return new Place(line + 1, column + 1);
                }
            }
        }

        throw new ArgumentException($"'{marker}' is not in the file {occurrence} times.", nameof(marker));
    }

    [Fact]
    public void TheFileNamesTheVersionItWasWrittenAgainst_FirstOfAll()
    {
        var trained = Trained();

        using var declaration = JsonDocument.Parse(trained.Declaration.ToJson());
        using var whole = JsonDocument.Parse(trained.ToJson());

        Assert.Equal(2, PipelineDeclaration.Version);
        Assert.Equal(["version", "declaration"], declaration.RootElement.EnumerateObject().Select(property => property.Name));
        Assert.Equal(["version", "declaration", "fitted"], whole.RootElement.EnumerateObject().Select(property => property.Name));
        Assert.Equal(PipelineDeclaration.Version, declaration.RootElement.GetProperty("version").GetInt32());
        Assert.Equal(PipelineDeclaration.Version, whole.RootElement.GetProperty("version").GetInt32());
    }

    [Fact]
    public void AFileThatNamesNoVersion_IsReadAsTheFirst_WhereEveryStepItHoldsMeansWhatItMeantThen()
    {
        const string json = """
            {"declaration":[{"step":"read.csv","path":"x.csv"},
                            {"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false}]},
                            {"step":"target","column":"a"}]}
            """;

        var read = PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn());

        Assert.Equal("read.csv -> declare -> target", read.ToString());
    }

    [Fact]
    public void AFileFromANewerVersion_IsRefusedWhole_WithoutReadingAStepOfIt()
    {
        // A newer file may use words this version never had, so none of it is read: the one thing said is
        // that it is newer, rather than a list of faults that are only faults to an older reader.
        const string json = """
            {"version": 3,
             "declaration": [{"step": "read.nowhere"}],
             "colour": "red"}
            """;

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
        var fault = Assert.Single(refused.Faults);

        Assert.Equal(Where(json, "\"version\""), At(fault));
        Assert.Contains("version 3", fault.Message, StringComparison.Ordinal);
        Assert.Contains("version 2", fault.Message, StringComparison.Ordinal);
        Assert.Contains("newer", fault.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("\"2\"")]
    [InlineData("2.5")]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("null")]
    [InlineData("true")]
    [InlineData("[2]")]
    [InlineData("{\"number\":2}")]
    public void AVersionThatIsNotAWholeNumberFromOne_IsRefused(string version)
    {
        var json = $$"""{"version":{{version}},"declaration":[]}""";

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Equal(new Place(1, 2), At(Assert.Single(refused.Faults)));
    }

    [Fact]
    public void AStepWhoseMeaningChanged_IsRefusedByName_InAFileFromBeforeTheChange()
    {
        // A random split written by 0.2 divided the rows by their place in the file; this version divides
        // them by what they hold. Reading the old file the new way would train on other rows and say nothing,
        // so it is refused, naming the step, both versions, and where it is.
        const string json = """
            {"declaration": [
              {"step": "read.csv", "path": "titanic.csv"},
              {"step": "declare", "remainder": "keep", "columns": [{"name": "survived", "kind": "integer", "optional": false}]},
              {"step": "split.atRandom", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 42}
            ]}
            """;

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
        var fault = Assert.Single(refused.Faults);

        Assert.Equal(Where(json, "{\"step\": \"split.atRandom\""), At(fault));
        Assert.Contains("Step 3", fault.Message, StringComparison.Ordinal);
        Assert.Contains("split.atRandom", fault.Message, StringComparison.Ordinal);
        Assert.Contains("version 1", fault.Message, StringComparison.Ordinal);
        Assert.Contains("version 2", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AStepRead_AgainstAnEarlierVersion_IsRefusedWhenItsMeaningChangedSince()
    {
        using var split = JsonDocument.Parse("""{"step":"split.atRandom","train":0.7,"validation":0.15,"test":0.15,"seed":42}""");
        using var source = JsonDocument.Parse("""{"step":"read.csv","path":"x.csv"}""");
        var catalog = StepCatalog.BuiltIn();

        var refused = Assert.Throws<FormatException>(() => catalog.Read(split.RootElement, writtenAgainst: 1));

        Assert.Contains("split.atRandom", refused.Message, StringComparison.Ordinal);
        Assert.Equal("read.csv", catalog.Read(source.RootElement, writtenAgainst: 1).Verb);
        Assert.Equal("split.atRandom", catalog.Read(split.RootElement, writtenAgainst: 2).Verb);
        Assert.Throws<ArgumentOutOfRangeException>(() => catalog.Read(source.RootElement, writtenAgainst: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => catalog.Read(source.RootElement, writtenAgainst: 3));
    }

    [Fact]
    public void EveryUnreadableStepIsReportedAtOnce_EachAtItsLineAndColumn()
    {
        const string json = """
            {
              "version": 2,
              "declaration": [
                {"step": "read.csv"},
                {"step": "declare", "remainder": "drop", "columns": [{"name": "a", "kind": "number", "optional": false}]},
                {"step": "normalise", "column": "a", "scale": "sideways", "outOfRange": "pass"},
                {"step": "read.cvs", "path": "x.csv"}
              ]
            }
            """;

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Equal(
            [Where(json, "{\"step\": \"read.csv\""), Where(json, "{\"step\": \"normalise\""), Where(json, "{\"step\": \"read.cvs\"")],
            refused.Faults.Select(At));
        Assert.Contains("path", refused.Faults[0].Message, StringComparison.Ordinal);
        Assert.Contains("sideways", refused.Faults[1].Message, StringComparison.Ordinal);
        Assert.Contains("'read.csv'", refused.Faults[2].Message, StringComparison.Ordinal);
        Assert.Contains("(4,5): Step 1", refused.Message, StringComparison.Ordinal);
        Assert.Equal($"(4,5): {refused.Faults[0].Message}", refused.Faults[0].ToString());
    }

    [Fact]
    public void AStepRefusingItsOwnValues_IsReportedInTheFilesWords_NotInThoseOfACSharpParameter()
    {
        // A step refuses a value the way a C# method refuses an argument, naming the parameter; a file has keys,
        // and a key written in lower case is not the parameter a message would name.
        const string json = """
            {"version": 2, "declaration": [{"step": "split.atRandom", "train": 0.7, "validation": 0.45, "test": 0.15, "seed": 1}]}
            """;

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
        var fault = Assert.Single(refused.Faults).Message;

        Assert.Contains("add up to 1.3", fault, StringComparison.Ordinal);
        Assert.DoesNotContain("Parameter", fault, StringComparison.Ordinal);
        Assert.EndsWith("every row.", fault, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnIsCountedInCharacters_NotInTheBytesAWordTakes()
    {
        const string json = """{"version":2,"declaration":[{"step":"read.csv","path":"Größe.csv"},{"step":"read.cvs"}]}""";

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Equal(Where(json, "{\"step\":\"read.cvs\""), At(Assert.Single(refused.Faults)));
    }

    [Fact]
    public void ARuleABrokenFileBreaks_IsReportedAtTheStepThatBreaksIt()
    {
        const string json = """
            {
              "version": 2,
              "declaration": [
                {"step": "read.csv", "path": "a.csv"},
                {"step": "declare", "remainder": "drop", "columns": [{"name": "a", "kind": "number", "optional": false}]},
                {"step": "target", "column": "a"},
                {"step": "target", "column": "a"}
              ]
            }
            """;

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
        var fault = Assert.Single(refused.Faults);

        Assert.Equal(Where(json, "{\"step\": \"target\"", occurrence: 2), At(fault));
        Assert.Contains("Step 4, 'target'", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AKeyNoPipelineFileHas_IsRefusedWhereItIs()
    {
        const string json = """{"version":2,"declaration":[],"colour":"red","fitted":[],"size":3}""";

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Equal([Where(json, "\"colour\""), Where(json, "\"size\"")], refused.Faults.Select(At));
        Assert.All(refused.Faults, fault => Assert.Contains("version, declaration, fitted", fault.Message, StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("[]")]
    [InlineData("\"declaration\"")]
    [InlineData("{\"version\":2}")]
    [InlineData("{\"version\":2,\"declaration\":{\"step\":\"read.csv\"}}")]
    public void AFileThatIsNotAPipeline_IsRefused(string json)
    {
        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Single(refused.Faults);
    }

    [Fact]
    public void TextThatStopsBeingJson_IsRefusedWhereItStops_CountedFromOne()
    {
        const string json = """
            {
              "version": 2,
              "declaration": [
                {"step": "read.csv", "path": "x.csv",}
              ]
            }
            """;

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
        var fault = Assert.Single(refused.Faults);
        var comma = Where(json, ",}");

        Assert.Equal(new Place(comma.Line, comma.Column + 1), At(fault));
        Assert.DoesNotContain("LineNumber", fault.Message, StringComparison.Ordinal);
        Assert.Equal(new Place(1, 1), At(Assert.Single(Assert.Throws<PipelineFileException>(
            () => PipelineDeclaration.FromJson("", StepCatalog.BuiltIn())).Faults)));
    }

    [Fact]
    public void AKeyWrittenTwice_IsRefusedAtTheSecond_BecauseOnlyOneOfThemWouldBeRead()
    {
        const string json = """
            {"version": 2,
             "declaration": [{"step": "read.csv", "path": "a.csv", "path": "b.csv"}],
             "declaration": []}
            """;

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Equal([Where(json, "\"path\"", occurrence: 2), Where(json, "\"declaration\"", occurrence: 2)], refused.Faults.Select(At));
        Assert.Contains("'path'", refused.Faults[0].Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryStepHasAKey_TheSameInEveryProcessOnEveryMachine()
    {
        // The key a fit is filed under is written into files that outlive the process that wrote them, so it
        // is pinned here to the digest worked out apart from this library: SHA-256 over a fixed label and the
        // step as it writes itself, chained from one step to the next.
        var declaration = new PipelineDeclaration([
            new ReadCsvStep("x.csv"),
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, false)]),
        ]);

        Assert.Equal("ff81243c839982c4f1289883986c265d80726e9fb64e645c2186af3ec527a752", declaration.KeyAt(0));
        Assert.Equal("455fc308c88266eb4cbceb99a103a4ccec50f1e4a0cbbcbb961c1d12d7b2ead2", declaration.KeyAt(1));
        Assert.Throws<ArgumentOutOfRangeException>(() => declaration.KeyAt(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => declaration.KeyAt(-1));
    }

    [Fact]
    public void TwoIdenticalStepsAtTwoPlaces_HaveTwoKeys_AndAPrefixHasTheKeysOfTheWhole()
    {
        var declaration = Pdd.Create()
            .ReadCsv("x.csv")
            .Declare(schema => schema.Number("a"))
            .Profile()
            .Profile()
            .Declaration;

        var keys = Enumerable.Range(0, 4).Select(declaration.KeyAt).ToArray();

        Assert.Equal(4, keys.Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(keys[..3], Enumerable.Range(0, 3).Select(new PipelineDeclaration(declaration.Steps.Take(3)).KeyAt));
    }

    [Fact]
    public void TheKeyIsTheStepsAndNotHowTheFileHappensToBeTyped()
    {
        const string compact = """
            {"version":2,"declaration":[{"step":"read.csv","path":"x.csv"},{"step":"declare","remainder":"drop","columns":[{"name":"t","kind":"timestamp","optional":false},{"name":"a","kind":"number","optional":false}]},{"step":"split.byTime","column":"t","train":0.7,"validation":0.15,"test":0.15},{"step":"normalise","column":"a","scale":"minMax","outOfRange":"pass"}]}
            """;
        const string typedByHand = """
            {
              "declaration": [
                { "path": "x.csv", "step": "read.csv" },
                { "columns": [ { "optional": false, "kind": "Timestamp", "name": "t" }, { "kind": "NUMBER", "name": "a", "optional": false } ],
                  "step": "declare", "remainder": "Drop" },
                { "test": 0.150, "train": 0.70, "column": "t", "validation": 1.5e-1, "step": "split.byTime" },
                { "outOfRange": "Pass", "scale": "MinMax", "column": "a", "step": "normalise" }
              ],
              "version": 2
            }
            """;

        var one = PipelineDeclaration.FromJson(compact, StepCatalog.BuiltIn());
        var other = PipelineDeclaration.FromJson(typedByHand, StepCatalog.BuiltIn());

        Assert.Equal(Enumerable.Range(0, 4).Select(one.KeyAt), Enumerable.Range(0, 4).Select(other.KeyAt));
    }

    [Fact]
    public void AFittedPipeline_FilesEachEntryUnderItsStepAndTheKeyOfTheStepsItWasFittedBehind()
    {
        var trained = Trained();
        var declaration = trained.Declaration;

        using var file = JsonDocument.Parse(trained.ToJson());
        var entries = file.RootElement.GetProperty("fitted").EnumerateArray().ToArray();
        var belong = WhereEntriesBelong(declaration);

        Assert.Equal(belong.Count, entries.Length);

        for (var entry = 0; entry < entries.Length; entry++)
        {
            Assert.Equal(["step", "prefix", "learned"], entries[entry].EnumerateObject().Select(property => property.Name));
            Assert.Equal(declaration.Steps[belong[entry]].Verb, entries[entry].GetProperty("step").GetString());
            Assert.Equal(declaration.KeyAt(belong[entry]), entries[entry].GetProperty("prefix").GetString());
            Assert.Equal(JsonValueKind.Object, entries[entry].GetProperty("learned").ValueKind);
        }
    }

    [Fact]
    public void ASavedPipeline_ComesBackWithEveryFitWhereItWasLearned()
    {
        var trained = Trained();

        var loaded = PreparedData.FromJson(trained.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(trained.Declaration, loaded.Declaration);
        Assert.Equal(trained.Fitted.Keys, loaded.Fitted.Keys);
        Assert.All(trained.Fitted, fit => Assert.Equal(fit.Value.Numbers, loaded.Fitted[fit.Key].Numbers));
        Assert.Equal(trained.Fitted[2].Texts, loaded.Fitted[2].Texts);
    }

    [Fact]
    public void ADeclarationWithoutItsFit_IsRefused_NamingEveryStepThatLearnedSomething()
    {
        // Loaded as a fitted pipeline, the declaration alone used to serve every age unscaled — 22 where
        // -0.5665 was meant — because a missing fit was skipped rather than missed.
        var trained = Trained();
        var json = trained.Declaration.ToJson();

        var refused = Refused(json);

        Assert.Equal(
            WhereEntriesBelong(trained.Declaration).Select(at => trained.Declaration.Steps[at].Verb),
            refused.Faults.Select(fault => fault.Message.Split('\'')[1]));
        Assert.Equal(Where(json, "\"step\": \"split.stratified\"").Line - 1, refused.Faults[0].Line);
        Assert.All(refused.Faults, fault => Assert.Contains("Fit the pipeline again", fault.Message, StringComparison.Ordinal));
    }

    [Fact]
    public void AFitUnderAnEditedDeclaration_IsRefused_BecauseItWasLearnedBehindOtherSteps()
    {
        // The fitted half of one pipeline under the declaration of another used to load and serve: every
        // number wrong and nothing said.
        var file = JsonNode.Parse(Trained().ToJson())!;
        file["declaration"] = JsonNode.Parse(Passengers(train: 0.60).ToJson())!["declaration"]!.DeepClone();
        var json = file.ToJsonString();

        var refused = Refused(json);

        Assert.Equal(4, refused.Faults.Count);
        Assert.All(refused.Faults, fault => Assert.Contains("changed after", fault.Message, StringComparison.Ordinal));
        Assert.Equal(Where(json, "{\"step\":\"split.stratified\",\"prefix\""), At(refused.Faults[0]));
    }

    [Fact]
    public void AnEditBelowTheLastStepThatLearned_KeepsEveryFit()
    {
        // The key of a step covers the steps above it and nothing below, so changing the target leaves
        // every fit where it was.
        var trained = Trained();
        var edited = Passengers(target: "pclass");
        var file = JsonNode.Parse(trained.ToJson())!;
        file["declaration"] = JsonNode.Parse(edited.ToJson())!["declaration"]!.DeepClone();

        var loaded = PreparedData.FromJson(file.ToJsonString(), StepCatalog.BuiltIn());

        Assert.Equal(edited, loaded.Declaration);
        Assert.Equal(trained.Fitted.Keys, loaded.Fitted.Keys);
        Assert.Equal(trained.Fitted[4].Number("centre"), loaded.Fitted[4].Number("centre"));
        Assert.NotEqual(trained.Declaration.KeyAt(6), edited.KeyAt(6));
    }

    [Fact]
    public void AFittedHalfWrittenByPosition_IsRefused_AsTheOneFaultItIs()
    {
        var file = JsonNode.Parse(Trained().ToJson())!;
        file["fitted"] = JsonNode.Parse("""{"2":{"rows.train":623},"3":{"value":28}}""");
        var json = file.ToJsonString();

        var fault = Assert.Single(Refused(json).Faults);

        Assert.Equal(Where(json, "\"fitted\""), At(fault));
        Assert.Contains("position", fault.Message, StringComparison.Ordinal);
        Assert.Contains("Fit the pipeline again", fault.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("\"none\"")]
    [InlineData("3")]
    [InlineData("null")]
    public void AFittedHalfThatIsNotAList_IsRefused(string fitted)
    {
        var file = JsonNode.Parse(Trained().ToJson())!;
        file["fitted"] = JsonNode.Parse(fitted);
        var json = file.ToJsonString();

        var fault = Assert.Single(Refused(json).Faults);

        Assert.Equal(Where(json, "\"fitted\""), At(fault));
        Assert.Contains("a list", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileSavedByTheLastVersion_IsRefusedForEveryReasonAtOnce()
    {
        const string json = """
            {
              "declaration": [
                {"step": "read.csv", "path": "titanic.csv"},
                {"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}]},
                {"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 42},
                {"step": "fill.missing", "column": "age", "with": "median"}
              ],
              "fitted": {"3": {"gaps": 177, "value": 28}}
            }
            """;

        var refused = Refused(json);

        Assert.Equal([Where(json, "{\"step\": \"split.stratified\""), Where(json, "\"fitted\"")], refused.Faults.Select(At));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(99)]
    [InlineData(0)]
    public void ThePreparedDataConstructor_RefusesAFitThatIsNotALearningStepsOrTheSplits(int at)
    {
        // The same rule in memory as in the file: a fit is where a step learned, or where the rows were divided.
        var trained = Trained();
        var fitted = trained.Fitted.ToDictionary();
        fitted[at] = new FittedStepValues();

        Assert.Throws<ArgumentException>(() => new PreparedData(trained.Declaration, trained.Table, trained.Parts, fitted));
    }

    [Fact]
    public void AnEntryForAStepThatLearnsNothing_IsRefused()
    {
        var trained = Trained();
        var file = JsonNode.Parse(trained.ToJson())!;
        file["fitted"]!.AsArray().Add(JsonNode.Parse($$$"""{"step":"read.csv","prefix":"{{{trained.Declaration.KeyAt(0)}}}","learned":{}}"""));
        var json = file.ToJsonString();

        var fault = Assert.Single(Refused(json).Faults);

        Assert.Equal(Where(json, "{\"step\":\"read.csv\",\"prefix\""), At(fault));
        Assert.Contains("learns nothing", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASecondEntryForOneStep_IsRefused()
    {
        var file = JsonNode.Parse(Trained().ToJson())!;
        var entries = file["fitted"]!.AsArray();
        entries.Add(entries[1]!.DeepClone());
        var json = file.ToJsonString();

        var fault = Assert.Single(Refused(json).Faults);

        // The step itself stands in the declaration, then its entry, then the entry written again.
        Assert.Equal(Where(json, "{\"step\":\"fill.missing\"", occurrence: 3), At(fault));
        Assert.Contains("second", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEntryFiledUnderAnotherVerbThanItsStep_IsRefused()
    {
        var file = JsonNode.Parse(Trained().ToJson())!;
        file["fitted"]![0]!["step"] = "normalise";
        var json = file.ToJsonString();

        var fault = Assert.Single(Refused(json).Faults);

        Assert.Contains("split.stratified", fault.Message, StringComparison.Ordinal);
        Assert.Contains("normalise", fault.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("\"split\"")]
    [InlineData("""{"step":"split.stratified","learned":{}}""")]
    [InlineData("""{"prefix":"PREFIX","learned":{}}""")]
    [InlineData("""{"step":"split.stratified","prefix":"PREFIX"}""")]
    [InlineData("""{"step":"split.stratified","prefix":"PREFIX","learned":{"a":1e400}}""")]
    [InlineData("""{"step":"split.stratified","prefix":"PREFIX","learned":{"a":[1,1e400]}}""")]
    [InlineData("""{"step":"split.stratified","prefix":"PREFIX","learned":{},"colour":"red"}""")]
    [InlineData("""{"step":3,"prefix":"PREFIX","learned":{}}""")]
    [InlineData("""{"step":"split.stratified","prefix":7,"learned":{}}""")]
    [InlineData("""{"step":"split.stratified","prefix":"PREFIX","learned":[]}""")]
    [InlineData("""{"step":"split.stratified","prefix":"PREFIX","learned":{"a":[1,"b"]}}""")]
    [InlineData("""{"step":"split.stratified","prefix":"PREFIX","learned":{"a":true}}""")]
    [InlineData("""{"step":"split.stratified","prefix":"PREFIX","learned":{"a":{"b":1}}}""")]
    [InlineData("""{"step":"split.stratified","prefix":"PREFIX","learned":{"a":null}}""")]
    [InlineData("""{"step":"split.stratified","prefix":"PREFIX","learned":{"a":[[1]]}}""")]
    [InlineData("""{"step":"split.stratified","prefix":"PREFIX","learned":{"a":[null]}}""")]
    public void AnEntryThatIsNotWhatAFitWrites_IsRefusedAsOneFault(string entry)
    {
        var trained = Trained();
        var file = JsonNode.Parse(trained.ToJson())!;
        file["fitted"]![0] = JsonNode.Parse(entry.Replace("PREFIX", trained.Declaration.KeyAt(2), StringComparison.Ordinal));
        var json = file.ToJsonString();

        var fault = Assert.Single(Refused(json).Faults);

        Assert.Equal(Where(json, entry.Replace("PREFIX", trained.Declaration.KeyAt(2), StringComparison.Ordinal)), At(fault));
    }

    [Fact]
    public void AnEmptyListAndAnEmptyRun_SurviveTheFile_AsWhicheverTheStepAsksFor()
    {
        // Both are written as [], and a file cannot say which of the two an empty one was.
        var catalog = StepCatalog.BuiltIn();
        catalog.Register<LearnNothingStep>();
        var rows = CsvRowSource.FromText("a,b\n1,2\n3,4\n5,6\n7,8\n");
        var trained = Pdd.Create()
            .Read(rows, "four rows")
            .Declare(schema => schema.Number("a", "b"))
            .SplitAtRandom(0.50, 0.25)
            .Add(new LearnNothingStep())
            .Build()
            .Run();

        var loaded = PreparedData.FromJson(trained.ToJson(), catalog);

        Assert.Equal(4, loaded.Replay(rows).RowCount);
        Assert.Empty(loaded.Fitted[3].List("words"));
        Assert.Empty(loaded.Fitted[3].Curve("run"));
        Assert.Throws<InvalidOperationException>(() => loaded.Fitted[3].Curve("words and more"));
    }

    [Fact]
    public void ANameALearnedValueIsFiledUnder_HoldsOneKindOfThing()
    {
        // Two kinds under one name would write the name twice, and a file saying one thing twice is refused.
        var learned = new FittedStepValues();
        learned.Learned("a", 1);
        learned.Learned("a", 2);

        Assert.Throws<ArgumentException>(() => learned.Learned("a", "one"));
        Assert.Throws<ArgumentException>(() => learned.Learned("a", ["one"]));
        Assert.Throws<ArgumentException>(() => learned.Learned("a", [1.0]));
        Assert.Equal(2, learned.Number("a"));

        learned.Learned("b", ["x"]);
        learned.Learned("c", "words");
        learned.Learned("d", [1.0]);

        Assert.Throws<ArgumentException>(() => learned.Learned("b", 3));
        Assert.Throws<ArgumentException>(() => learned.Learned("c", 3));
        Assert.Throws<ArgumentException>(() => learned.Learned("d", "words"));
        Assert.Throws<ArgumentNullException>(() => learned.Learned(null!, 3));
        Assert.Throws<ArgumentNullException>(() => learned.Learned("e", (string)null!));
        Assert.Throws<ArgumentNullException>(() => learned.Learned("e", (IReadOnlyList<string>)null!));
        Assert.Throws<ArgumentNullException>(() => learned.Learned("e", (IReadOnlyList<double>)null!));
    }

    [Fact]
    public void AnEmptyListAndAnEmptyRun_AreOneNothing_AndOnlyAnEmptyOne()
    {
        var learned = new FittedStepValues();
        learned.Learned("run", Array.Empty<double>());
        learned.Learned("words", Array.Empty<string>());
        learned.Learned("full run", [1.0]);
        learned.Learned("full list", ["x"]);

        Assert.Empty(learned.List("run"));
        Assert.Empty(learned.Curve("words"));
        Assert.Throws<InvalidOperationException>(() => learned.List("full run"));
        Assert.Throws<InvalidOperationException>(() => learned.Curve("full list"));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void ANumberThatIsNotFinite_IsNotLearned_BecauseNoFileCanHoldIt(double value)
    {
        var learned = new FittedStepValues();

        Assert.Throws<ArgumentOutOfRangeException>(() => learned.Learned("a", value));
        Assert.Throws<ArgumentOutOfRangeException>(() => learned.Learned("b", [1.0, value]));
        Assert.Empty(learned.Numbers);
        Assert.Empty(learned.Curves);
    }

    [Fact]
    public void TheDeclarationDoor_ReadsTheStepsOfAFittedFile_AndLeavesTheFitToTheFit()
    {
        // Reading the declaration alone is how a pipeline is fitted again on fresh data, so a fit that no
        // longer matches it is not the declaration's business.
        var file = JsonNode.Parse(Trained().ToJson())!;
        file["declaration"] = JsonNode.Parse(Passengers(train: 0.60).ToJson())!["declaration"]!.DeepClone();

        Assert.Equal(Passengers(train: 0.60), PipelineDeclaration.FromJson(file.ToJsonString(), StepCatalog.BuiltIn()));
    }

    [Fact]
    public void AStepWrittenOnItsOwn_IsReadWithEveryFaultAtItsLineAndColumn()
    {
        // A notebook block holds one step as text; its faults are placed the way a file's are, by the same door.
        var catalog = StepCatalog.BuiltIn();

        Assert.Equal(new ReadCsvStep("x.csv"), catalog.ReadStep("""{"step": "read.csv", "path": "x.csv"}"""));
        Assert.Equal(new ReadCsvStep("x.csv"), catalog.ReadStep("""{"step": "read.csv", "path": "x.csv"}""", writtenAgainst: 1));

        const string twice = "{\"step\": \"read.csv\", \"path\": \"x.csv\"}\n{\"step\": \"read.csv\", \"path\": \"y.csv\"}";
        const string keyTwice = "{\"step\": \"read.csv\",\n \"path\": \"x.csv\", \"path\": \"y.csv\"}";

        Assert.Equal(new Place(1, 1), At(Assert.Single(Assert.Throws<PipelineFileException>(() => catalog.ReadStep("""{"step": "read.csv"}""")).Faults)));
        Assert.Contains("path", Assert.Throws<PipelineFileException>(() => catalog.ReadStep("""{"step": "read.csv"}""")).Message, StringComparison.Ordinal);
        Assert.Contains("'split.byTime'", Assert.Throws<PipelineFileException>(() => catalog.ReadStep("""{"step": "split.byTime", "column": "t"}""")).Message, StringComparison.Ordinal);
        Assert.Equal(new Place(2, 1), At(Assert.Single(Assert.Throws<PipelineFileException>(() => catalog.ReadStep(twice)).Faults)));
        Assert.Equal(new Place(2, 19), At(Assert.Single(Assert.Throws<PipelineFileException>(() => catalog.ReadStep(keyTwice)).Faults)));
        Assert.Equal(new Place(1, 1), At(Assert.Single(Assert.Throws<PipelineFileException>(() => catalog.ReadStep("""[{"step": "read.csv", "path": "x.csv"}]""")).Faults)));
        Assert.Contains("'split.atRandom'", Assert.Throws<PipelineFileException>(
            () => catalog.ReadStep("""{"step":"split.atRandom","train":0.7,"validation":0.15,"test":0.15,"seed":1}""", writtenAgainst: 1)).Message, StringComparison.Ordinal);
        Assert.Contains("read.cvs", Assert.Throws<PipelineFileException>(() => catalog.ReadStep("""{"step": "read.cvs"}""")).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => catalog.ReadStep(null!));
    }

    [Fact]
    public void NothingIsReadWithoutTheTextOrTheVerbs()
    {
        Assert.Throws<ArgumentNullException>(() => PipelineDeclaration.FromJson(null!, StepCatalog.BuiltIn()));
        Assert.Throws<ArgumentNullException>(() => PreparedData.FromJson(null!, StepCatalog.BuiltIn()));
        Assert.Throws<ArgumentNullException>(() => PreparedData.FromJson("{}", null!));
        Assert.Throws<ArgumentNullException>(() => new PipelineFileException(null!));
    }
}
