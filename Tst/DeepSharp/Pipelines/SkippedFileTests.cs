// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;
using DeepSharp.Tests.Learners;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A run for a learner writes the steps it left out into its file, under <c>skipped</c>, each by its verb and the key of
/// the steps up to it, after what the fit learned; a run that left nothing out writes the file it always wrote, but for the
/// version, so it is the fit a 0.4.0 file carries. Read back, the file leaves out the same steps, and whatever in it cannot be
/// what a run for one learner left out is refused where it stands.
/// </summary>
public class SkippedFileTests
{
    private static readonly JsonSerializerOptions Indented = new() { WriteIndented = true };

    private static Pipeline Titanic => WikiTitanic.In(WikiTitanic.DataFolder);

    [Fact]
    public void ARunThatLeftOutSteps_WritesThemUnderSkipped_ByVerbAndPrefix_AfterWhatItLearned()
    {
        var categories = Titanic.RunFor(Needs.Categories);
        using var file = JsonDocument.Parse(categories.ToJson());
        using var every = JsonDocument.Parse(Titanic.Run().ToJson());
        var root = file.RootElement;

        Assert.Equal(5, PipelineDeclaration.Version);
        Assert.Equal(["version", "declaration", "fitted", "skipped"], root.EnumerateObject().Select(member => member.Name));
        Assert.Equal(5, root.GetProperty("version").GetInt32());
        Assert.Equal(
            ["encode.categories", "normalise", "normalise", "normalise", "normalise"],
            root.GetProperty("skipped").EnumerateArray().Select(entry => entry.GetProperty("step").GetString()));
        Assert.Equal(
            [.. new[] { 4, 5, 6, 7, 8 }.Select(categories.Declaration.KeyAt)],
            root.GetProperty("skipped").EnumerateArray().Select(entry => entry.GetProperty("prefix").GetString()));

        // The split, the fill and the categories are learned alike for every learner: the same entries, byte for byte.
        Assert.Equal(
            every.RootElement.GetProperty("fitted").EnumerateArray().Take(3).Select(entry => entry.GetRawText()),
            root.GetProperty("fitted").EnumerateArray().Select(entry => entry.GetRawText()));
        Assert.Equal(every.RootElement.GetProperty("declaration").GetRawText(), root.GetProperty("declaration").GetRawText());
    }

    [Fact]
    public void ARunOfEveryStep_WritesNothingNew_SoItIsTheFitThePipelineOfA040FileIs()
    {
        var every = Titanic.Run();
        using var file = JsonDocument.Parse(every.ToJson());
        var carried = JsonNode.Parse(File.ReadAllText(Path.Join(AppContext.BaseDirectory, "Learners", "Fixtures", "titanic-0.4.0.network.json")))!["pipeline"]!;

        Assert.Equal(["version", "declaration", "fitted"], file.RootElement.EnumerateObject().Select(member => member.Name));
        Assert.Equal(PipelineText.Of(Encoding.UTF8.GetBytes(carried.ToJsonString(Indented))).FitDigest, PipelineText.Of(every).FitDigest);
        Assert.Equal(3, (int)carried["version"]!);
    }

    [Theory]
    [InlineData(Needs.NoScale, new[] { 5, 6, 7, 8 })]
    [InlineData(Needs.Categories, new[] { 4, 5, 6, 7, 8 })]
    public void ARunReadBackFromItsFile_LeavesOutTheSameSteps_AndServesAsItDid(Needs needs, int[] skipped)
    {
        var run = Titanic.RunFor(needs);
        var passenger = new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22", "1", "0", "7.25"]]);

        var read = PreparedData.FromJson(run.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(skipped, read.Skipped);
        Assert.Equal(run.ToJson(), read.ToJson());
        Assert.Equal(run.Served(passenger, needs).Features[0], read.Served(passenger, needs).Features[0]);
    }

    [Fact]
    public void WhatARunLeftOut_NamedInAFileOfAnEarlierVersion_IsRefusedAtItsKey()
    {
        var json = Edited(Titanic.RunFor(Needs.NoScale), file => file["version"] = 3);

        var fault = Assert.Single(Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(json, StepCatalog.BuiltIn())).Faults);

        Assert.Equal(KeyAt(json, "skipped"), (fault.Line, fault.Column));
        Assert.Contains("version 4", fault.Message, StringComparison.Ordinal);
        Assert.Contains("version 3", fault.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("prefix", "0000000000000000000000000000000000000000000000000000000000000000", "behind steps this declaration does not have")]
    [InlineData("step", "fill.missing", "the step at that place is step 6, 'normalise'")]
    public void AnEntryOfWhatARunLeftOut_ThatNamesNoStepOfItsDeclaration_IsRefusedAtTheEntry(string key, string value, string said)
    {
        var json = Edited(Titanic.RunFor(Needs.NoScale), file => file["skipped"]![0]![key] = value);

        var fault = Assert.Single(Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(json, StepCatalog.BuiltIn())).Faults);

        Assert.Equal(EntryAt(json, "skipped", 0), (fault.Line, fault.Column));
        Assert.Contains(said, fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AStepNoLearnerDoesWithout_SaidToBeLeftOut_IsRefusedAtItsEntry()
    {
        var run = Titanic.RunFor(Needs.NoScale);
        var json = Edited(run, file => file["skipped"]!.AsArray().Insert(0, new JsonObject { ["step"] = "fill.missing", ["prefix"] = run.Declaration.KeyAt(3) }));

        var fault = Assert.Single(Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(json, StepCatalog.BuiltIn())).Faults);

        Assert.Equal(EntryAt(json, "skipped", 0), (fault.Line, fault.Column));
        Assert.Contains("Step 4, 'fill.missing': is taken by every run", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void StepsNoRunForOneLearnerLeavesOutTogether_AreRefusedAtTheFirst()
    {
        // The encoder is left out for a learner that takes categories, and with it every scale; never on its own.
        var json = Edited(Titanic.RunFor(Needs.Categories), file =>
        {
            var skipped = file["skipped"]!.AsArray();

            while (skipped.Count > 1)
            {
                skipped.RemoveAt(1);
            }

            file["fitted"]!.AsArray().Add(JsonNode.Parse(Titanic.Run().ToJson())!["fitted"]![3]!.DeepClone());
        });

        var faults = Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(json, StepCatalog.BuiltIn())).Faults;

        Assert.Equal(EntryAt(json, "skipped", 0), (faults[0].Line, faults[0].Column));
        Assert.Contains("Step 5, 'encode.categories': is one of the steps left out here, 5,", faults[0].Message, StringComparison.Ordinal);
        Assert.Contains("no run for one learner leaves out exactly these", faults[0].Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WhatAStepTheRunLeftOutLearned_IsRefusedAtItsEntry_ForItLearnedNothing()
    {
        var every = JsonNode.Parse(Titanic.Run().ToJson())!;
        var json = Edited(Titanic.RunFor(Needs.NoScale), file => file["fitted"]!.AsArray().Add(every["fitted"]![3]!.DeepClone()));

        var fault = Assert.Single(Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(json, StepCatalog.BuiltIn())).Faults);

        Assert.Equal(EntryAt(json, "fitted", 3), (fault.Line, fault.Column));
        Assert.Contains("Step 6, 'normalise', was left out of the run", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AStepTheRunTookInAnotherForm_WithoutWhatItLearned_IsRefusedAtTheStep()
    {
        // In a run for a learner that takes categories, the encoder learns its categories as it always does.
        var json = Edited(Titanic.RunFor(Needs.Categories), file => file["fitted"]!.AsArray().RemoveAt(2));

        var fault = Assert.Single(Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(json, StepCatalog.BuiltIn())).Faults);

        Assert.Equal(EntryAt(json, "declaration", 4), (fault.Line, fault.Column));
        Assert.Contains("learns from the data, and nothing it learned is here", fault.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("""{"step": "normalise"}""", "the key of the steps up to it")]
    [InlineData("""{"prefix": "00"}""", "names the step it left out")]
    [InlineData("""{"step": 5, "prefix": "00"}""", "names the step it left out")]
    [InlineData("""{"step": "normalise", "prefix": 5}""", "the key of the steps up to it")]
    [InlineData("""{"step": "normalise", "prefix": "00", "learned": {}}""", "has no 'learned'")]
    [InlineData("""["normalise"]""", "is an object")]
    public void AnEntryOfWhatARunLeftOut_ThatIsNotAVerbAndAPrefix_IsRefusedAtTheEntry(string entry, string said)
    {
        var json = Edited(Titanic.RunFor(Needs.NoScale), file => file["skipped"]![0] = JsonNode.Parse(entry));

        var fault = Assert.Single(Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(json, StepCatalog.BuiltIn())).Faults);

        Assert.Equal(EntryAt(json, "skipped", 0), (fault.Line, fault.Column));
        Assert.Contains(said, fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WhatARunLeftOut_WrittenTwiceForOneStep_OrAsNoList_IsRefused()
    {
        var run = Titanic.RunFor(Needs.NoScale);
        var twice = Edited(run, file => file["skipped"]!.AsArray().Add(file["skipped"]![0]!.DeepClone()));
        var notAList = Edited(run, file => file["skipped"] = "normalise");

        var second = Assert.Single(Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(twice, StepCatalog.BuiltIn())).Faults);
        var list = Assert.Single(Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(notAList, StepCatalog.BuiltIn())).Faults);

        Assert.Equal(EntryAt(twice, "skipped", 4), (second.Line, second.Column));
        Assert.Contains("second entry for step 6, 'normalise'", second.Message, StringComparison.Ordinal);
        Assert.Equal(KeyAt(notAList, "skipped"), (list.Line, list.Column));
        Assert.Contains("a list", list.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEmptyListOfWhatARunLeftOut_IsARunOfEveryStep()
    {
        var json = Edited(Titanic.Run(), file => file["skipped"] = new JsonArray());

        Assert.Empty(PreparedData.FromJson(json, StepCatalog.BuiltIn()).Skipped);
    }

    [Fact]
    public void TheDeclarationAlone_IsReadWhateverTheRunLeftOut()
    {
        var json = Titanic.RunFor(Needs.Categories).ToJson();

        Assert.Equal(WikiTitanic.Declaration, PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    private static string Edited(PreparedData run, Action<JsonNode> edit)
    {
        var file = JsonNode.Parse(run.ToJson())!;

        edit(file);

        return file.ToJsonString(Indented).ReplaceLineEndings("\n");
    }

    // Where a key at the top of the file stands, line and column from one.
    private static (int Line, int Column) KeyAt(string json, string key)
    {
        var bytes = Encoding.UTF8.GetBytes(json);
        var reader = new Utf8JsonReader(bytes);

        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.PropertyName && reader.CurrentDepth == 1 && reader.ValueTextEquals(key))
            {
                return Placed(json, (int)reader.TokenStartIndex);
            }
        }

        throw new ArgumentException($"There is no '{key}' at the top of the file.", nameof(key));
    }

    // Where the given element of a list at the top of the file stands, line and column from one.
    private static (int Line, int Column) EntryAt(string json, string key, int entry)
    {
        var bytes = Encoding.UTF8.GetBytes(json);
        var reader = new Utf8JsonReader(bytes);
        var under = false;
        var at = 0;

        while (reader.Read())
        {
            if (reader.CurrentDepth == 1 && reader.TokenType == JsonTokenType.PropertyName)
            {
                under = reader.ValueTextEquals(key);
            }
            else if (under && reader.CurrentDepth == 2 && reader.TokenType is not (JsonTokenType.EndObject or JsonTokenType.EndArray or JsonTokenType.PropertyName))
            {
                if (at++ == entry)
                {
                    return Placed(json, (int)reader.TokenStartIndex);
                }

                if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
                {
                    reader.Skip();
                }
            }
        }

        throw new ArgumentException($"There is no entry {entry} under '{key}'.", nameof(entry));
    }

    private static (int Line, int Column) Placed(string json, int offset)
    {
        var start = json.LastIndexOf('\n', Math.Max(offset - 1, 0)) + 1;

        return (json[..offset].Count(letter => letter == '\n') + 1, offset - start + 1);
    }
}
