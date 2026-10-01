// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;
using DeepSharp.Tests.Learners;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a model predicted for the parts its pipeline's report measures crosses as text to wherever the same pipeline runs —
/// a notebook's report block, whose types are not a C# cell's even when their names are — and is measured there again, on
/// that run: each part's rows found by their keys, in the units they were handed over in, with what the model said it
/// learned nothing about. The same predictions measured again, on the very fit they were made behind, are the same
/// measures; predictions made behind any other fit — other steps, or other numbers learned from other rows, of a feature
/// or of the answer — are refused before anything is measured, by the rule a network's file is held to its pipeline by, and
/// so are predictions of another answer, or of other rows.
/// </summary>
public class HandBackTests
{
    private static PreparedData Passengers(int seed = 20260923, string target = "survived", Metric[]? metrics = null, Scale pclass = Scale.MidRange) =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass").Category("sex"))
            .SplitStratified("survived", 0.70, 0.15, seed)
            .EncodeCategories()
            .Normalise("pclass", pclass)
            .Target(target)
            .Report(report => report
                .Measure(metrics ?? [Metric.Accuracy, Metric.Precision, Metric.ConfusionMatrix, Metric.Rmse])
                .On(Part.Train, Part.Validation, Part.Test)
                .As(Shown.Numbers, Shown.Drawn))
            .Build()
            .Run();

    // Most women and few men; and of every tenth row the model says it learned nothing about its class.
    private static PartPredictions Guessed(Batch batch)
    {
        var female = batch.FeatureNames.ToList().IndexOf("sex_female");

        return new PartPredictions(batch, [.. batch.Features.Select(row => new[] { row[female] == 1 ? 0.83 : 0.21 })])
        {
            Unfamiliar = [.. Enumerable.Range(0, batch.RowCount).Select(row => row % 10 == 0 ? (IReadOnlyList<string>)["pclass"] : [])],
        };
    }

    private static Measures Measured(PreparedData prepared) =>
        prepared.Measure([.. prepared.Declaration.Report!.Parts.Select(part => Guessed(prepared.Batch(part)))]);

    [Fact]
    public void PredictionsMeasuredAgainFromTheirText_AreTheSameMeasures()
    {
        var prepared = Passengers();
        var measures = Measured(prepared);

        var again = prepared.MeasureAgain(measures.PredictionsToJson());

        Assert.Equal(measures.Metrics, again.Metrics);
        Assert.Equal(measures.Shown, again.Shown);
        Assert.Equal(measures.Answers, again.Answers);
        Assert.Equal(measures.Parts.Select(part => part.Part), again.Parts.Select(part => part.Part));
        Assert.Equal(measures.Parts.Select(part => part.Rows), again.Parts.Select(part => part.Rows));
        Assert.Equal(measures.Parts.Select(part => part.UnfamiliarRows), again.Parts.Select(part => part.UnfamiliarRows));
        Assert.Equal(measures.Parts.SelectMany(part => part.Values), again.Parts.SelectMany(part => part.Values));
        Assert.Equal(
            measures.Parts.SelectMany(part => part.Confusions).SelectMany(confusion => confusion.Counts).SelectMany(row => row),
            again.Parts.SelectMany(part => part.Confusions).SelectMany(confusion => confusion.Counts).SelectMany(row => row));
        Assert.Equal(measures.Parts.SelectMany(part => part.Predicted).SelectMany(row => row), again.Parts.SelectMany(part => part.Predicted).SelectMany(row => row));

        // Measured again, they are the same text.
        Assert.Equal(measures.PredictionsToJson(), again.PredictionsToJson());
        Assert.Equal(new[] { 63, 14, 14 }, again.Parts.Select(part => part.UnfamiliarRows!.Value));
    }

    [Fact]
    public void TheText_HoldsEachPartsRowsByTheirKeys_WhatWasPredictedAsItWasHandedOver_AndWhatTheModelSaidItLearnedNothingAbout()
    {
        // A fare scaled onto nought to one, predicted in those units: the text holds the numbers the model gave, and the way
        // back is the run's that measures them.
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("pclass").Number("fare"))
            .SplitStratified("pclass", 0.70, 0.15)
            .Normalise("fare", Scale.MinMax)
            .Target("fare")
            .Report(report => report.Measure(Metric.Rmse).On(Part.Test, Part.Validation).As(Shown.Numbers))
            .Build()
            .Run();
        var test = prepared.Batch(Part.Test);
        var validation = prepared.Batch(Part.Validation);
        var told = new PartPredictions(test, [.. test.Features.Select(row => new[] { row[0] / 7 })])
        {
            Unfamiliar = [.. test.Features.Select((_, row) => row == 2 ? (IReadOnlyList<string>)["pclass"] : [])],
        };

        var text = JsonNode.Parse(prepared.Measure([told, new PartPredictions(validation, [.. validation.Features.Select(_ => new[] { 0.1 })])]).PredictionsToJson())!;

        Assert.Equal(1, (int)text["version"]!);

        // The fit they were made behind, as two pipelines are compared: the digest of its text with the version left out, and
        // the version it names.
        Assert.Equal(PipelineText.Of(prepared).FitDigest, (string)text["pipeline"]!["fit"]!);
        Assert.Equal(PipelineDeclaration.Version, (int)text["pipeline"]!["version"]!);
        Assert.Equal(["fare"], text["answers"]!.AsArray().Select(answer => (string)answer!));

        // The parts in the order the report names them; a part the model said nothing of has no such list.
        var parts = text["parts"]!.AsArray();

        Assert.Equal(["test", "validation"], parts.Select(part => (string)part!["part"]!));
        Assert.Equal(test.Keys!.Select(key => key.ToString()), parts[0]!["keys"]!.AsArray().Select(key => (string)key!));
        Assert.Equal(test.Features.Select(row => row[0] / 7), parts[0]!["predictions"]!.AsArray().Select(row => (double)row!.AsArray().Single()!));
        Assert.Equal(["pclass"], parts[0]!["unfamiliar"]!.AsArray()[2]!.AsArray().Select(name => (string)name!));
        Assert.Empty(parts[0]!["unfamiliar"]!.AsArray()[1]!.AsArray());
        Assert.Null(parts[1]!["unfamiliar"]);
        Assert.Equal(64, ((string)parts[1]!["keys"]![0]!).Length);
    }

    [Fact]
    public void PredictionsMadeBehindAnotherFitOfAFeature_AreRefused_ThoughTheirRowsAndTheirAnswersAreTheSame()
    {
        // The class scaled one way or another: the same rows in every part, the same answer, the same numbers as answers — and
        // another fit, whose network would have been handed other numbers.
        var text = Measured(Passengers(pclass: Scale.MinMax)).PredictionsToJson();

        var refused = Assert.Throws<ArgumentException>(() => Passengers(pclass: Scale.Standard).MeasureAgain(text));

        Assert.Contains("made behind another fit", refused.Message, StringComparison.Ordinal);
        Assert.Contains("again", refused.Message, StringComparison.Ordinal);
        Assert.Equal("predictions", refused.ParamName);
    }

    [Fact]
    public void PredictionsMadeBehindAnotherFitOfTheAnswer_AreRefused_TheyWouldComeBackThroughAnotherWayBack()
    {
        // The fare scaled one way when the model was trained, another way now: the same rows and the same answer's name, and
        // each prediction would come back into pounds through a way back it was not made behind.
        var minMax = Fares(Scale.MinMax);
        var text = minMax.Measure([.. new[] { Part.Validation, Part.Test }.Select(part => Constant(minMax.Batch(part)))]).PredictionsToJson();

        Assert.Contains("made behind another fit", Assert.Throws<ArgumentException>(() => Fares(Scale.Standard).MeasureAgain(text)).Message, StringComparison.Ordinal);
        Assert.Equal(2, Fares(Scale.MinMax).MeasureAgain(text).Parts.Count);
    }

    [Fact]
    public void TheSameFitRunAgain_MeasuresThePredictionsItWasHanded_ToTheSameMeasures()
    {
        // Another run of the same steps over the same rows is the same fit, whoever ran it.
        var measures = Measured(Passengers());

        var again = Passengers().MeasureAgain(measures.PredictionsToJson());

        Assert.Equal(measures.Parts.SelectMany(part => part.Values), again.Parts.SelectMany(part => part.Values));
        Assert.Equal(measures.Parts.Select(part => part.UnfamiliarRows), again.Parts.Select(part => part.UnfamiliarRows));
    }

    [Fact]
    public void PredictionsOfAnotherSplit_AreMadeBehindAnotherFit_AndRefused()
    {
        // The same pipeline divided by another seed: as many rows in each part, and other rows in them.
        var text = Measured(Passengers(seed: 7)).PredictionsToJson();

        Assert.Contains("made behind another fit", Assert.Throws<ArgumentException>(() => Passengers().MeasureAgain(text)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PredictionsWhoseRowsAreNotThePartsRows_AreRefused_ByTheKeysOfTheRowsTheyWereMadeFor()
    {
        // Behind the same fit, the rows a text names are the part's; a text whose keys were written otherwise is refused by
        // them, as predictions of other rows or in another order always are.
        var prepared = Passengers();
        var text = JsonNode.Parse(Measured(prepared).PredictionsToJson())!;
        var keys = text["parts"]![0]!["keys"]!.AsArray();
        var first = (string)keys[0]!;

        keys[0] = (string)keys[1]!;
        keys[1] = first;

        var refused = Assert.Throws<ArgumentException>(() => prepared.MeasureAgain(text.ToJsonString()));

        Assert.Contains("'train'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("other rows than the part's, or in another order", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(1, "by a newer DeepSharp")]
    [InlineData(-1, "no DeepSharp")]
    public void PredictionsMadeBehindAPipelineOfAVersionNoComparisonKnows_AreRefused(int beyond, string said)
    {
        // A pipeline is compared, never read, so the version it names must be one in which every step means what it means now.
        var prepared = Passengers();
        var text = JsonNode.Parse(Measured(prepared).PredictionsToJson())!;

        text["pipeline"]!["version"] = beyond > 0 ? PipelineDeclaration.Version + beyond : PipelineText.FirstComparable + beyond;

        var refused = Assert.Throws<ArgumentException>(() => prepared.MeasureAgain(text.ToJsonString()));

        Assert.Contains(said, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PredictionsOfAnotherAnswer_AreRefused_NamingBoth()
    {
        var text = Measured(Passengers()).PredictionsToJson();

        var refused = Assert.Throws<ArgumentException>(() => Passengers(target: "pclass", metrics: [Metric.Rmse]).MeasureAgain(text));

        Assert.Contains("answer 'survived'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("answers 'pclass'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineWithoutTheReport_OrWithoutTheAnswer_IsAnotherFit_AndMeasuresNothing()
    {
        var text = Measured(Passengers()).PredictionsToJson();
        var answered = Pdd.Create().ReadCsv(Repository.Data("titanic.csv")).Declare(schema => schema.Integer("survived", "pclass"))
            .SplitStratified("survived", 0.70, 0.15).Target("survived").Build().Run();
        var unanswered = Pdd.Create().ReadCsv(Repository.Data("titanic.csv")).Declare(schema => schema.Integer("survived", "pclass"))
            .SplitStratified("survived", 0.70, 0.15).Build().Run();

        Assert.Contains("made behind another fit", Assert.Throws<ArgumentException>(() => answered.MeasureAgain(text)).Message, StringComparison.Ordinal);
        Assert.Contains("made behind another fit", Assert.Throws<ArgumentException>(() => unanswered.MeasureAgain(text)).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => answered.MeasureAgain(null!));
    }

    [Fact]
    public void PredictionsNotOneForEachRowTheyNameByKey_AreRefusedAsMeasuringRefusesThem()
    {
        var prepared = Passengers();
        var text = JsonNode.Parse(Measured(prepared).PredictionsToJson())!;

        text["parts"]![2]!["predictions"]!.AsArray().RemoveAt(0);

        Assert.Contains("predictions for", Assert.Throws<ArgumentException>(() => prepared.MeasureAgain(text.ToJsonString())).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WhatTheModelSaidNothingOf_WrittenAsNothing_IsNotCounted()
    {
        var prepared = Passengers();
        var text = JsonNode.Parse(Measured(prepared).PredictionsToJson())!;

        foreach (var part in text["parts"]!.AsArray())
        {
            part!["unfamiliar"] = null;
        }

        Assert.All(prepared.MeasureAgain(text.ToJsonString()).Parts, part => Assert.Null(part.UnfamiliarRows));
    }

    [Fact]
    public void ATextANewerDeepSharpWrote_IsRefusedAsNewer()
    {
        var refused = Assert.Throws<ArgumentException>(() => Passengers().MeasureAgain("""{"version": 2, "answers": [], "parts": []}"""));

        Assert.Contains("newer DeepSharp", refused.Message, StringComparison.Ordinal);
        Assert.Contains("version 2", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PredictionsFromALearnerOfAnotherKind_CarryTheStepsItsRunLeftOut_AndAreMeasuredAgainOnARunOfEveryStep()
    {
        // A tree trained on the run made for it, its predictions handed back as text and measured where the pipeline is run
        // with every step — as a notebook's report block runs it.
        var titanic = WikiTitanic.In(WikiTitanic.DataFolder);
        var places = titanic.RunFor(Stump.Needs);
        var tree = Stump.Fit(places.Batch(Part.Train, Stump.Needs));
        var measures = places.Measure([.. places.Declaration.Report!.Parts.Select(part => tree.Predict(places.Batch(part, Stump.Needs)))]);
        var text = measures.PredictionsToJson();

        var again = titanic.Run().MeasureAgain(text);

        Assert.Equal(
            [.. new[] { 4, 5, 6, 7, 8 }.Select(places.Declaration.KeyAt)],
            JsonNode.Parse(text)!["pipeline"]!["skipped"]!.AsArray().Select(prefix => (string)prefix!));
        Assert.Equal(measures.Parts.SelectMany(part => part.Values), again.Parts.SelectMany(part => part.Values));
        Assert.Equal(measures.Parts.Select(part => part.UnfamiliarRows), again.Parts.Select(part => part.UnfamiliarRows));
    }

    [Fact]
    public void PredictionsMadeBehindARunOfEveryStep_AreRefusedByARunThatLeftStepsOut_NamingThem()
    {
        var titanic = WikiTitanic.In(WikiTitanic.DataFolder);
        var every = titanic.Run();
        var tree = Stump.Fit(every.Batch(Part.Train, Stump.Needs));
        var text = every.Measure([.. every.Declaration.Report!.Parts.Select(part => tree.Predict(every.Batch(part, Stump.Needs)))]).PredictionsToJson();

        var refused = Assert.Throws<ArgumentException>(() => titanic.RunFor(Needs.Categories).MeasureAgain(text));

        // A run of every step writes the text it always wrote: the fit and its version, and nothing it left out.
        Assert.Equal(["fit", "version"], JsonNode.Parse(text)!["pipeline"]!.AsObject().Select(member => member.Key));
        Assert.Contains("step 5, 'encode.categories'; step 6, 'normalise'", refused.Message, StringComparison.Ordinal);
        Assert.Equal("predictions", refused.ParamName);
    }

    [Theory]
    [InlineData("a step no pipeline here has", "behind steps this pipeline does not have")]
    [InlineData("the fill", "is taken by every run")]
    [InlineData("the encoder alone", "no run for one learner leaves out exactly these")]
    public void PredictionsThatSayTheirRunLeftOutWhatNoRunOfThisPipelineLeavesOut_AreRefused(string what, string said)
    {
        var titanic = WikiTitanic.In(WikiTitanic.DataFolder);
        var places = titanic.RunFor(Stump.Needs);
        var tree = Stump.Fit(places.Batch(Part.Train, Stump.Needs));
        var text = JsonNode.Parse(places.Measure([.. places.Declaration.Report!.Parts.Select(part => tree.Predict(places.Batch(part, Stump.Needs)))]).PredictionsToJson())!;
        var skipped = text["pipeline"]!["skipped"]!.AsArray();

        switch (what)
        {
            case "a step no pipeline here has":
                skipped[0] = new string('0', 64);
                break;

            case "the fill":
                skipped[0] = places.Declaration.KeyAt(3);
                break;

            default:
                while (skipped.Count > 1)
                {
                    skipped.RemoveAt(1);
                }

                break;
        }

        var refused = Assert.Throws<ArgumentException>(() => titanic.Run().MeasureAgain(text.ToJsonString()));

        Assert.Contains(said, refused.Message, StringComparison.Ordinal);
        Assert.Equal("predictions", refused.ParamName);
    }

    // The fare predicted from the class, the fare scaled onto a range as the pipeline says: the same rows whatever the scale.
    private static PreparedData Fares(Scale fare) =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("pclass").Number("fare"))
            .SplitStratified("pclass", 0.70, 0.15)
            .Normalise("fare", fare)
            .Target("fare")
            .Report(report => report.Measure(Metric.Rmse).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .Build()
            .Run();

    private static PartPredictions Constant(Batch batch) => new(batch, [.. batch.Features.Select(_ => new[] { 0.25 })]);

    // A key of the right length, and one of the right length that is no number written in hexadecimal.
    private const string Key = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    private const string NotHex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg";

    [Theory]
    [InlineData("not a text of predictions", "not JSON")]
    [InlineData("""[]""", "not one JSON object")]
    [InlineData("""{}""", "names no version")]
    [InlineData("""{"version": "1"}""", "names no version")]
    [InlineData("""{"version": 1.5}""", "names no version")]
    [InlineData("""{"version": 0}""", "names no version")]
    [InlineData("""{"version": 1}""", "has no 'answers' list")]
    [InlineData("""{"version": 1, "answers": "survived"}""", "has no 'answers' list")]
    [InlineData("""{"version": 1, "answers": [1]}""", "which is not text")]
    [InlineData("""{"version": 1, "answers": ["survived"]}""", "has no 'parts' list")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [1]}""", "part 1 is not an object")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{}]}""", "part 1 names no part")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{"part": 3}]}""", "part 1 names no part")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{"part": "somewhere"}]}""", "'somewhere'")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{"part": "test"}]}""", "part 1 has no 'keys' list")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{"part": "test", "keys": [7]}]}""", "which is not text")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{"part": "test", "keys": ["0123"]}]}""", "'0123' is no row's key")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{"part": "test", "keys": [""" + "\"" + NotHex + "\"" + """]}]}""", "is no row's key")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{"part": "test", "keys": [""" + "\"" + Key + "\"" + """]}]}""", "part 1 has no 'predictions' list")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{"part": "test", "keys": [], "predictions": [0.5]}]}""", "not a list of numbers")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{"part": "test", "keys": [], "predictions": [["half"]]}]}""", "not a list of numbers")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{"part": "test", "keys": [], "predictions": [], "unfamiliar": "pclass"}]}""", "not a list of names")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{"part": "test", "keys": [], "predictions": [], "unfamiliar": ["pclass"]}]}""", "not a list of names")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [{"part": "test", "keys": [], "predictions": [], "unfamiliar": [[1]]}]}""", "which is not text")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": []}""", "names no pipeline")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [], "pipeline": "fit"}""", "names no pipeline")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [], "pipeline": {"version": 3}}""", "names no fit")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [], "pipeline": {"fit": 3, "version": 3}}""", "names no fit")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [], "pipeline": {"fit": "0a"}}""", "names no version of the pipeline file")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [], "pipeline": {"fit": "0a", "version": 3.5}}""", "names no version of the pipeline file")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [], "pipeline": {"fit": "0a", "version": 4, "skipped": "normalise"}}""", "under 'skipped'")]
    [InlineData("""{"version": 1, "answers": ["survived"], "parts": [], "pipeline": {"fit": "0a", "version": 4, "skipped": [4]}}""", "under 'skipped'")]
    public void ATextThatIsNotPredictions_IsRefused_SayingWhatIsWrongWithIt(string text, string said)
    {
        var refused = Assert.Throws<ArgumentException>(() => Passengers().MeasureAgain(text));

        Assert.Contains("not the text", refused.Message, StringComparison.Ordinal);
        Assert.Contains(said, refused.Message, StringComparison.Ordinal);
        Assert.Equal("predictions", refused.ParamName);
    }

    [Fact]
    public void TheNumbers_ComeBackAsTheyWereWritten_InEveryCulture()
    {
        var prepared = Passengers();
        var before = CultureInfo.CurrentCulture;

        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("nl-NL");
            var text = Measured(prepared).PredictionsToJson();

            Assert.Contains("0.83", text, StringComparison.Ordinal);
            Assert.Equal(JsonValueKind.Object, JsonDocument.Parse(text).RootElement.ValueKind);
            Assert.Equal(Measured(prepared).Parts.SelectMany(part => part.Values), prepared.MeasureAgain(text).Parts.SelectMany(part => part.Values));
        }
        finally
        {
            CultureInfo.CurrentCulture = before;
        }
    }
}
