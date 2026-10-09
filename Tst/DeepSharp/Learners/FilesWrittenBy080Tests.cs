// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The files DeepSharp 0.8.0 — the packages on nuget.org — wrote of a generated flock series: its pipelines, what they
/// fitted, the preset and the courses, a network trained behind the series and its checkpoint, and the predictions it
/// measured. This DeepSharp reads each of them and writes it again as 0.8.0 wrote it, but for the number of the version the
/// file names; serves what 0.8.0 served; goes on from the checkpoint to what an uninterrupted run came to; and still knows
/// every verb and every key 0.8.0 knew, asking for no key it did not ask for.
/// </summary>
/// <remarks>
/// What the release after 0.8.0 adds is held to one rule by this class: a file written before it keeps reading. A key a verb
/// gained must not be required of a file that was written without it, and a word it gained is refused by name by the library
/// that does not know it, not by this one. The program that wrote the files, once, is <c>Fixtures/make-0.8.0.cs</c>.
/// </remarks>
public class FilesWrittenBy080Tests
{
    private static readonly string Folder = Path.Join(AppContext.BaseDirectory, "Learners", "Fixtures");
    private static readonly JsonNode Facts = JsonNode.Parse(Text("facts.json"))!;

    [Theory]
    [InlineData("series.pipeline.json")]
    [InlineData("table.pipeline.json")]
    [InlineData("ahead.pipeline.json")]
    public void ADeclaration_ReadsBack_AndIsWrittenAsItWasWritten(string name) =>
        Assert.Equal(AtThisVersion(Text(name)), PipelineDeclaration.FromJson(Text(name), Catalog()).ToJson());

    [Theory]
    [InlineData("series.run.json")]
    [InlineData("table.run.json")]
    [InlineData("ahead.run.json")]
    public void AFitted_PipelineReadsBack_WithWhatItLearned_AndIsWrittenAsItWasWritten(string name) =>
        Assert.Equal(AtThisVersion(Text(name)), PreparedData.FromJson(Text(name), Catalog()).ToJson());

    [Fact]
    public void ThePreset_ReadsBack_AndIsWrittenAsItWasWritten() =>
        Assert.Equal(AtThisVersion(Text("series.preset.json")), PipelinePreset.FromJson(Text("series.preset.json"), Catalog()).ToJson());

    [Theory]
    [InlineData("series.course.json")]
    [InlineData("table.course.json")]
    public void ACourse_ReadsBack_AndIsWrittenAsItWasWritten(string name) =>
        Assert.Equal(AtThisVersion(Text(name)), PipelineCourse.FromJson(Text(name), Catalog()).ToJson());

    [Fact]
    public void ThePredictionsItMeasured_ReadBack_AndAreWrittenAsTheyWere()
    {
        var predictions = Text("series.predictions.json");

        Assert.Equal(predictions, WrittenPredictions.FromJson(predictions).ToJson());
    }

    [Fact]
    public void ThePredictions_AreMeasuredAgainOnThePipelineTheyWereMadeBehind_ToWhatItMeasured()
    {
        var measured = Prepared().MeasureAgain(Text("series.predictions.json"));

        foreach (var part in Facts["measures"]!.AsArray())
        {
            var again = measured.Parts.Single(each => each.Part.ToString() == (string)part!["part"]!);

            Assert.Equal((int)part!["rows"]!, again.Rows);

            foreach (var value in part["values"]!.AsArray())
            {
                var one = again.Values.Single(each => each.Metric.ToString() == (string)value!["metric"]!);

                Near(Double(value!["value"]!), one.Value);
                Near(Double(value["baseline"]!), one.Baseline);
            }
        }
    }

    [Fact]
    public void TheNetworksOneFile_ReadsBack_ServesWhat080Served_AndIsWrittenAsItWasWritten()
    {
        var file = Text("series.network.json");
        var trained = TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), Catalog());

        Assert.Equal((int)Facts["seriesKeptEpoch"]!, trained.TrainedOn.Epoch);
        Told(Facts["seriesAnswers"]!, trained.Predict(Served()));
        Assert.Equal(file, trained.ToJson());
    }

    [Fact]
    public void ANetworkWrittenInKerasWordsAndFittedByHand_ReadsBack_AndServesWhat080Served()
    {
        var file = Text("series.fitted.network.json");
        var trained = TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), Catalog());

        Told(Facts["fittedAnswers"]!, trained.Predict(Served()));
        Assert.Equal(file, trained.ToJson());
    }

    [Fact]
    public void TheCheckpoint_ReadsBack_ServesAsTheNetworkItWas_AndIsNotGoneOnFromUnderAnotherVersionOfTheEngine()
    {
        var file = Text("series.checkpoint.json");
        var prepared = Prepared();
        var resumed = CheckpointFile.Read(file, NetworkCatalog.BuiltIn(), prepared);

        Assert.Equal((int)Facts["checkpointEpochs"]!, resumed.Checkpoint.Epochs);
        Told(Facts["checkpointAnswers"]!, TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), Catalog()).Predict(Served()));

        // The light engine names DeepSharp's own version, and a run goes on from a checkpoint on the same engine in the same
        // version only: another version could round every step otherwise. A checkpoint taken by 0.8.0 is therefore read and
        // served by every later release, and gone on from by 0.8.0 alone, which the refusal says in the version it names.
        var options = new FitOptions(seed: 20261009) { Epochs = (int)Facts["fittedEpochs"]!, BatchSize = 16, ResumeFrom = resumed.Checkpoint };
        var refused = Assert.Throws<ArgumentException>(() => resumed.Compiled.Fit(prepared, options));

        Assert.Contains("0.8.0", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryVerbAndKey080Knew_IsStillKnown_MeansWhatItMeant_AndNothingNewIsRequiredOfAFileWrittenBeforeIt()
    {
        var catalog = Catalog();

        foreach (var verb in JsonNode.Parse(Text("verbs.json"))!.AsArray())
        {
            var name = (string)verb!["verb"]!;

            Assert.True(catalog.Knows(name), $"The verb '{name}' was known to 0.8.0 and is not known now.");

            var now = catalog.Describe(name);
            var keys = Words(verb["keys"]!);
            var required = Words(verb["required"]!);
            var requiredNow = now.Parameters.SelectMany(parameter => parameter.RequiredKeys).ToHashSet(StringComparer.Ordinal);

            Assert.Equal((int)verb["since"]!, now.Since);
            Assert.True(keys.IsSubsetOf(now.Keys), $"'{name}' lost {string.Join(", ", keys.Except(now.Keys))}, which 0.8.0 wrote.");
            Assert.True(requiredNow.IsSubsetOf(required), $"'{name}' now requires {string.Join(", ", requiredNow.Except(required))}, which a file written by 0.8.0 does not hold.");
        }
    }

    // The version a file names is the one thing a file read and written again may differ in: the file says which version it
    // was written against, and this library writes the one it is.
    private static string AtThisVersion(string file) =>
        file.Replace("\"version\": 7,", string.Create(CultureInfo.InvariantCulture, $"\"version\": {PipelineDeclaration.Version},"), StringComparison.Ordinal);

    private static void Told(JsonNode expected, Predictions told)
    {
        var rows = expected.AsArray();

        Assert.Equal(rows.Count, told.Answers.Count);

        for (var row = 0; row < rows.Count; row++)
        {
            var answers = rows[row]!.AsArray();

            Assert.Equal(answers.Count, told.Answers[row].Length);

            for (var at = 0; at < answers.Count; at++)
            {
                Near(Double(answers[at]!), told.Answers[row][at]);
            }
        }
    }

    // A number worked out through the machine's own exponential, held to within a float's width of what 0.8.0 printed.
    private static void Near(double expected, double actual) => Assert.Equal(expected, actual, (Math.Abs(expected) * 1e-7) + 1e-9);

    private static double Double(JsonNode node) => double.Parse((string)node!, CultureInfo.InvariantCulture);

    private static HashSet<string> Words(JsonNode node) => [.. node.AsArray().Select(word => (string)word!)];

    private static string Text(string name) => File.ReadAllText(Path.Join(Folder, $"flocks-0.8.0.{name}"));

    private static StepCatalog Catalog() => StepCatalog.BuiltIn().WithNetworks();

    private static IRowSource Served() => CsvRowSource.FromText(Text("served.csv"));

    // The series fitted here, behind the file 0.8.0 wrote of it, from the rows 0.8.0 was given.
    private static PreparedData Prepared() =>
        new Pipeline(PipelineDeclaration.FromJson(Text("series.pipeline.json"), Catalog()), rows: null, SourceFolder.Of(Folder)).Run();
}
