// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The files DeepSharp 0.13.0 — the packages on nuget.org — wrote of a generated flock series: a pipeline that keeps the training
/// rows of the columns it watches, counted, beside a farm's history and how alike the parts are; networks declared in the words
/// 0.12.0 added and written by 0.13.0 — the activations, a vector for a category, a periodic embedding, a cross-entropy that softens
/// its labels — and a run judged by the average of its weights, with a checkpoint of it; a part of a network that moves at a rate of
/// its own beside a low-rank pair, and a variation of the training rows; a regression behind rows set aside to calibrate an interval
/// on, with the interval it was given; one slice of a small table; reviews cut into ids and the vector a network makes of them; and
/// three networks answering as one, in the one file they are kept in. This DeepSharp reads each of them and writes it again as
/// 0.13.0 wrote it, but for the number of the version a pipeline file names; serves what 0.13.0 served; measures a window as 0.13.0
/// measured it; and still knows every verb, every key, every word and every setting 0.13.0 knew, asking for none it did not ask for.
/// </summary>
/// <remarks>
/// What the release after 0.13.0 adds is held to one rule by this class: a file written before it keeps reading, and a file
/// that says none of what the release added is written exactly as it was. A setting a word gained must not be required of a
/// file written without it. The program that wrote the files, once, is <c>Fixtures/make-0.13.0.cs</c>.
/// </remarks>
public class FilesWrittenBy0130Tests
{
    private static readonly string Folder = Path.Join(AppContext.BaseDirectory, "Learners", "Fixtures");
    private static readonly JsonNode Facts = JsonNode.Parse(Text("facts.json"))!;

    public static TheoryData<string> Declared => ["vocabulary", "embedding", "periodic", "judged", "rateshare", "noise"];

    [Theory]
    [InlineData("series.pipeline.json")]
    [InlineData("table.pipeline.json")]
    [InlineData("vocabulary.pipeline.json")]
    [InlineData("embedding.pipeline.json")]
    [InlineData("periodic.pipeline.json")]
    [InlineData("judged.pipeline.json")]
    [InlineData("rateshare.pipeline.json")]
    [InlineData("noise.pipeline.json")]
    [InlineData("interval.pipeline.json")]
    [InlineData("fold.pipeline.json")]
    [InlineData("reviews.pipeline.json")]
    public void ADeclaration_ReadsBack_AndIsWrittenAsItWasWritten(string name) =>
        Assert.Equal(AtThisVersion(Text(name)), PipelineDeclaration.FromJson(Text(name), Catalog()).ToJson());

    [Theory]
    [InlineData("series.run.json")]
    [InlineData("table.run.json")]
    [InlineData("fold.run.json")]
    public void AFitted_PipelineReadsBack_WithWhatItLearned_AndIsWrittenAsItWasWritten(string name) =>
        Assert.Equal(AtThisVersion(Text(name)), PreparedData.FromJson(Text(name), Catalog()).ToJson());

    [Fact]
    public void ThePredictionsItMeasured_ReadBack_AndAreMeasuredAgainToWhatItMeasured()
    {
        var predictions = Text("series.predictions.json");
        var measured = Prepared().MeasureAgain(predictions);

        Assert.Equal(predictions, WrittenPredictions.FromJson(predictions).ToJson());

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

    [Theory]
    [InlineData("series.network.json", "seriesAnswers")]
    [InlineData("table.network.json", "tableAnswers")]
    [InlineData("judged.fitted.network.json", "judgedFittedAnswers")]
    public void ANetworksOneFile_ReadsBack_ServesWhat0130Served_AndIsWrittenAsItWasWritten(string name, string answers)
    {
        // The table's network was trained by a focal loss behind measures of classes, and the judged run keeps an average its
        // early stopping is judged by. Each is written again to the byte, so a part that says none of what came after is the part
        // 0.13.0 wrote.
        var file = Text(name);
        var trained = TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), Catalog());

        Told(Facts[answers]!, trained.Predict(Served()));
        Assert.Equal(file, trained.ToJson());
    }

    [Theory]
    [MemberData(nameof(Declared))]
    public void ANetworkDeclaredWithWhat0130LetItSay_ReadsBack_ServesWhat0130Served_AndIsWrittenAsItWasWritten(string name)
    {
        var file = Text($"{name}.network.json");
        var trained = TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), Catalog());

        Assert.Equal((int)Facts["declared"]![name]!["keptEpoch"]!, trained.TrainedOn.Epoch);
        Told(Facts["declared"]![name]!["answers"]!, trained.Predict(Served()));
        Assert.Equal(file, trained.ToJson());
    }

    [Fact]
    public void ARegressionGivenAnInterval_ReadsBack_KeepsTheMarginsItWasGiven_ServesWhat0130Served_AndIsWrittenAsItWasWritten()
    {
        // The margins are worked out from the errors on the rows set aside to calibrate on, and are part of the file: they come back
        // as they were, and are not worked out again.
        var file = Text("interval.network.json");
        var trained = TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), Catalog());

        Assert.Equal(Facts["intervalMargins"]!.AsArray().Select(margin => Double(margin!)), trained.Interval!.Margins);
        Assert.Equal((int)Facts["intervalRows"]!, trained.Interval.Rows);
        Told(Facts["intervalAnswers"]!, trained.Predict(Served()));
        Assert.Equal(file, trained.ToJson());
    }

    [Fact]
    public void AnEnsembleOfThreeNetworks_ReadsBack_AnswersAsOne_AndIsWrittenAsItWasWritten()
    {
        var file = Text("ensemble.json");
        var ensemble = TrainedEnsemble.FromJson(file, NetworkCatalog.BuiltIn(), Catalog());

        Assert.Equal(3, ensemble.Members.Count);
        Told(Facts["ensembleAnswers"]!, ensemble.Predict(Served()));
        Assert.Equal(file, ensemble.ToJson());
    }

    [Fact]
    public void ReviewsCutIntoIds_ReadBack_ServeWhat0130Served_AndAreWrittenAsTheyWereWritten()
    {
        var file = Text("reviews.network.json");
        var trained = TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), Catalog());

        Told(Facts["reviewsAnswers"]!, trained.Predict(CsvRowSource.FromText(Text("reviews.served.csv"))));
        Assert.Equal(file, trained.ToJson());
    }

    [Fact]
    public void ACheckpointOfARunJudgedByItsAverage_ReadsBack_ServesAsTheNetworkItWas_AndIsWrittenAsItWasWritten()
    {
        // A run that keeps an average, judged by it, beside early stopping that restores the best: it reads as 0.13.0 judged it
        // and is written again so.
        var file = Text("judged.checkpoint.json");
        var prepared = Prepared();
        var resumed = CheckpointFile.Read(file, NetworkCatalog.BuiltIn(), prepared);

        Assert.Equal((int)Facts["judgedCheckpointEpochs"]!, resumed.Checkpoint.Epochs);
        Told(Facts["judgedCheckpointAnswers"]!, TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), Catalog()).Predict(Served()));

        var rewritten = CheckpointFile.Write(resumed.Compiled, prepared, resumed.Checkpoint);

        Assert.Equal(Part(file, "training"), Part(rewritten, "training"));

        foreach (var key in new[] { "layers", "parameters", "state", "loss" })
        {
            Assert.Equal(Part(file, "network", key), Part(rewritten, "network", key));
        }
    }

    [Fact]
    public void ACheckpointTakenBy0130_IsNotGoneOnFromUnderAnotherVersionOfTheEngine()
    {
        // The light engine names DeepSharp's own version, and a run goes on from a checkpoint on the same engine in the same
        // version only, which the refusal says in the version it names.
        var file = Text("judged.checkpoint.json");
        var prepared = Prepared();
        var resumed = CheckpointFile.Read(file, NetworkCatalog.BuiltIn(), prepared);
        var options = new FitOptions(seed: 20261017)
        {
            Epochs = 4,
            BatchSize = 16,
            Average = 0.9,
            JudgedByAverage = true,
            EarlyStopping = new EarlyStopping { Patience = 2, RestoreBest = true },
            ResumeFrom = resumed.Checkpoint,
        };

        var refused = Assert.Throws<ArgumentException>(() => resumed.Compiled.Fit(prepared, options));

        Assert.Contains("0.13.0", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineFittedBy0130_KeepsTheTrainingRowsItWatched_AndMeasuresAWindowAsItDid()
    {
        // 0.13.0 kept the training rows of the columns it watched, counted, in the fit it wrote, and measured a window of served
        // rows against them: the file this release reads measures the same window to the same distances, flagged the same.
        var window = PreparedData.FromJson(Text("series.run.json"), Catalog()).Watched(Served());
        var told = Facts["watched"]!;

        Assert.Equal((int)told["rows"]!, window.Rows);

        foreach (var column in told["columns"]!.AsArray())
        {
            var measured = window.Columns.Single(each => each.Column == (string)column!["column"]!);

            Near(Double(column!["apart"]!), measured.Apart);
            Assert.Equal((bool)column["above"]!, measured.Above);
            Assert.Equal((int)column["trainingRows"]!, measured.TrainingRows);
            Assert.Equal((int)column["windowRows"]!, measured.WindowRows);
        }

        Assert.Equal(told["columns"]!.AsArray().Count, window.Columns.Count);
    }

    [Fact]
    public void ANetworkTrainedBehindAWatchingFit_IsRefusedBesideAFitThatWatchesOtherColumns()
    {
        // The counts are part of the fit, so a fit that watches other columns is another fit, and the network trained behind the
        // first is refused beside it.
        var declaration = JsonNode.Parse(Text("series.pipeline.json"))!;
        var watch = declaration["declaration"]!.AsArray().Single(step => (string)step!["step"]! == WatchStep.Name)!;

        watch["columns"] = new JsonArray("Age");

        var other = new Pipeline(PipelineDeclaration.FromJson(declaration.ToJsonString(), Catalog()), rows: null, SourceFolder.Of(Folder)).Run();
        var network = JsonNode.Parse(Text("series.network.json"))!;

        network["pipeline"] = JsonNode.Parse(other.ToJson());

        var refused = Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(network.ToJsonString(), NetworkCatalog.BuiltIn(), Catalog()));

        Assert.Contains("trained behind another fit of its pipeline", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryVerbAndKey0130Knew_IsStillKnown_MeansWhatItMeant_AndNothingNewIsRequiredOfAFileWrittenBeforeIt()
    {
        var catalog = Catalog();

        foreach (var verb in JsonNode.Parse(Text("verbs.json"))!.AsArray())
        {
            var name = (string)verb!["verb"]!;

            Assert.True(catalog.Knows(name), $"The verb '{name}' was known to 0.13.0 and is not known now.");

            var now = catalog.Describe(name);
            var keys = Words(verb["keys"]!);
            var required = Words(verb["required"]!);
            var requiredNow = now.Parameters.SelectMany(parameter => parameter.RequiredKeys).ToHashSet(StringComparer.Ordinal);

            Assert.Equal((int)verb["since"]!, now.Since);
            Assert.True(keys.IsSubsetOf(now.Keys), $"'{name}' lost {string.Join(", ", keys.Except(now.Keys))}, which 0.13.0 wrote.");
            Assert.True(requiredNow.IsSubsetOf(required), $"'{name}' now requires {string.Join(", ", requiredNow.Except(required))}, which a file written by 0.13.0 does not hold.");
        }
    }

    [Fact]
    public void EveryPartWordAndSetting0130Knew_IsStillKnown_AndNothingNewIsRequiredOfAPartWrittenWithoutIt()
    {
        var families = new Dictionary<string, IReadOnlyList<PartKind>>
        {
            ["layers"] = NetworkWords.Layers,
            ["optimizers"] = NetworkWords.Optimizers,
            ["losses"] = NetworkWords.Losses,
            ["schedules"] = NetworkWords.Schedules,
            ["stopping"] = NetworkWords.Stopping,
        };

        var words = JsonNode.Parse(Text("words.json"))!.AsObject();

        Assert.Equal(families.Keys.Order(StringComparer.Ordinal), words.Select(family => family.Key).Order(StringComparer.Ordinal));

        foreach (var (family, kinds) in families)
        {
            foreach (var (word, settings) in words[family]!.AsObject())
            {
                var kind = kinds.SingleOrDefault(each => each.Name == word);
                var knew = Words(settings!).ToHashSet(StringComparer.Ordinal);

                Assert.True(kind is not null, $"The {family} word '{word}' was known to 0.13.0 and is not known now.");
                Assert.True(
                    knew.IsSubsetOf(kind.Settings.Select(setting => setting.Key)),
                    $"'{word}' lost {string.Join(", ", knew.Except(kind.Settings.Select(setting => setting.Key)))}, which 0.13.0 wrote.");

                foreach (var gained in kind.Settings.Where(setting => !knew.Contains(setting.Key)))
                {
                    Assert.True(
                        gained.RequiredKeys.Count == 0,
                        $"'{word}' now requires '{gained.Key}', which a part written by 0.13.0 does not hold.");
                }
            }
        }
    }

    private static string Part(string file, params string[] path)
    {
        using var document = System.Text.Json.JsonDocument.Parse(file);
        var at = document.RootElement;

        foreach (var key in path)
        {
            at = at.GetProperty(key);
        }

        return at.GetRawText();
    }

    // The version a pipeline file names is the one thing a file read and written again may differ in: the file says which
    // version it was written against, and this library writes the one it is.
    private static string AtThisVersion(string file) =>
        file.Replace("\"version\": 12,", string.Create(CultureInfo.InvariantCulture, $"\"version\": {PipelineDeclaration.Version},"), StringComparison.Ordinal);

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

    // A number worked out through the machine's own exponential, held to within a float's width of what 0.13.0 printed.
    private static void Near(double expected, double actual) => Assert.Equal(expected, actual, (Math.Abs(expected) * 1e-7) + 1e-9);

    private static double Double(JsonNode node) => double.Parse((string)node!, CultureInfo.InvariantCulture);

    private static HashSet<string> Words(JsonNode node) => [.. node.AsArray().Select(word => (string)word!)];

    private static string Text(string name) => File.ReadAllText(Path.Join(Folder, $"flocks-0.13.0.{name}"));

    private static StepCatalog Catalog() => StepCatalog.BuiltIn().WithNetworks();

    private static IRowSource Served() => CsvRowSource.FromText(Text("served.csv"));

    // The series fitted here, behind the file 0.13.0 wrote of it, from the rows 0.13.0 was given.
    private static PreparedData Prepared() =>
        new Pipeline(PipelineDeclaration.FromJson(Text("series.pipeline.json"), Catalog()), rows: null, SourceFolder.Of(Folder)).Run();
}
