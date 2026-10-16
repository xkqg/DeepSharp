// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The files DeepSharp 0.10.0 — the packages on nuget.org — wrote of a generated flock series: networks declared with
/// everything 0.10.0 let a part be told, under each optimizer's options and with a run that clips each value, steps once every
/// two batches and keeps an average of its weights while early stopping watches it; a checkpoint of that run; a network a
/// binary cross-entropy trained behind a table; one behind a padding that mirrors the example; and the words a declared network
/// is written in. This DeepSharp reads each of them and writes it again as 0.10.0 wrote it, but for the number of the version a
/// pipeline file names; serves what 0.10.0 served; and still knows every verb, every key, every word and every setting 0.10.0
/// knew, asking for none it did not ask for.
/// </summary>
/// <remarks>
/// What the release after 0.10.0 adds is held to one rule by this class: a file written before it keeps reading, and a file
/// that says none of what the release added is written exactly as it was. A setting a word gained must not be required of a
/// file written without it. The program that wrote the files, once, is <c>Fixtures/make-0.10.0.cs</c>.
/// </remarks>
public class FilesWrittenBy0100Tests
{
    private static readonly string Folder = Path.Join(AppContext.BaseDirectory, "Learners", "Fixtures");
    private static readonly JsonNode Facts = JsonNode.Parse(Text("facts.json"))!;

    public static TheoryData<string> Declared => ["parts", "sgd", "adam", "adamw", "rmsprop", "nadam", "run"];

    [Theory]
    [InlineData("series.pipeline.json")]
    [InlineData("table.pipeline.json")]
    [InlineData("parts.pipeline.json")]
    [InlineData("sgd.pipeline.json")]
    [InlineData("adam.pipeline.json")]
    [InlineData("adamw.pipeline.json")]
    [InlineData("rmsprop.pipeline.json")]
    [InlineData("nadam.pipeline.json")]
    [InlineData("run.pipeline.json")]
    public void ADeclaration_ReadsBack_AndIsWrittenAsItWasWritten(string name) =>
        Assert.Equal(AtThisVersion(Text(name)), PipelineDeclaration.FromJson(Text(name), Catalog()).ToJson());

    [Theory]
    [InlineData("series.run.json")]
    [InlineData("table.run.json")]
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
    [InlineData("series.mirrored.network.json", "mirroredAnswers")]
    [InlineData("averaged.fitted.network.json", "averagedFittedAnswers")]
    public void ANetworksOneFile_ReadsBack_ServesWhat0100Served_AndIsWrittenAsItWasWritten(string name, string answers)
    {
        // The table's network was trained by a binary cross-entropy, whose part the release after this one may give a setting
        // to; the mirrored padding and the averaged run are network and training parts of the fifth version. Each is written
        // again to the byte, so a part that says none of what came after is the part 0.10.0 wrote.
        var file = Text(name);
        var trained = TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), Catalog());

        Told(Facts[answers]!, trained.Predict(Served()));
        Assert.Equal(file, trained.ToJson());
    }

    [Theory]
    [MemberData(nameof(Declared))]
    public void ANetworkDeclaredWithWhat0100LetItSay_ReadsBack_ServesWhat0100Served_AndIsWrittenAsItWasWritten(string name)
    {
        var file = Text($"{name}.network.json");
        var trained = TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), Catalog());

        Assert.Equal((int)Facts["declared"]![name]!["keptEpoch"]!, trained.TrainedOn.Epoch);
        Told(Facts["declared"]![name]!["answers"]!, trained.Predict(Served()));
        Assert.Equal(file, trained.ToJson());
    }

    [Fact]
    public void ACheckpointOfARunThatKeptAnAverage_ReadsBack_ServesAsTheNetworkItWas_AndIsWrittenAsItWasWritten()
    {
        // A run that keeps an average beside early stopping that does not restore: what the next release lets a run say about
        // how such an epoch is judged, this checkpoint never said, so it reads as 0.10.0 judged it and is written again so.
        var file = Text("averaged.checkpoint.json");
        var prepared = Prepared();
        var resumed = CheckpointFile.Read(file, NetworkCatalog.BuiltIn(), prepared);

        Assert.Equal((int)Facts["averagedCheckpointEpochs"]!, resumed.Checkpoint.Epochs);
        Told(Facts["averagedCheckpointAnswers"]!, TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), Catalog()).Predict(Served()));

        var rewritten = CheckpointFile.Write(resumed.Compiled, prepared, resumed.Checkpoint);

        Assert.Equal(Part(file, "training"), Part(rewritten, "training"));

        foreach (var key in new[] { "layers", "parameters", "state", "loss" })
        {
            Assert.Equal(Part(file, "network", key), Part(rewritten, "network", key));
        }
    }

    [Fact]
    public void ACheckpointTakenBy0100_IsNotGoneOnFromUnderAnotherVersionOfTheEngine()
    {
        // The light engine names DeepSharp's own version, and a run goes on from a checkpoint on the same engine in the same
        // version only, which the refusal says in the version it names.
        var file = Text("averaged.checkpoint.json");
        var prepared = Prepared();
        var resumed = CheckpointFile.Read(file, NetworkCatalog.BuiltIn(), prepared);
        var options = new FitOptions(seed: 20261016)
        {
            Epochs = 4,
            BatchSize = 16,
            ClipValue = 0.5,
            StepEvery = 2,
            Average = 0.9,
            EarlyStopping = new EarlyStopping { Patience = 2 },
            ResumeFrom = resumed.Checkpoint,
        };

        var refused = Assert.Throws<ArgumentException>(() => resumed.Compiled.Fit(prepared, options));

        Assert.Contains("0.10.0", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryVerbAndKey0100Knew_IsStillKnown_MeansWhatItMeant_AndNothingNewIsRequiredOfAFileWrittenBeforeIt()
    {
        var catalog = Catalog();

        foreach (var verb in JsonNode.Parse(Text("verbs.json"))!.AsArray())
        {
            var name = (string)verb!["verb"]!;

            Assert.True(catalog.Knows(name), $"The verb '{name}' was known to 0.10.0 and is not known now.");

            var now = catalog.Describe(name);
            var keys = Words(verb["keys"]!);
            var required = Words(verb["required"]!);
            var requiredNow = now.Parameters.SelectMany(parameter => parameter.RequiredKeys).ToHashSet(StringComparer.Ordinal);

            Assert.Equal((int)verb["since"]!, now.Since);
            Assert.True(keys.IsSubsetOf(now.Keys), $"'{name}' lost {string.Join(", ", keys.Except(now.Keys))}, which 0.10.0 wrote.");
            Assert.True(requiredNow.IsSubsetOf(required), $"'{name}' now requires {string.Join(", ", requiredNow.Except(required))}, which a file written by 0.10.0 does not hold.");
        }
    }

    [Fact]
    public void EveryPartWordAndSetting0100Knew_IsStillKnown_AndNothingNewIsRequiredOfAPartWrittenWithoutIt()
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

                Assert.True(kind is not null, $"The {family} word '{word}' was known to 0.10.0 and is not known now.");
                Assert.True(
                    knew.IsSubsetOf(kind.Settings.Select(setting => setting.Key)),
                    $"'{word}' lost {string.Join(", ", knew.Except(kind.Settings.Select(setting => setting.Key)))}, which 0.10.0 wrote.");

                foreach (var gained in kind.Settings.Where(setting => !knew.Contains(setting.Key)))
                {
                    Assert.True(
                        gained.RequiredKeys.Count == 0,
                        $"'{word}' now requires '{gained.Key}', which a part written by 0.10.0 does not hold.");
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
        file.Replace("\"version\": 9,", string.Create(CultureInfo.InvariantCulture, $"\"version\": {PipelineDeclaration.Version},"), StringComparison.Ordinal);

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

    // A number worked out through the machine's own exponential, held to within a float's width of what 0.10.0 printed.
    private static void Near(double expected, double actual) => Assert.Equal(expected, actual, (Math.Abs(expected) * 1e-7) + 1e-9);

    private static double Double(JsonNode node) => double.Parse((string)node!, CultureInfo.InvariantCulture);

    private static HashSet<string> Words(JsonNode node) => [.. node.AsArray().Select(word => (string)word!)];

    private static string Text(string name) => File.ReadAllText(Path.Join(Folder, $"flocks-0.10.0.{name}"));

    private static StepCatalog Catalog() => StepCatalog.BuiltIn().WithNetworks();

    private static IRowSource Served() => CsvRowSource.FromText(Text("served.csv"));

    // The series fitted here, behind the file 0.10.0 wrote of it, from the rows 0.10.0 was given.
    private static PreparedData Prepared() =>
        new Pipeline(PipelineDeclaration.FromJson(Text("series.pipeline.json"), Catalog()), rows: null, SourceFolder.Of(Folder)).Run();
}
