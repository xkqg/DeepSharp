// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The files DeepSharp 0.4.0 — the packages on nuget.org — wrote of the wiki's Titanic network, trained eight epochs from
/// seed 20260929: its one file, and a checkpoint taken at the end of the fourth epoch. This DeepSharp serves them as 0.4.0
/// served them and goes on from the checkpoint as 0.4.0 went on, however another writer spaced or escaped them and whatever
/// version of the pipeline file their pipeline names; beside another fit of that pipeline they are refused, as they were.
/// </summary>
public class FilesWrittenBy040Tests
{
    // What 0.4.0 recorded and printed: the one file holds the network of the eighth epoch, the best one; the checkpoint the
    // network of the fourth. The logits come from arithmetic done the same way on every machine; a chance goes through the
    // machine's own exponential, so it is held to within a float's width of what 0.4.0 printed.
    private const string TrainedBehind = "944569445bd62b04644c2305382fc2c9c94404812390a174d81f338979fe8542";
    private const float Logit = -1.7700741f;
    private const double Chance = 0.14553309977054596;
    private const float CheckpointLogit = -1.8047297f;
    private const double CheckpointChance = 0.1412762999534607;

    private static readonly double[] Losses =
        [0.5907333820914763, 0.48346101448394324, 0.46330666690347283, 0.45439478831536123, 0.4527060731073444, 0.4458784142428378, 0.44006233737709816, 0.4316884417021064];

    private static readonly double[] ValidationLosses =
        [0.5142430058099273, 0.4472855800076535, 0.46015897550080953, 0.434416523553375, 0.425612455472014, 0.420608661676708, 0.4204860921192886, 0.419584324037222];

    private static readonly string Network = Fixture("titanic-0.4.0.network.json");
    private static readonly string Checkpoint = Fixture("titanic-0.4.0.checkpoint.json");
    private static readonly JsonSerializerOptions Indented = new() { WriteIndented = true };

    [Fact]
    public void EachFile_CarriesThePipelineItsNetworkWasTrainedBehind_AsTheVeryTextItsNetworkRecorded()
    {
        foreach (var file in (string[])[Network, Checkpoint])
        {
            Assert.Equal(TrainedBehind, JsonNode.Parse(file)!["network"]!["trainedOn"]!["trainedBehind"]!.GetValue<string>());
            Assert.Equal(TrainedBehind, Digest(PipelineAsCarried(file)));
        }
    }

    [Fact]
    public void TheOneFile_ServesThePassengerAs040Served_AndTheCheckpointAsItStoodAtTheEndOfItsEpoch()
    {
        var trained = TrainedNetwork.FromJson(Network, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());
        var checkpoint = TrainedNetwork.FromJson(Checkpoint, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());

        Assert.Equal(7, trained.TrainedOn.Epoch);
        Assert.Equal(20260929, trained.TrainedOn.Seed);
        Assert.Equal(TrainedBehind, trained.TrainedOn.TrainedBehind);
        Assert.Equal(Logit, LogitOf(trained));
        Assert.Equal(Chance, ChanceOf(trained), 1e-7);
        Assert.Equal(3, checkpoint.TrainedOn.Epoch);
        Assert.Equal(CheckpointLogit, LogitOf(checkpoint));
        Assert.Equal(CheckpointChance, ChanceOf(checkpoint), 1e-7);
    }

    [Fact]
    public void TheCheckpoint_GoesOnBehindItsPipelineFittedAgainHere_As040WentOn()
    {
        var prepared = FittedIn(DataFolder());
        var resumed = CheckpointFile.Read(Checkpoint, NetworkCatalog.BuiltIn(), prepared);

        GoneOnAs040WentOn(resumed.Compiled.Fit(prepared, Options(resumed.Checkpoint)));
    }

    [Fact]
    public void TheCheckpoint_RecordsNeitherItsBatchesNorItsEarlyStopping_SoItGoesOnUnderWhateverItIsHanded_AndItsRunRecordsWhatItWasHanded()
    {
        // 0.4.0 wrote down neither, so nothing says what the run went under: it goes on under what it is handed, as 0.4.0 let
        // it, and the checkpoints of the run that goes on record what that run was handed, and hold every later resume to it.
        var prepared = FittedIn(DataFolder());
        var resumed = CheckpointFile.Read(Checkpoint, NetworkCatalog.BuiltIn(), prepared);
        var files = new List<string>();

        var goneOn = resumed.Compiled.Fit(prepared, new FitOptions(seed: 20260929)
        {
            Epochs = 6,
            BatchSize = 16,
            EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true },
            ResumeFrom = resumed.Checkpoint,
            Checkpoints = new Checkpoints(checkpoint => files.Add(CheckpointFile.Write(resumed.Compiled, prepared, checkpoint))),
        });
        var training = JsonNode.Parse(files[^1])!["training"]!;
        var again = CheckpointFile.Read(files[^1], NetworkCatalog.BuiltIn(), prepared);

        Assert.Equal(6, goneOn.History!.Epochs.Count);
        Assert.Equal(2, files.Count);
        Assert.Equal(16, (int)training["batchSize"]!);
        Assert.Equal("""{"patience":10,"minDelta":0,"restoreBest":true}""", training["earlyStopping"]!.ToJsonString());
        Assert.Contains(
            "in batches of 16, and going on in batches of 32",
            Assert.Throws<ArgumentException>(() => again.Compiled.Fit(prepared, Options(again.Checkpoint))).Message,
            StringComparison.Ordinal);
    }

    [Fact]
    public void TheCheckpoint_NamesNoEngine_SoItGoesOnOnAnyEngine_AndItsRunNamesTheEngineItWentOnOn()
    {
        // 0.4.0 recorded no engine: its checkpoint goes on on whatever engine it is handed, and the checkpoints of the run that
        // goes on name that engine, and hold every later resume to it.
        var prepared = FittedIn(DataFolder());
        var resumed = CheckpointFile.Read(Checkpoint, NetworkCatalog.BuiltIn(), prepared);
        var native = new NativeMemoryBackend();
        var files = new List<string>();

        var goneOn = resumed.Compiled.Fit(prepared, new FitOptions(seed: 20260929)
        {
            Epochs = 6,
            Backend = native,
            EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true },
            ResumeFrom = resumed.Checkpoint,
            Checkpoints = new Checkpoints(checkpoint => files.Add(CheckpointFile.Write(resumed.Compiled, prepared, checkpoint))),
        });
        var again = CheckpointFile.Read(files[^1], NetworkCatalog.BuiltIn(), prepared);

        Assert.Null(JsonNode.Parse(Checkpoint)!["training"]!["engine"]);
        Assert.Equal(Enumerable.Range(0, 6), goneOn.History!.Epochs.Select(epoch => epoch.Number));
        Assert.Equal("nativememory", (string)JsonNode.Parse(files[^1])!["training"]!["engine"]!["name"]!);
        Assert.Contains(
            "a run on the engine 'nativememory'",
            Assert.Throws<ArgumentException>(() => again.Compiled.Fit(prepared, new FitOptions(seed: 20260929)
            {
                Epochs = 8, EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true }, ResumeFrom = again.Checkpoint,
            })).Message,
            StringComparison.Ordinal);
    }

    [Fact]
    public void TheCheckpoint_ReadAndWrittenAgain_StillSaysNothingOfItsBatchesOrItsEarlyStopping_AndGoesOnAs040WentOn()
    {
        var prepared = FittedIn(DataFolder());
        var read = CheckpointFile.Read(Checkpoint, NetworkCatalog.BuiltIn(), prepared);

        var again = CheckpointFile.Write(read.Compiled, prepared, read.Checkpoint);
        var training = JsonNode.Parse(again)!["training"]!.AsObject();
        var resumed = CheckpointFile.Read(again, NetworkCatalog.BuiltIn(), prepared);

        Assert.Equal(1, (int)training["version"]!);
        Assert.False(training.ContainsKey("batchSize"));
        Assert.False(training.ContainsKey("earlyStopping"));
        Assert.False(training.ContainsKey("engine"));
        GoneOnAs040WentOn(resumed.Compiled.Fit(prepared, Options(resumed.Checkpoint)));
    }

    [Fact]
    public void EachFile_RecordsNotWhichFeaturesHeldOneValue_SoWhatItServesIsSaidToBeNeitherFamiliarNorNot()
    {
        // 0.4.0 did not record which features held one value on every training row, so a passenger written 'Male' — whose
        // sex_other no training row held — is answered as 0.4.0 answered him, and nothing is said of him either way: not an
        // empty list, which would say that nothing about him is unfamiliar.
        var male = new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "Male", "22", "1", "0", "7.25"]]);

        foreach (var file in (string[])[Network, Checkpoint])
        {
            var trained = TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());
            var predictions = trained.Predict(male);

            Assert.Null(trained.TrainedOn.Unvaried);
            Assert.Null(predictions.Unfamiliar);
            Assert.Single(predictions.Answers);
            Assert.Null(TrainedNetwork.FromJson(trained.ToJson(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn()).TrainedOn.Unvaried);
        }
    }

    [Fact]
    public void TheCheckpoint_GoneOnFromHere_RecordsWhichFeaturesHeldOneValue_AsEveryRunHereDoes()
    {
        var prepared = FittedIn(DataFolder());
        var resumed = CheckpointFile.Read(Checkpoint, NetworkCatalog.BuiltIn(), prepared);

        var goneOn = resumed.Compiled.Fit(prepared, Options(resumed.Checkpoint));

        Assert.Equal(["pclass_other", "pclass_was_missing", "sex_other", "sex_was_missing"], goneOn.TrainedOn.Unvaried!.Keys);
        Assert.Equal<int?>([0, 0, 0], goneOn.Measures!.Parts.Select(part => part.UnfamiliarRows));
    }

    [Theory]
    [InlineData("compact")]
    [InlineData("tabs")]
    [InlineData("carriage returns")]
    [InlineData("escaped")]
    public void EachFile_SpacedOrEscapedAgainByAnotherWriter_IsReadAsItWasWritten(string how)
    {
        var trained = TrainedNetwork.FromJson(WrittenAgain(Network, how), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());
        var resumed = CheckpointFile.Read(WrittenAgain(Checkpoint, how), NetworkCatalog.BuiltIn(), FittedIn(DataFolder()));

        Assert.Equal(TrainedBehind, trained.TrainedOn.TrainedBehind);
        Assert.Equal(Logit, LogitOf(trained));
        Assert.Equal(4, resumed.Checkpoint.Epochs);
    }

    [Fact]
    public void EachFile_BesideAnotherFitOfItsPipeline_IsRefused_WhereItIsReadAndWhereItGoesOn()
    {
        // The same declaration fitted again once the file has grown by forty passengers: every name as it was, and the
        // numbers the fit learned not.
        var grown = FittedIn(GrownFolder());
        var swapped = JsonNode.Parse(Network)!;
        swapped["pipeline"] = JsonNode.Parse(grown.ToJson());
        var text = swapped.ToJsonString(Indented);

        var refused = Assert.Single(Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(text, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())).Faults);

        Assert.Equal(At(text, "\"pipeline\""), new Place(refused.Line, refused.Column));
        Assert.Contains("another fit", refused.Message, StringComparison.Ordinal);
        Assert.Contains("another fit", Assert.Throws<ArgumentException>(() => CheckpointFile.Read(Checkpoint, NetworkCatalog.BuiltIn(), grown)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheCheckpoint_BesideAPipelineAnsweringSomethingElse_IsRefused_NamingWhatEachAnswers()
    {
        var parch = new Pipeline(
            Pdd.Create()
                .ReadCsv("titanic.csv")
                .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("pclass", "sex").Optional("age", ColumnKind.Number).Number("fare"))
                .SplitStratified("survived", train: 0.70, validation: 0.15)
                .FillMissing("age", With.Median)
                .EncodeCategories()
                .Normalise("age", Scale.MidRange)
                .Normalise("fare", Scale.MidRange)
                .Normalise("sibsp", Scale.MidRange)
                .Target("parch")
                .Build()
                .Declaration,
            rows: null,
            SourceFolder.Of(DataFolder())).Run();

        var refused = Assert.Throws<ArgumentException>(() => CheckpointFile.Read(Checkpoint, NetworkCatalog.BuiltIn(), parch));

        Assert.Contains("trained to answer 'survived', and its pipeline answers 'parch'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheOneFile_WhosePipelineNamesAnotherVersionThanThisLibraryWrites_IsRead_AndServesAsItDid()
    {
        // The pipeline beside the network names another version of the pipeline file than the one this library writes: what
        // every file 0.4.0 wrote will name the day that version goes up. Shown with version 2, which reads every step of this
        // pipeline but its report, as a writer stamping 2 wrote it and its network recorded it.
        var older = Restamped(Network, JsonValue.Create(2), withReport: false);

        var trained = TrainedNetwork.FromJson(older, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());

        Assert.Equal(Logit, LogitOf(trained));
        Assert.Equal(ChanceOf(TrainedNetwork.FromJson(Network, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())), ChanceOf(trained));
    }

    [Fact]
    public void AFileReadAndWrittenAgain_CarriesItsPipelineAsItWasRead_AndIsReadAgain()
    {
        var older = TrainedNetwork.FromJson(Restamped(Network, JsonValue.Create(2), withReport: false), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());

        var again = older.ToJson();

        Assert.Equal(2, JsonNode.Parse(again)!["pipeline"]!["version"]!.GetValue<int>());
        Assert.Equal(Logit, LogitOf(TrainedNetwork.FromJson(again, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())));
        Assert.Equal(PipelineAsCarried(Network), PipelineAsCarried(TrainedNetwork.FromJson(Network, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn()).ToJson()));
    }

    [Fact]
    public void EachFile_ItsNetworksPartWrittenWithCarriageReturns_IsRead_AndWrittenAgainWithLineFeeds()
    {
        // 0.4.0 wrote the network's part with the line endings of the machine it ran on — these on Windows, a carriage
        // return before every line feed — and the pipeline's with line feeds. Both files are read as they are, and a
        // network read from one writes the whole file with line feeds, its pipeline the text the file carried.
        WrittenAgainWithLineFeeds(Network, Logit);
        WrittenAgainWithLineFeeds(Checkpoint, CheckpointLogit);

        static void WrittenAgainWithLineFeeds(string file, float logit)
        {
            Assert.Contains("\r\n", file[..file.IndexOf("\"pipeline\"", StringComparison.Ordinal)], StringComparison.Ordinal);

            var again = TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn()).ToJson();

            Assert.DoesNotContain('\r', again);
            Assert.Equal(PipelineAsCarried(file), PipelineAsCarried(again));
            Assert.Equal(logit, LogitOf(TrainedNetwork.FromJson(again, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())));
        }
    }

    [Theory]
    [InlineData(null)]
    [InlineData("3")]
    public void ACheckpointCarryingNoPipeline_IsRefused_ForNothingSaysItsNetworkWasTrainedBehindTheOneHandedOver(string? pipeline)
    {
        var file = JsonNode.Parse(Checkpoint)!.AsObject();

        if (pipeline is null)
        {
            file.Remove("pipeline");
        }
        else
        {
            file["pipeline"] = JsonNode.Parse(pipeline);
        }

        var refused = Assert.Throws<NetworkFileException>(() => CheckpointFile.Read(file.ToJsonString(Indented), NetworkCatalog.BuiltIn(), FittedIn(DataFolder())));

        Assert.Contains("carries no pipeline", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpointCarryingAnotherFitThanItsNetworkWasTrainedBehind_IsRefused_EvenBesideTheFitItWasTrainedBehind()
    {
        var swapped = JsonNode.Parse(Checkpoint)!;
        swapped["pipeline"] = JsonNode.Parse(FittedIn(GrownFolder()).ToJson());
        var text = swapped.ToJsonString(Indented);

        var refused = Assert.Single(Assert.Throws<NetworkFileException>(() => CheckpointFile.Read(text, NetworkCatalog.BuiltIn(), FittedIn(DataFolder()))).Faults);

        Assert.Equal(At(text, "\"pipeline\""), new Place(refused.Line, refused.Column));
        Assert.Contains("another fit", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("8", "by a newer DeepSharp")]
    [InlineData("2", "no DeepSharp wrote")]
    [InlineData("\"three\"", "no DeepSharp wrote")]
    [InlineData(null, "no DeepSharp wrote")]
    public void ACheckpointWhosePipelineNamesNoVersionANetworksFileIsWrittenWith_IsRefused(string? version, string says)
    {
        var text = Restamped(Checkpoint, version is null ? null : JsonNode.Parse(version), withReport: true);

        var refused = Assert.Single(Assert.Throws<NetworkFileException>(() => CheckpointFile.Read(text, NetworkCatalog.BuiltIn(), FittedIn(DataFolder()))).Faults);

        Assert.Equal(At(text, "\"pipeline\""), new Place(refused.Line, refused.Column));
        Assert.Contains(says, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EachFile_WhoseNetworkEndsInTheSigmoidItsLossApplies_IsReadServedAndGoneOnAsItWasWritten()
    {
        // 0.4.0 compiled a stack ending in the sigmoid a binary cross-entropy applies itself, and wrote it down; this DeepSharp
        // refuses to compile such a pair, and still reads a file holding one, serves it as 0.4.0 served it — through the
        // sigmoid twice — and goes on from its checkpoint.
        var network = EndingInASigmoid(Network);
        var checkpoint = EndingInASigmoid(Checkpoint);
        var prepared = FittedIn(DataFolder());

        var trained = TrainedNetwork.FromJson(network, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());
        var resumed = CheckpointFile.Read(checkpoint, NetworkCatalog.BuiltIn(), prepared);
        var goneOn = resumed.Compiled.Fit(prepared, Options(resumed.Checkpoint));

        Assert.IsType<Sigmoid>(Assert.IsType<LayerStack>(trained.Network).Layers[^1]);
        Assert.Equal(1 / (1 + Math.Exp(-1 / (1 + Math.Exp(-Logit)))), ChanceOf(trained), 1e-7);
        Assert.IsType<Sigmoid>(Assert.IsType<LayerStack>(resumed.Compiled.Network).Layers[^1]);
        Assert.Equal(Enumerable.Range(0, 8), goneOn.History!.Epochs.Select(epoch => epoch.Number));
    }

    // The file with a sigmoid after its network's last layer, the dense layer of one output.
    private static string EndingInASigmoid(string file)
    {
        var ending = Regex.Replace(file, "(\"outputs\": 1\\s*})", "$1, {\"kind\": \"sigmoid\"}");

        Assert.Single(Regex.Matches(ending, "\"sigmoid\""));

        return ending;
    }

    // The wiki's Titanic pipeline, reading its source from where the pipeline stands, as 0.4.0 ran it.
    private static PreparedData FittedIn(string folder) => WikiTitanic.In(folder).Run();

    private static FitOptions Options(Checkpoint from) =>
        new(seed: 20260929) { Epochs = 8, EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true }, ResumeFrom = from };

    // The run gone on from the checkpoint, as 0.4.0 went on: the four epochs the checkpoint carried read back to the last bit,
    // and the four it goes on to worked out again — through the machine's own exponential and powers, so held to within a
    // float's width of what 0.4.0 printed — ending on the network of the eighth, the best one.
    private static void GoneOnAs040WentOn(TrainedNetwork goneOn)
    {
        var epochs = goneOn.History!.Epochs;

        Assert.Equal(Enumerable.Range(0, 8), epochs.Select(epoch => epoch.Number));
        Assert.All(epochs, epoch => Assert.Equal(0.01, epoch.LearningRate));

        for (var at = 0; at < epochs.Count; at++)
        {
            var within = at < 4 ? 0 : 1e-7;

            Assert.Equal(Losses[at], epochs[at].Loss, within);
            Assert.Equal(ValidationLosses[at], epochs[at].ValidationLoss!.Value, within);
        }

        Assert.Equal(7, goneOn.TrainedOn.Epoch);
        Assert.Equal(Logit, LogitOf(goneOn));
        Assert.Equal(Chance, ChanceOf(goneOn), 1e-7);
    }

    private static string DataFolder() => WikiTitanic.DataFolder;

    // A folder holding the passenger list grown by its own first forty passengers, under the name the pipeline reads.
    private static string GrownFolder()
    {
        var folder = Directory.CreateTempSubdirectory("deepsharp-040-").FullName;
        var path = Path.Join(folder, "titanic.csv");

        File.Copy(Repository.Data("titanic.csv"), path);
        File.AppendAllLines(path, File.ReadLines(Repository.Data("titanic.csv")).Skip(1).Take(40));

        return folder;
    }

    private static InMemoryRowSource Passenger() =>
        new(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22", "1", "0", "7.25"]]);

    // What the network answers the passenger before its loss's activation: dense layers and a relu, summed in double
    // precision, the same on every machine.
    private static float LogitOf(TrainedNetwork trained)
    {
        var features = trained.Prepared.Served(Passenger(), Needs.OneScale).Features[0];

        return trained.Network.Forward(Tensor.From(new Shape(1, features.Length), [.. features.Select(value => (float)value)]), Pass.Evaluation(new CpuBackend())).Values[0];
    }

    private static double ChanceOf(TrainedNetwork trained) => Assert.Single(Assert.Single(trained.Predict(Passenger()).Answers));

    private static string Fixture(string name) => File.ReadAllText(Path.Join(AppContext.BaseDirectory, "Learners", "Fixtures", name));

    // The file as another writer would have written it: without a space between its tokens, indented by tabs, ending every
    // line with a carriage return and a line feed, or escaping a letter a writer needs not escape.
    private static string WrittenAgain(string file, string how) => how switch
    {
        "compact" => JsonNode.Parse(file)!.ToJsonString(),
        "tabs" => string.Join('\n', file.Split('\n').Select(line => new string('\t', (line.Length - line.TrimStart(' ').Length) / 2) + line.TrimStart(' '))),
        "carriage returns" => file.ReplaceLineEndings("\r\n"),
        _ => Once(file, "\"titanic.csv\"").Replace("\"titanic.csv\"", "\"\\u0074itanic.csv\"", StringComparison.Ordinal),
    };

    // The file with its pipeline as a writer stamping another version — or none — would have written it, and the digest its
    // network would then have recorded; without its report when asked, which no version before the third reads.
    private static string Restamped(string file, JsonNode? version, bool withReport)
    {
        var node = JsonNode.Parse(file)!;
        var pipeline = node["pipeline"]!.AsObject();

        if (!withReport)
        {
            var steps = pipeline["declaration"]!.AsArray();
            steps.Remove(steps.Single(step => (string?)step!["step"] == "evidence.report"));
        }

        if (version is null)
        {
            pipeline.Remove("version");
        }
        else
        {
            pipeline["version"] = version;
        }

        // A pipeline's own file: indented by two spaces, a line feed between lines.
        var text = pipeline.ToJsonString(Indented).ReplaceLineEndings("\n");
        node["pipeline"] = JsonNode.Parse(text);
        node["network"]!["trainedOn"]!["trainedBehind"] = Digest(Encoding.UTF8.GetBytes(text));

        return node.ToJsonString(Indented);
    }

    // The bytes of the value under "pipeline", exactly as they stand in the file.
    private static byte[] PipelineAsCarried(string file)
    {
        var bytes = Encoding.UTF8.GetBytes(file);
        var reader = new Utf8JsonReader(bytes);

        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.PropertyName && reader.CurrentDepth == 1 && reader.ValueTextEquals("pipeline"))
            {
                reader.Read();
                var start = (int)reader.TokenStartIndex;
                reader.Skip();

                return bytes[start..(int)reader.BytesConsumed];
            }
        }

        throw new InvalidOperationException("The file carries no pipeline.");
    }

    private static string Digest(byte[] text) => Convert.ToHexString(SHA256.HashData(text)).ToLowerInvariant();

    private static string Once(string text, string piece)
    {
        Assert.Contains(piece, text, StringComparison.Ordinal);
        Assert.Equal(text.IndexOf(piece, StringComparison.Ordinal), text.LastIndexOf(piece, StringComparison.Ordinal));

        return text;
    }

    // Where a piece of the text stands, as an editor counts: line and column, both from one.
    private static Place At(string text, string piece)
    {
        var at = Once(text, piece).IndexOf(piece, StringComparison.Ordinal);
        var start = text.LastIndexOf('\n', Math.Max(at - 1, 0)) + 1;

        return new Place(text[..at].Count(letter => letter == '\n') + 1, at - start + 1);
    }

    private readonly record struct Place(int Line, int Column);
}
