// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A network file that cannot be read as the network it names is refused — never read as something else, and never a
/// crash — with every fault at once, each at the line and column of the file it stands in: text that stops being JSON, a
/// part of no version this library reads, a kind standing where another role does, a setting of the wrong kind, missing,
/// or read by no kind, a slot that is missing, extra, of another shape or holding what is not a finite number, and a
/// checkpoint whose seed, memory, judgement or history is not what a run could have left.
/// </summary>
public class NetworkFileRefusalTests
{
    // A network of one dense layer, one input to one output, and the checkpoint of a run two epochs into it, written by
    // hand: the file a person might open. The first epoch was the best, and the second did not improve on it.
    private static readonly string Run = """
        {
          "network": {
            "version": 1,
            "layers": {"kind": "stack", "layers": [{"kind": "dense", "inputs": 1, "outputs": 1}]},
            "parameters": {
              "0.weight": {"shape": [1, 1], "values": [0.5]},
              "0.bias": {"shape": [1], "values": [0]}
            },
            "state": {},
            "loss": {"kind": "meanSquaredError"}
          },
          "training": {
            "version": 1,
            "seed": 5,
            "optimizer": {"kind": "sgd", "rate": 0.1, "momentum": 0.9},
            "schedule": {"kind": "constant"},
            "memory": {
              "0.weight": {"steps": 0, "tensors": {"momentum_buffer": {"shape": [1, 1], "values": [0.25]}}},
              "0.bias": {"steps": 0, "tensors": {"momentum_buffer": {"shape": [1], "values": [0.125]}}}
            },
            "judgement": {
              "wait": 1,
              "best": 0.25,
              "bestEpoch": 0,
              "stops": false,
              "bestSlots": {
                "parameters": {"0.weight": {"shape": [1, 1], "values": [0.375]}, "0.bias": {"shape": [1], "values": [0.0625]}},
                "state": {}
              }
            },
            "history": [
              {"number": 0, "loss": 1, "validationLoss": 0.25, "learningRate": 0.1},
              {"number": 1, "loss": 0.5, "validationLoss": 0.75, "learningRate": 0.1}
            ]
          }
        }
        """.ReplaceLineEndings("\n");

    // The early stopping of the run above, as the second version records it: a patience of five, restoring the best epoch.
    private const string RecordedStopping = "\"earlyStopping\": {\"patience\": 5, \"minDelta\": 0, \"restoreBest\": true}";

    // The same checkpoint as the second version writes it: the training part records the batch size and the early stopping
    // the run went under, which a part of the first version — 0.4.0's — does not.
    private static readonly string Recorded = Run.Replace(
        "\"version\": 1,\n    \"seed\": 5,",
        "\"version\": 2,\n    \"seed\": 5,\n    \"batchSize\": 32,\n    " + RecordedStopping + ",",
        StringComparison.Ordinal);

    // The same checkpoint naming the engine its run was on, the light one: its name, its version and its device.
    private static readonly string Engined = Recorded.Replace(
        RecordedStopping + ",",
        RecordedStopping + ",\n    \"engine\": {\"name\": \"cpu\", \"version\": \"" + new CpuBackend().Version + "\", \"device\": \"cpu\"},",
        StringComparison.Ordinal);

    private const string EngineFault =
        "The engine the run was on is written under 'engine' as its 'name' and, where the engine names them, its 'version' and its 'device', each as text.";

    private const string BatchSizeFault = "The batch size is the whole number of rows a batch of the run held, from one, under 'batchSize'.";

    private const string EarlyStoppingFault =
        "The early stopping the run went under is written under 'earlyStopping' as its 'patience' — a whole number from nought — its 'minDelta' — a number from nought — and its 'restoreBest', true or false; or as null, for a run that had none.";

    private const string JudgementFault =
        "How far early stopping had got is written as its 'wait', 'best', 'bestEpoch' — an epoch of the history — and 'stops'.";

    private const string EpochFault =
        "An epoch is written as its 'number' — its place in the history — its 'loss', its 'learningRate' and, with validation rows, its 'validationLoss'.";

    [Fact]
    public void ACheckpointWrittenByHand_IsRead_AndTheRunGoesOnFromIt()
    {
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(Run, "training", catalog, NetworkDocument.ReadNetwork(Run, "network", catalog));

        Assert.Equal(0.9, Assert.IsType<Sgd>(resumed.Compiled.Optimizer).Momentum);
        Assert.IsType<ConstantRate>(resumed.Compiled.Schedule);
        Assert.Equal(5, resumed.Checkpoint.Seed);
        Assert.Equal([0.25, 0.75], resumed.Checkpoint.History.Select(epoch => epoch.ValidationLoss));

        // Answers that contradict each other never bring the validation loss below the quarter of the first epoch, so the
        // run ends holding the best epoch's slots, as the file wrote them.
        var contradicting = new TrainingData(Tensor.From(new Shape(2, 1), [1f, 1f]), Tensor.From(new Shape(2, 1), [1f, -1f]));
        var history = resumed.Compiled.Fit(Rows(4), contradicting, new FitOptions(seed: 5)
        {
            Epochs = 3, EarlyStopping = new EarlyStopping { Patience = 5, RestoreBest = true }, ResumeFrom = resumed.Checkpoint,
        });

        Assert.Equal([0, 1, 2], history.Epochs.Select(epoch => epoch.Number));
        Assert.Equal(0, history.BestEpoch);
        Assert.Equal([[0.375f], [0.0625f]], resumed.Compiled.Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
    }

    [Fact]
    public void ACheckpointReadOnce_IsTheNetworkAndTheRunItsTwoPartsAreEachReadAs_AndGoesOnAlike()
    {
        var catalog = NetworkCatalog.BuiltIn();
        var trainedOn = Broken(Engined, "\"state\": {},\n", "\"state\": {},\n    \"trainedOn\": " + TrainedOnWritten + "\"seed\": 5, \"epoch\": 1},\n");
        var once = NetworkDocument.ReadCheckpoint(trainedOn, "network", "training", catalog);
        var network = NetworkDocument.ReadNetwork(trainedOn, "network", catalog);
        var twice = NetworkDocument.ReadTraining(trainedOn, "training", catalog, network);

        Assert.Equal(network.TrainedOn!.Features, once.Network.TrainedOn!.Features);
        Assert.Same(once.Network.Network, once.Run.Compiled.Network);
        Assert.Same(once.Network.Loss, once.Run.Compiled.Loss);
        Assert.Equal(Numbers(network.Network), Numbers(once.Network.Network));
        Assert.Equal(twice.Checkpoint.Seed, once.Run.Checkpoint.Seed);
        Assert.Equal(twice.Checkpoint.History, once.Run.Checkpoint.History);
        var paced = twice.Checkpoint.State.Pace!.Value;
        var pacedOnce = once.Run.Checkpoint.State.Pace!.Value;

        Assert.Equal(paced.BatchSize, pacedOnce.BatchSize);
        Assert.Equal(paced.EarlyStopping!.Patience, pacedOnce.EarlyStopping!.Patience);
        Assert.Equal(paced.EarlyStopping.MinDelta, pacedOnce.EarlyStopping.MinDelta);
        Assert.Equal(paced.EarlyStopping.RestoreBest, pacedOnce.EarlyStopping.RestoreBest);
        Assert.Equal(twice.Checkpoint.State.Engine, once.Run.Checkpoint.State.Engine);

        // Gone on alike: to the best epoch's slots the file holds, as the checkpoint read by its two doors goes on.
        var contradicting = new TrainingData(Tensor.From(new Shape(2, 1), [1f, 1f]), Tensor.From(new Shape(2, 1), [1f, -1f]));
        var options = new FitOptions(seed: 5) { Epochs = 3, EarlyStopping = new EarlyStopping { Patience = 5, RestoreBest = true }, ResumeFrom = once.Run.Checkpoint };

        Assert.Equal([0, 1, 2], once.Run.Compiled.Fit(Rows(4), contradicting, options).Epochs.Select(epoch => epoch.Number));
        Assert.Equal([[0.375f], [0.0625f]], once.Run.Compiled.Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
    }

    [Theory]
    [InlineData("\"values\": [0.5]", "\"values\": [\"x\"]")]
    [InlineData("\"values\": [0.5]", "\"values\": [0.5, 1]")]
    [InlineData("\"kind\": \"dense\"", "\"kind\": \"dens\"")]
    [InlineData("\"seed\": 5", "\"seed\": \"5\"")]
    [InlineData("\"values\": [0.25]", "\"values\": [[0.25]]")]
    [InlineData("\"values\": [0.375]", "\"values\": [{\"a\": 1}]")]
    [InlineData("\"learningRate\": 0.1}\n    ]", "\"learningRate\": \"fast\"}\n    ]")]
    public void ACheckpointReadOnce_IsRefusedAsItsTwoPartsAreEachRefused_InTheSameWords_AtTheSamePlaces(string find, string replace)
    {
        // The network's part is refused first, as its own door refuses it, and a run is read only beside a network.
        var broken = Broken(Recorded, find, replace);
        var catalog = NetworkCatalog.BuiltIn();
        var once = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadCheckpoint(broken, "network", "training", catalog));
        var apart = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadTraining(broken, "training", catalog, NetworkDocument.ReadNetwork(broken, "network", catalog)));

        Assert.Equal(apart.Faults, once.Faults);
    }

    [Fact]
    public void AValueOfASlotWrittenAsAListOrAnObject_IsRefusedAtTheValueItself()
    {
        var listed = Broken(Run, "\"values\": [0.5]", "\"values\": [[0.5]]");
        var named = Broken(Run, "\"values\": [0.5]", "\"values\": [{\"value\": 0.5}]");

        Assert.Equal([Place(listed, "[0.5]]", "'0.weight' holds finite numbers, and this is not one.")], NetworkRefused(listed).Faults);
        Assert.Equal([Place(named, "{\"value\"", "'0.weight' holds finite numbers, and this is not one.")], NetworkRefused(named).Faults);
    }

    [Fact]
    public void ACheckpointReadOnce_NamesEachPartItReads()
    {
        var catalog = NetworkCatalog.BuiltIn();

        Assert.Throws<ArgumentNullException>(() => NetworkDocument.ReadCheckpoint(null!, "network", "training", catalog));
        Assert.Throws<ArgumentNullException>(() => NetworkDocument.ReadCheckpoint(Run, null!, "training", catalog));
        Assert.Throws<ArgumentNullException>(() => NetworkDocument.ReadCheckpoint(Run, "network", null!, catalog));
        Assert.Throws<ArgumentNullException>(() => NetworkDocument.ReadCheckpoint(Run, "network", "training", null!));
        Assert.Equal(
            [new NetworkFileFault(1, 1, "This file holds no object under 'run'.")],
            Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadCheckpoint(Run, "network", "run", catalog)).Faults);
    }

    [Fact]
    public void ACheckpointOfARunThatWasToStop_GoesOnToNoFurtherEpoch()
    {
        var stopped = Broken(Run, "\"stops\": false", "\"stops\": true");
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(stopped, "training", catalog, NetworkDocument.ReadNetwork(stopped, "network", catalog));

        var history = resumed.Compiled.Fit(Rows(4), Rows(2), new FitOptions(seed: 5)
        {
            Epochs = 5, EarlyStopping = new EarlyStopping { Patience = 1 }, ResumeFrom = resumed.Checkpoint,
        });

        Assert.Equal(2, history.Epochs.Count);
        Assert.Equal(Stopping.NoLongerImproving, history.Stopped);
    }

    [Fact]
    public void ACheckpointOfTheSecondVersion_GoesOnUnderTheBatchesAndTheEarlyStoppingItRecords_AndIsRefusedUnderOthers()
    {
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(Recorded, "training", catalog, NetworkDocument.ReadNetwork(Recorded, "network", catalog));
        var contradicting = new TrainingData(Tensor.From(new Shape(2, 1), [1f, 1f]), Tensor.From(new Shape(2, 1), [1f, -1f]));

        var wrong = Assert.Throws<ArgumentException>(() => resumed.Compiled.Fit(Rows(4), contradicting, new FitOptions(seed: 5)
        {
            Epochs = 3, BatchSize = 16, EarlyStopping = new EarlyStopping { Patience = 5, RestoreBest = true }, ResumeFrom = resumed.Checkpoint,
        }));
        var history = resumed.Compiled.Fit(Rows(4), contradicting, new FitOptions(seed: 5)
        {
            Epochs = 3, EarlyStopping = new EarlyStopping { Patience = 5, RestoreBest = true }, ResumeFrom = resumed.Checkpoint,
        });

        Assert.Equal(
            "The checkpoint was taken of a run in batches of 32, and going on in batches of 16 would take other rows into every step. (Parameter 'options')",
            wrong.Message);
        Assert.Equal([0, 1, 2], history.Epochs.Select(epoch => epoch.Number));
        Assert.Equal([[0.375f], [0.0625f]], resumed.Compiled.Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
    }

    [Fact]
    public void ACheckpointOfARunThatHadNoEarlyStopping_SaysSoWithNull_AndGoesOnOnlyWithoutIt()
    {
        var none = Broken(Recorded, RecordedStopping, "\"earlyStopping\": null");
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(none, "training", catalog, NetworkDocument.ReadNetwork(none, "network", catalog));

        var wrong = Assert.Throws<ArgumentException>(
            () => resumed.Compiled.Fit(Rows(4), Rows(2), new FitOptions(seed: 5) { Epochs = 3, EarlyStopping = new EarlyStopping(), ResumeFrom = resumed.Checkpoint }));
        var history = resumed.Compiled.Fit(Rows(4), Rows(2), new FitOptions(seed: 5) { Epochs = 3, ResumeFrom = resumed.Checkpoint });

        Assert.Contains("taken of a run with no early stopping", wrong.Message, StringComparison.Ordinal);
        Assert.Equal(3, history.Epochs.Count);
    }

    [Fact]
    public void ACheckpointOfTheFirstVersion_RecordsNeitherItsBatchesNorItsEarlyStopping_AndGoesOnUnderWhateverItIsHanded()
    {
        // As 0.4.0 went on from it: only its seed is held to.
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(Run, "training", catalog, NetworkDocument.ReadNetwork(Run, "network", catalog));

        var history = resumed.Compiled.Fit(Rows(4), Rows(2), new FitOptions(seed: 5) { Epochs = 3, BatchSize = 1, ResumeFrom = resumed.Checkpoint });

        Assert.Equal([0, 1, 2], history.Epochs.Select(epoch => epoch.Number));
        Assert.Contains("seeded 5", Assert.Throws<ArgumentException>(
            () => resumed.Compiled.Fit(Rows(4), Rows(2), new FitOptions(seed: 6) { Epochs = 3, ResumeFrom = resumed.Checkpoint })).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpointOfTheFirstVersion_NeverJudged_GoneOnAtItsLastEpochRestoringTheBest_TrainsNothing_AndRestoresNothingItNeverJudged()
    {
        var file = JsonNode.Parse(Run)!;
        file["training"]!.AsObject().Remove("judgement");
        var unjudged = file.ToJsonString();
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(unjudged, "training", catalog, NetworkDocument.ReadNetwork(unjudged, "network", catalog));

        var history = resumed.Compiled.Fit(Rows(4), Rows(2), new FitOptions(seed: 5)
        {
            Epochs = 2, EarlyStopping = new EarlyStopping { RestoreBest = true }, ResumeFrom = resumed.Checkpoint,
        });

        Assert.Equal(2, history.Epochs.Count);
        Assert.Null(history.BestEpoch);
        Assert.Equal([[0.5f], [0f]], resumed.Compiled.Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
    }

    [Theory]
    [InlineData("\"batchSize\": 32,\n    ", "", "\"training\"", BatchSizeFault)]
    [InlineData("\"batchSize\": 32", "\"batchSize\": \"32\"", "\"batchSize\"", BatchSizeFault)]
    [InlineData("\"batchSize\": 32", "\"batchSize\": 0", "\"batchSize\"", BatchSizeFault)]
    [InlineData("\"batchSize\": 32", "\"batchSize\": 1.5", "\"batchSize\"", BatchSizeFault)]
    [InlineData(RecordedStopping + ",\n    ", "", "\"training\"", EarlyStoppingFault)]
    [InlineData(RecordedStopping, "\"earlyStopping\": 3", "\"earlyStopping\"", EarlyStoppingFault)]
    [InlineData(RecordedStopping, "\"earlyStopping\": false", "\"earlyStopping\"", EarlyStoppingFault)]
    [InlineData("\"patience\": 5, ", "", "\"earlyStopping\"", EarlyStoppingFault)]
    [InlineData("\"patience\": 5", "\"patience\": -1", "\"earlyStopping\"", EarlyStoppingFault)]
    [InlineData("\"patience\": 5", "\"patience\": 2.5", "\"earlyStopping\"", EarlyStoppingFault)]
    [InlineData("\"minDelta\": 0, ", "", "\"earlyStopping\"", EarlyStoppingFault)]
    [InlineData("\"minDelta\": 0", "\"minDelta\": \"x\"", "\"earlyStopping\"", EarlyStoppingFault)]
    [InlineData("\"minDelta\": 0", "\"minDelta\": -0.5", "\"earlyStopping\"", EarlyStoppingFault)]
    [InlineData(", \"restoreBest\": true", "", "\"earlyStopping\"", EarlyStoppingFault)]
    [InlineData("\"restoreBest\": true", "\"restoreBest\": 1", "\"earlyStopping\"", EarlyStoppingFault)]
    [InlineData("\"restoreBest\": true}", "\"restoreBest\": true, \"monitor\": \"val_loss\"}", "\"monitor\"",
        "'monitor' is not written here: this holds 'patience', 'minDelta', 'restoreBest'.")]
    public void WhatTheRunWentUnder_MissingOrWrittenAsAnythingElse_IsRefusedWhereItStands(string find, string replace, string at, string message)
    {
        var broken = Broken(Recorded, find, replace);

        Assert.Equal([Place(broken, at, message)], TrainingRefused(broken).Faults);
    }

    [Fact]
    public void APartOfTheSecondVersion_HoldsBothTheBatchSizeAndTheEarlyStopping_AndOneOfTheFirst_BothOrNeither()
    {
        var neither = Broken(Broken(Recorded, "\"batchSize\": 32,\n    ", string.Empty), RecordedStopping + ",\n    ", string.Empty);
        var batchesOnly = Broken(Run, "\"seed\": 5,", "\"seed\": 5, \"batchSize\": 32,");
        var stoppingOnly = Broken(Run, "\"seed\": 5,", "\"seed\": 5, " + RecordedStopping + ",");

        Assert.Equal([Place(neither, "\"training\"", BatchSizeFault), Place(neither, "\"training\"", EarlyStoppingFault)], TrainingRefused(neither).Faults);
        Assert.Equal([Place(batchesOnly, "\"training\"", EarlyStoppingFault)], TrainingRefused(batchesOnly).Faults);
        Assert.Equal([Place(stoppingOnly, "\"training\"", BatchSizeFault)], TrainingRefused(stoppingOnly).Faults);

        // One of the first version that holds both says what its run went under, and a run going on from it is held to it.
        var both = Broken(Run, "\"seed\": 5,", "\"seed\": 5, \"batchSize\": 32, " + RecordedStopping + ",");
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(both, "training", catalog, NetworkDocument.ReadNetwork(both, "network", catalog));

        Assert.Contains("in batches of 32, and going on in batches of 16", Assert.Throws<ArgumentException>(() => resumed.Compiled.Fit(Rows(4), Rows(2), new FitOptions(seed: 5)
        {
            Epochs = 3, BatchSize = 16, EarlyStopping = new EarlyStopping { Patience = 5, RestoreBest = true }, ResumeFrom = resumed.Checkpoint,
        })).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpointThatNamesItsEngine_GoesOnOnIt_AndIsRefusedOnAnother_NamingBoth()
    {
        var catalog = NetworkCatalog.BuiltIn();
        var refused = NetworkDocument.ReadTraining(Engined, "training", catalog, NetworkDocument.ReadNetwork(Engined, "network", catalog));
        var resumed = NetworkDocument.ReadTraining(Engined, "training", catalog, NetworkDocument.ReadNetwork(Engined, "network", catalog));
        var native = new NativeMemoryBackend();

        var wrong = Assert.Throws<ArgumentException>(() => refused.Compiled.Fit(Rows(4), Rows(2), new FitOptions(seed: 5)
        {
            Epochs = 3, Backend = native, EarlyStopping = new EarlyStopping { Patience = 5, RestoreBest = true }, ResumeFrom = refused.Checkpoint,
        }));
        var history = resumed.Compiled.Fit(Rows(4), Rows(2), new FitOptions(seed: 5)
        {
            Epochs = 3, EarlyStopping = new EarlyStopping { Patience = 5, RestoreBest = true }, ResumeFrom = resumed.Checkpoint,
        });

        Assert.Equal(
            $"The checkpoint was taken of a run on the engine 'cpu' {new CpuBackend().Version} on cpu, and going on under the engine 'nativememory' {native.Version} on cpu would round every step otherwise. (Parameter 'options')",
            wrong.Message);
        Assert.Equal([0, 1, 2], history.Epochs.Select(epoch => epoch.Number));
    }

    [Theory]
    [InlineData("0.4.0", "cpu")]
    [InlineData(null, "cuda:0")]
    public void ACheckpointOfTheLightEngine_InAnotherVersionOrOnAnotherDevice_IsRefusedByTheLightEngineOfThisOne(string? version, string device)
    {
        // Another release of the light engine, or the same one somewhere else, is another engine to the checkpoint.
        var light = new CpuBackend();
        var taken = Broken(Engined, $"\"version\": \"{light.Version}\", \"device\": \"cpu\"", $"\"version\": \"{version ?? light.Version}\", \"device\": \"{device}\"");
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(taken, "training", catalog, NetworkDocument.ReadNetwork(taken, "network", catalog));

        var wrong = Assert.Throws<ArgumentException>(() => resumed.Compiled.Fit(Rows(4), Rows(2), new FitOptions(seed: 5)
        {
            Epochs = 3, EarlyStopping = new EarlyStopping { Patience = 5, RestoreBest = true }, ResumeFrom = resumed.Checkpoint,
        }));

        Assert.Equal(
            $"The checkpoint was taken of a run on the engine 'cpu' {version ?? light.Version} on {device}, and going on under the engine 'cpu' {light.Version} on cpu would round every step otherwise. (Parameter 'options')",
            wrong.Message);
    }

    [Fact]
    public void ACheckpointThatNamesNoEngine_OfTheFirstVersionOrOfTheSecond_GoesOnOnAnyEngine()
    {
        // 0.4.0 recorded no engine, and a part of the second version written before the engine was recorded names none
        // either: neither says which engine its run was on, so a run goes on from it on whatever engine it is handed.
        var catalog = NetworkCatalog.BuiltIn();

        foreach (var file in (string[])[Run, Recorded])
        {
            var resumed = NetworkDocument.ReadTraining(file, "training", catalog, NetworkDocument.ReadNetwork(file, "network", catalog));
            var history = resumed.Compiled.Fit(Rows(4), Rows(2), new FitOptions(seed: 5)
            {
                Epochs = 3, Backend = new NativeMemoryBackend(), EarlyStopping = new EarlyStopping { Patience = 5, RestoreBest = true }, ResumeFrom = resumed.Checkpoint,
            });

            Assert.Equal([0, 1, 2], history.Epochs.Select(epoch => epoch.Number));
        }
    }

    [Fact]
    public void AnEngineThatNamesOnlyItself_IsRecordedByItsName_AndHeldToIt()
    {
        var named = Broken(Engined, $", \"version\": \"{new CpuBackend().Version}\", \"device\": \"cpu\"", string.Empty);
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(named, "training", catalog, NetworkDocument.ReadNetwork(named, "network", catalog));

        // Named 'cpu' alone, the light engine — which names its version and its device too — is another engine to it.
        var wrong = Assert.Throws<ArgumentException>(() => resumed.Compiled.Fit(Rows(4), Rows(2), new FitOptions(seed: 5)
        {
            Epochs = 3, EarlyStopping = new EarlyStopping { Patience = 5, RestoreBest = true }, ResumeFrom = resumed.Checkpoint,
        }));

        Assert.StartsWith("The checkpoint was taken of a run on the engine 'cpu', and going on under the engine 'cpu' ", wrong.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("{\"name\": \"cpu\", ", "null, \"unused\": {", "\"engine\"", EngineFault)]
    [InlineData("{\"name\": \"cpu\", ", "\"cpu\", \"unused\": {", "\"engine\"", EngineFault)]
    [InlineData("\"name\": \"cpu\", ", "", "\"engine\"", EngineFault)]
    [InlineData("\"name\": \"cpu\"", "\"name\": 3", "\"engine\"", EngineFault)]
    [InlineData("\"device\": \"cpu\"", "\"device\": null", "\"engine\"", EngineFault)]
    [InlineData("\"device\": \"cpu\"", "\"device\": \"cpu\", \"threads\": 16", "\"threads\"", "'threads' is not written here: this holds 'name', 'version', 'device'.")]
    public void TheEngineARunWasOn_WrittenAsAnythingElse_IsRefusedWhereItStands(string find, string replace, string at, string message)
    {
        var broken = Broken(Engined, find, replace);

        Assert.Contains(Place(broken, at, message), TrainingRefused(broken).Faults);
    }

    [Fact]
    public void TheEnginesVersion_WrittenAsANumber_IsRefusedWhereTheEngineStands()
    {
        var broken = Broken(Engined, $"\"version\": \"{new CpuBackend().Version}\"", "\"version\": 2");

        Assert.Equal([Place(broken, "\"engine\"", EngineFault)], TrainingRefused(broken).Faults);
    }

    [Fact]
    public void TextThatStopsBeingJson_IsRefusedWhereItStops()
    {
        var broken = Broken(Run, "\"state\": {},\n", "\"state\": {},,\n");

        var fault = Assert.Single(NetworkRefused(broken).Faults);

        Assert.Equal(Place(broken, ",,").Line, fault.Line);
        Assert.StartsWith("The text stops being JSON here: ", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AKeyWrittenTwice_IsRefusedAtItsSecondPlace()
    {
        var broken = Broken(Run, "\"inputs\": 1,", "\"inputs\": 1, \"inputs\":2,");

        Assert.Equal(
            [Place(broken, "\"inputs\":2", "'inputs' is written twice here, and only one of the two would be read.")],
            NetworkRefused(broken).Faults);
    }

    [Theory]
    [InlineData("""{"pipeline": {}}""")]
    [InlineData("""[]""")]
    [InlineData("""{"network": 3}""")]
    [InlineData("""3""")]
    public void AFileWithNoObjectUnderTheKey_IsRefused(string text)
    {
        var fault = Assert.Single(NetworkRefused(text).Faults);

        Assert.Equal("This file holds no object under 'network'.", fault.Message);
        Assert.Equal(text.StartsWith("{\"network\"", StringComparison.Ordinal) ? 2 : 1, fault.Column);
    }

    [Theory]
    [InlineData("\"version\": 1,\n    \"layers\"", "\"layers\"", "\"network\"")]
    [InlineData("\"version\": 1,\n    \"layers\"", "\"version\": \"one\",\n    \"layers\"", "\"version\": \"one\"")]
    [InlineData("\"version\": 1,\n    \"layers\"", "\"version\": 0,\n    \"layers\"", "\"version\": 0")]
    [InlineData("\"version\": 1,\n    \"layers\"", "\"version\": 1.5,\n    \"layers\"", "\"version\": 1.5")]
    public void APartThatNamesNoVersionItCouldBe_IsRefusedWhole(string find, string replace, string at)
    {
        var broken = Broken(Run, find, replace);

        Assert.Equal(
            [Place(broken, at, "The 'network' part names the whole number of the version it was written against, from 1, under 'version'.")],
            NetworkRefused(broken).Faults);
    }

    [Fact]
    public void AKeyThePartDoesNotHold_IsRefusedAtTheKey()
    {
        var broken = Broken(Run, "\"state\": {},\n", "\"state\": {}, \"weights\": {},\n");

        Assert.Equal(
            [Place(broken, "\"weights\"", "The 'network' part has no 'weights'. It holds: version, layers, parameters, state, loss, trainedOn.")],
            NetworkRefused(broken).Faults);
    }

    [Fact]
    public void ANetworkPartWithoutItsLoss_IsRefused()
    {
        var broken = Broken(Run, ",\n    \"loss\": {\"kind\": \"meanSquaredError\"}", string.Empty);

        Assert.Equal([Place(broken, "\"network\"", "The loss is written under 'loss'.")], NetworkRefused(broken).Faults);
    }

    [Fact]
    public void ALayerWhereTheNetworkStands_IsRefused_AndSoIsTheSettingItNeverReads()
    {
        var broken = Broken(Run, "\"kind\": \"stack\"", "\"kind\": \"relu\"");

        Assert.Equal(
            [
                Place(broken, "\"layers\": {", "What is written under 'layers' is no network: a stack, or a network written as code."),
                Place(broken, "\"layers\": [", "'layers' is not a setting of 'relu'."),
            ],
            NetworkRefused(broken).Faults);
    }

    [Fact]
    public void AKindOfAnotherRole_IsRefusedWhereItStands()
    {
        var broken = Broken(Run, "\"loss\": {\"kind\": \"meanSquaredError\"}", "\"loss\": {\"kind\": \"relu\"}");

        Assert.Equal([Place(broken, "\"loss\": {\"kind\": \"relu\"}", "'relu' is a layer, and a loss stands here.")], NetworkRefused(broken).Faults);
    }

    [Theory]
    [InlineData("3", "3]")]
    [InlineData("{\"kind\": \"relu\"}, {\"kind\": \"relu\"}, 3", "3]")]
    [InlineData("{\"inputs\": 1, \"outputs\": 1}", "{\"inputs\"")]
    public void ALayerThatNamesNoKind_IsRefusedAtItsPlaceInTheList(string layer, string at)
    {
        var broken = Broken(Run, "{\"kind\": \"dense\", \"inputs\": 1, \"outputs\": 1}", layer);

        Assert.Equal([Place(broken, at, "Where a layer stands, an object names its kind under 'kind'.")], NetworkRefused(broken).Faults);
    }

    [Theory]
    [InlineData("\"inputs\": 1,", "\"inputs\": \"one\",", "\"inputs\"", "'inputs' is a whole number here.")]
    [InlineData("\"inputs\": 1,", "\"inputs\": 1.5,", "\"inputs\"", "'inputs' is a whole number here.")]
    [InlineData(", \"outputs\": 1}", "}", "{\"kind\": \"dense\"", "'outputs' is missing here: it is a whole number.")]
    [InlineData("\"outputs\": 1}", "\"outputs\": 1, \"units\": 3}", "\"units\"", "'units' is not a setting of 'dense'.")]
    public void ASettingMissing_OfTheWrongKind_OrReadByNoKind_IsRefusedWhereItIs(string find, string replace, string at, string message)
    {
        var broken = Broken(Run, find, replace);

        Assert.Equal([Place(broken, at, message)], NetworkRefused(broken).Faults);
    }

    [Theory]
    [InlineData("""{"kind": "dropout", "rate": 1.5}""", "{\"kind\": \"dropout\"", "A dropout leaves out a share of the values, from nothing to below one.")]
    [InlineData("""{"kind": "dropout", "rate": "x"}""", "\"rate\"", "'rate' is a number here.")]
    [InlineData("""{"kind": "reshape", "each": [2, "x"]}""", "\"each\"", "'each' is a list of whole numbers here.")]
    [InlineData("""{"kind": "reshape", "each": 3}""", "\"each\"", "'each' is a list of whole numbers here.")]
    [InlineData("""{"kind": "reshape"}""", "{\"kind\": \"reshape\"}", "'each' is missing here: it is a list of whole numbers.")]
    [InlineData("""{"kind": "stack", "layers": 3}""", "\"layers\": 3", "'layers' is a list of layers here.")]
    public void ASettingItsKindRefuses_IsRefusedInTheFilesOwnWords(string layer, string at, string message)
    {
        var text = Stack(layer);

        Assert.Equal([Place(text, at, message)], NetworkRefused(text).Faults);
    }

    [Theory]
    [InlineData("\"padding\": \"valid\"", "\"padding\"", "'padding' is a whole number, or 'same', here.")]
    [InlineData("\"padding\": 1, \"dilation\": 2", "\"dilation\"", "'dilation' is not a setting of 'conv2d'.")]
    [InlineData("\"padding\": 1, \"groups\": 2", "\"groups\"", "'groups' is not a setting of 'conv2d'.")]
    [InlineData("\"padding\": 1, \"strides\": [1, 2]", "\"strides\"", "'strides' is not a setting of 'conv2d'.")]
    public void AConvolutionWrittenAsAWindowDoesNotWalk_IsRefusedAtTheSettingItCannotRead(string settings, string at, string message)
    {
        // A window pads by a border stated for every side or as 'same', and walks one stride down and across, undilated, over
        // every channel at once: what else another framework's convolution can say has no setting here to be read into.
        var text = Stack("""{"kind": "conv2d", "inChannels": 2, "outChannels": 2, "height": 3, "width": 3, "stride": 1, """ + settings + "}");

        Assert.Contains(Place(text, at, message), NetworkRefused(text).Faults);
    }

    [Fact]
    public void APoolingLayer_IsNoKindThisLibraryKnows_AndIsRefusedWhereItStands()
    {
        var text = Stack("""{"kind": "maxPooling2d", "height": 2, "width": 2}""");

        var fault = Assert.Single(NetworkRefused(text).Faults);

        Assert.Equal(Place(text, "{\"kind\": \"maxPooling2d\"", fault.Message), fault);
        Assert.StartsWith("'maxPooling2d' is not a kind this catalog knows.", fault.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("3", "\"trainedOn\"", TrainedOnFault)]
    [InlineData(TrainedOnWritten + "\"seed\": \"5\", \"epoch\": 0}", "\"trainedOn\"", TrainedOnFault)]
    [InlineData(TrainedOnWritten + "\"seed\": 5, \"epoch\": -1}", "\"trainedOn\"", TrainedOnFault)]
    [InlineData("{\"features\": [\"x\", 3], \"answers\": [\"y\"], \"output\": \"target\", \"trainedBehind\": \"ab\", \"seed\": 5, \"epoch\": 0}", "\"trainedOn\"", TrainedOnFault)]
    [InlineData("{\"features\": [\"x\"], \"answers\": [\"y\"], \"output\": 7, \"trainedBehind\": \"ab\", \"seed\": 5, \"epoch\": 0}", "\"trainedOn\"", TrainedOnFault)]
    [InlineData(TrainedOnWritten + "\"seed\": 5, \"epoch\": 0, \"rows\": 3}", "\"rows\"",
        "'rows' is not written here: this holds 'features', 'answers', 'output', 'trainedBehind', 'seed', 'epoch', 'unvaried'.")]
    [InlineData(TrainedOnWritten + "\"seed\": 5, \"epoch\": 0, \"unvaried\": [\"x\"]}", "\"unvaried\"",
        "Which features held one value on every training row is written as an object under 'unvaried': each feature by its name, with that value.")]
    [InlineData(TrainedOnWritten + "\"seed\": 5, \"epoch\": 0, \"unvaried\": {\"z\": 0}}", "\"z\"", "'z' is no feature this network was trained on.")]
    [InlineData(TrainedOnWritten + "\"seed\": 5, \"epoch\": 0, \"unvaried\": {\"x\": \"0\"}}", "\"x\": \"0\"",
        "'x' is written with the one value it held on every training row, a finite number, and this is not one.")]
    [InlineData(TrainedOnWritten + "\"seed\": 5, \"epoch\": 0, \"unvaried\": {\"x\": null}}", "\"x\": null",
        "'x' is written with the one value it held on every training row, a finite number, and this is not one.")]
    public void WhatANetworkWasTrainedOn_WrittenAsAnythingElse_IsRefusedWhereItStands(string written, string at, string message)
    {
        var broken = Broken(Run, "\"state\": {},\n", $"\"state\": {{}},\n    \"trainedOn\": {written},\n");

        Assert.Equal([Place(broken, at, message)], NetworkRefused(broken).Faults);
    }

    [Fact]
    public void WhichFeaturesHeldOneValue_WrittenAsTheFeaturesTheNetworkWasTrainedOn_IsRead()
    {
        var written = Broken(Run, "\"state\": {},\n", "\"state\": {},\n    \"trainedOn\": " + TrainedOnWritten + "\"seed\": 5, \"epoch\": 0, \"unvaried\": {\"x\": -0.25}},\n");

        var read = NetworkDocument.ReadNetwork(written, "network", NetworkCatalog.BuiltIn());

        Assert.Equal([new KeyValuePair<string, double>("x", -0.25)], read.TrainedOn!.Unvaried!);
    }

    private const string TrainedOnWritten = "{\"features\": [\"x\"], \"answers\": [\"y\"], \"output\": \"target\", \"trainedBehind\": \"ab\", ";

    private const string TrainedOnFault =
        "What the network was trained on is written as its 'features' and 'answers' — lists of names — its 'output' and 'trainedBehind', its 'seed' and its 'epoch'.";

    [Fact]
    public void ASlotThatIsNotAShapeAndItsValues_IsRefusedAtTheSlot()
    {
        var broken = Broken(Run, "\"0.bias\": {\"shape\": [1], \"values\": [0]}", "\"0.bias\": [0]");

        Assert.Equal([Place(broken, "\"0.bias\": [0]", "'0.bias' is written as its 'shape' and its 'values'.")], NetworkRefused(broken).Faults);
    }

    [Fact]
    public void AShapeThatIsNotWholeNumbers_IsRefusedAtTheShape()
    {
        var broken = Broken(Run, "\"shape\": [1, 1], \"values\": [0.5]", "\"shape\": [1, \"x\"], \"values\": [0.5]");

        Assert.Equal([Place(broken, "\"shape\": [1, \"x\"]", "'0.weight' is a 1x1 slot here, and is written as 1x?.")], NetworkRefused(broken).Faults);
    }

    [Fact]
    public void AsManyValuesAsTheShapeHolds_AreWritten()
    {
        var broken = Broken(Run, "\"values\": [0.5]", "\"values\": [0.5, 1]");

        Assert.Equal(
            [Place(broken, "\"values\": [0.5, 1]", "'0.weight' holds as many values as its 1x1 shape, 1, and the file holds 2.")],
            NetworkRefused(broken).Faults);
    }

    [Theory]
    [InlineData("\"x\"")]
    [InlineData("1e39")]
    [InlineData("null")]
    public void AValueThatIsNoFiniteNumber_IsRefusedAtTheValueItself(string value)
    {
        var broken = Broken(Run, "\"values\": [0.5]", $"\"values\": [{value}]");

        Assert.Equal([Place(broken, value + "]", "'0.weight' holds finite numbers, and this is not one.")], NetworkRefused(broken).Faults);
    }

    [Fact]
    public void ASlotUnderTheOtherKindsKey_IsRefused_AndSoIsItsAbsenceWhereItBelongs()
    {
        var text = """
            {"network": {"version": 1,
              "layers": {"kind": "stack", "layers": [{"kind": "batchNorm", "features": 1, "momentum": 0.1, "epsilon": 1e-05}]},
              "parameters": {"0.bias": {"shape": [1], "values": [0]}, "0.running_mean": {"shape": [1], "values": [0]}},
              "state": {"0.weight": {"shape": [1], "values": [1]}, "0.running_var": {"shape": [1], "values": [1]}},
              "loss": {"kind": "meanSquaredError"}}}
            """.ReplaceLineEndings("\n");

        Assert.Equal(
            [
                Place(text, "\"parameters\"", "'0.weight' is missing: every slot of the network is written."),
                Place(text, "\"0.running_mean\"", "'0.running_mean' is a running statistic, and is written under 'state'."),
                Place(text, "\"state\"", "'0.running_mean' is missing: every slot of the network is written."),
                Place(text, "\"0.weight\"", "'0.weight' is a parameter, and is written under 'parameters'."),
            ],
            NetworkRefused(text).Faults);
    }

    [Theory]
    [InlineData("\"seed\": 5", "\"seed\": \"5\"", "\"seed\"", "The seed is the whole number the run was worked out from, under 'seed'.")]
    [InlineData("\"seed\": 5,\n    ", "", "\"training\"", "The seed is the whole number the run was worked out from, under 'seed'.")]
    [InlineData("\"0.bias\": {\"steps\"", "\"0.biass\": {\"steps\"", "\"0.biass\"", "'0.biass' is no parameter of this network.")]
    [InlineData("\"0.weight\": {\"steps\": 0,", "\"0.weight\": {\"steps\": \"0\",", "\"0.weight\": {\"steps\"",
        "What is remembered of '0.weight' is written as its 'steps' and its 'tensors'.")]
    [InlineData("\"momentum_buffer\": {\"shape\": [1, 1]", "\"velocity\": {\"shape\": [1, 1]", "\"0.weight\": {\"steps\"",
        "What the checkpoint remembers of 'weight' has no momentum_buffer of its 1x1 shape, which this optimizer keeps.")]
    [InlineData("\"values\": [0.25]", "\"values\": [0.25, 1]", "\"values\": [0.25, 1]",
        "'0.weight momentum_buffer' holds as many values as its 1x1 shape, 1, and the file holds 2.")]
    [InlineData("\"0.bias\": {\"steps\": 0, \"tensors\": {\"momentum_buffer\": {\"shape\": [1], \"values\": [0.125]}}}", "\"0.bias\": 3", "\"0.bias\": 3",
        "What is remembered of '0.bias' is written as its 'steps' and its 'tensors'.")]
    [InlineData("\"tensors\": {\"momentum_buffer\": {\"shape\": [1], \"values\": [0.125]}}", "\"tensors\": 3", "\"0.bias\": {\"steps\"",
        "What is remembered of '0.bias' is written as its 'steps' and its 'tensors'.")]
    [InlineData("\"wait\": 1", "\"wait\": -1", "\"judgement\"", JudgementFault)]
    [InlineData("\"wait\": 1,\n      \"best\"", "\"best\"", "\"judgement\"", JudgementFault)]
    [InlineData("\"bestEpoch\": 0", "\"bestEpoch\": -1", "\"judgement\"", JudgementFault)]
    [InlineData("\"best\": 0.25", "\"best\": \"x\"", "\"judgement\"", JudgementFault)]
    [InlineData("\"bestEpoch\": 0", "\"bestEpoch\": 2", "\"judgement\"", JudgementFault)]
    [InlineData("\"stops\": false", "\"stops\": 0", "\"judgement\"", JudgementFault)]
    [InlineData(", \"0.bias\": {\"shape\": [1], \"values\": [0.0625]}", "", "\"parameters\": {\"0.weight\": {\"shape\": [1, 1], \"values\": [0.375]}}",
        "'0.bias' is missing: every slot of the network is written.")]
    [InlineData("{\"number\": 1,", "{\"number\": 3,", "{\"number\": 3", EpochFault)]
    [InlineData("{\"number\": 1, \"loss\": 0.5, ", "{\"number\": 1, ", "{\"number\": 1", EpochFault)]
    [InlineData("\"validationLoss\": 0.75, \"learningRate\": 0.1", "\"validationLoss\": 0.75, \"learningRate\": \"fast\"", "{\"number\": 1", EpochFault)]
    [InlineData("\"validationLoss\": 0.75", "\"validationLoss\": null", "{\"number\": 1", EpochFault)]
    [InlineData("{\"kind\": \"sgd\", \"rate\": 0.1, \"momentum\": 0.9}", "3", "\"optimizer\"", "Where an optimizer stands, an object names its kind under 'kind'.")]
    [InlineData("{\"kind\": \"sgd\", \"rate\": 0.1, \"momentum\": 0.9}", "{\"kind\": \"constant\"}", "\"optimizer\"",
        "'constant' is a learning-rate schedule, and an optimizer stands here.")]
    [InlineData("{\"kind\": \"constant\"}", "{\"kind\": \"sgd\", \"rate\": 0.1, \"momentum\": 0}", "\"schedule\"",
        "'sgd' is an optimizer, and a learning-rate schedule stands here.")]
    [InlineData("{\"kind\": \"sgd\", \"rate\": 0.1, \"momentum\": 0.9}", "{\"kind\": \"adam\", \"rate\": 0.1, \"betas\": [0.9], \"epsilon\": 1e-08}", "\"optimizer\"",
        "'betas' is the two rates Adam's running means forget at, as a list of two numbers.")]
    [InlineData("{\"kind\": \"sgd\", \"rate\": 0.1, \"momentum\": 0.9}", "{\"kind\": \"adam\", \"rate\": 0.1, \"betas\": [0.9, \"x\"], \"epsilon\": 1e-08}", "\"betas\"",
        "'betas' is a list of numbers here.")]
    public void WhatNoRunCouldHaveLeft_IsRefusedWhereItStands(string find, string replace, string at, string message)
    {
        var broken = Broken(Run, find, replace);

        Assert.Equal([Place(broken, at, message)], TrainingRefused(broken).Faults);
    }

    [Fact]
    public void AParameterNameHoldingAForgedLineBreak_IsShownEscaped_NeverBreakingTheMessageIntoASecondLine()
    {
        var broken = Broken(Run, "\"0.bias\": {\"steps\"", "\"0.bias\\r\\nFAKE LINE\": {\"steps\"");

        Assert.Equal(
            [Place(broken, "\"0.bias\\r\\nFAKE LINE\"", "'0.bias\\r\\nFAKE LINE' is no parameter of this network.")],
            TrainingRefused(broken).Faults);
    }

    [Fact]
    public void AParameterNameOfExtremeLength_IsCutShortInTheRefusal_SayingHowLongItWas()
    {
        var huge = new string('m', 10_000);
        var broken = Broken(Run, "\"0.bias\": {\"steps\"", "\"" + huge + "\": {\"steps\"");

        var fault = Assert.Single(TrainingRefused(broken).Faults);

        Assert.True(fault.Message.Length < 1000, $"The message is {fault.Message.Length} characters long.");
        Assert.Contains("(10000 characters)", fault.Message, StringComparison.Ordinal);
        Assert.EndsWith("is no parameter of this network.", fault.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("history", "histories", "The epochs so far are written as a list under 'history'.")]
    [InlineData("memory", "memories", "What the optimizer remembers is written by path, as an object under 'memory'.")]
    [InlineData("optimizer", "optimiser", "The optimizer is written under 'optimizer'.")]
    [InlineData("schedule", "timetable", "The learning-rate schedule is written under 'schedule'.")]
    public void APartOfTheTrainingMissing_IsRefused_AndSoIsTheKeyWrittenInItsPlace(string key, string written, string message)
    {
        var broken = Broken(Run, $"\"{key}\": ", $"\"{written}\": ");

        Assert.Equal(
            [
                Place(broken, "\"training\"", message),
                Place(broken, $"\"{written}\"", $"The 'training' part has no '{written}'. It holds: version, seed, batchSize, earlyStopping, engine, optimizer, schedule, memory, judgement, history."),
            ],
            TrainingRefused(broken).Faults);
    }

    [Theory]
    [InlineData("\"stops\": false,", "\"stops\": false, \"patience\": 3,", "\"patience\"",
        "'patience' is not written here: this holds 'wait', 'best', 'bestEpoch', 'stops', 'bestSlots'.")]
    [InlineData("{\"number\": 1, \"loss\": 0.5,", "{\"number\": 1, \"loss\": 0.5, \"rows\": 16,", "\"rows\"",
        "'rows' is not written here: this holds 'number', 'loss', 'validationLoss', 'learningRate'.")]
    [InlineData("\"0.bias\": {\"steps\": 0,", "\"0.bias\": {\"steps\": 0, \"step\": 1,", "\"step\"", "'step' is not written here: this holds 'steps', 'tensors'.")]
    [InlineData("\"shape\": [1], \"values\": [0.125]", "\"shape\": [1], \"values\": [0.125], \"dtype\": \"float\"", "\"dtype\"",
        "'dtype' is not written here: this holds 'shape', 'values'.")]
    [InlineData("\"state\": {}\n", "\"state\": {}, \"moments\": {}\n", "\"moments\"", "'moments' is not written here: this holds 'parameters', 'state'.")]
    public void AKeyNoPartOfACheckpointHolds_IsRefused_RatherThanReadAsNothing(string find, string replace, string at, string message)
    {
        var broken = Broken(Run, find, replace);

        Assert.Equal([Place(broken, at, message)], TrainingRefused(broken).Faults);
    }

    [Fact]
    public void BestSlotsThatAreNoObject_AreRefused_EachKindOfSlotAtOnce()
    {
        var broken = Broken(Run, "\"bestSlots\": {", "\"bestSlots\": 3, \"unused\": {");

        Assert.Equal(
            [
                Place(broken, "\"bestSlots\"", "The parameters are written by their paths, as an object under 'parameters'."),
                Place(broken, "\"bestSlots\"", "The running statistics are written by their paths, as an object under 'state'."),
                Place(broken, "\"unused\"", "'unused' is not written here: this holds 'wait', 'best', 'bestEpoch', 'stops', 'bestSlots'."),
            ],
            TrainingRefused(broken).Faults);
    }

    [Fact]
    public void TheTrainingPart_IsReadBesideTheNetworkReadFromTheSameFile()
    {
        Assert.Throws<ArgumentNullException>(() => NetworkDocument.ReadTraining(Run, "training", NetworkCatalog.BuiltIn(), default));
        Assert.Throws<ArgumentNullException>(() => NetworkDocument.ReadNetwork(Run, null!, NetworkCatalog.BuiltIn()));
        Assert.Throws<ArgumentNullException>(() => NetworkDocument.ReadNetwork(Run, "network", null!));
        Assert.Throws<ArgumentNullException>(() => NetworkDocument.ReadNetwork(null!, "network", NetworkCatalog.BuiltIn()));
    }

    [Fact]
    public void ACheckpointOfANetworkThatHasMovedOnSince_CannotBeWrittenBesideIt()
    {
        var network = new LayerStack(new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias()));
        var compiled = network.Compile(new Sgd(0.1), new MeanSquaredError());
        var kept = new List<Checkpoint>();

        compiled.Fit(Rows(8, width: 4), validation: null, new FitOptions(seed: 1) { Epochs = 2, Checkpoints = new Checkpoints(kept.Add) });

        var wrong = Assert.Throws<InvalidOperationException>(() => Written(writer => NetworkDocument.WriteTraining(writer, compiled, kept[0])));

        Assert.Contains("moved on", wrong.Message, StringComparison.Ordinal);
        Written(writer => NetworkDocument.WriteTraining(writer, compiled, kept[1]));

        // Nor beside a network of other slots altogether.
        var other = new LayerStack(new Relu(), new Dense(4, 1, new RandomStream(2).Draw("initialise:test", 0, 0)));

        Assert.Throws<InvalidOperationException>(
            () => Written(writer => NetworkDocument.WriteTraining(writer, other.Compile(new Sgd(0.1), new MeanSquaredError()), kept[1])));
    }

    [Fact]
    public void AKindThatIsNoneOfTheFour_CannotBeRegistered()
    {
        var wrong = Assert.Throws<ArgumentException>(() => NetworkCatalog.BuiltIn().Register<Stray>());

        Assert.Contains("'stray' is no layer, loss, optimizer or learning-rate schedule", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALayerOfNoKindAFileCanName_CannotBeWritten()
    {
        var wrong = Assert.Throws<InvalidOperationException>(
            () => Written(writer => NetworkDocument.WriteNetwork(writer, new LayerStack(new Unnamed()), new MeanSquaredError())));

        Assert.Contains("A Unnamed cannot be written", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFault_SaysWhereItIs_AndARefusalSaysEveryOne()
    {
        NetworkFileFault[] faults = [new(3, 5, "one"), new(7, 1, "two")];
        var refused = new NetworkFileException(faults);

        Assert.Equal("(3,5): one", faults[0].ToString());
        Assert.Equal($"(3,5): one{Environment.NewLine}(7,1): two", refused.Message);
        Assert.Same(faults, refused.Faults);
        Assert.Throws<ArgumentNullException>(() => new NetworkFileException(null!));
    }

    // The text with one piece of it changed, which stands in it exactly once.
    private static string Broken(string text, string find, string replace)
    {
        Assert.Equal(1, Occurrences(text, find));

        return text.Replace(find, replace, StringComparison.Ordinal);
    }

    // A fault at the one place a piece of the text stands, counted as an editor counts: line and column, both from one.
    private static NetworkFileFault Place(string text, string marker, string message = "")
    {
        Assert.Equal(1, Occurrences(text, marker));

        var at = text.IndexOf(marker, StringComparison.Ordinal);
        var start = text.LastIndexOf('\n', Math.Max(at - 1, 0)) + 1;

        return new NetworkFileFault(text[..at].Count(letter => letter == '\n') + 1, at - start + 1, message);
    }

    private static int Occurrences(string text, string piece)
    {
        var count = 0;

        for (var at = text.IndexOf(piece, StringComparison.Ordinal); at >= 0; at = text.IndexOf(piece, at + 1, StringComparison.Ordinal))
        {
            count++;
        }

        return count;
    }

    // A network part of one stack of the given layers, with no slots, and a loss.
    private static string Stack(string layers) =>
        """{"network": {"version": 1, "layers": {"kind": "stack", "layers": [""" + layers
        + """]}, "parameters": {}, "state": {}, "loss": {"kind": "meanSquaredError"}}}""";

    private static NetworkFileException NetworkRefused(string text) =>
        Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()));

    private static NetworkFileException TrainingRefused(string text)
    {
        var catalog = NetworkCatalog.BuiltIn();
        var network = NetworkDocument.ReadNetwork(text, "network", catalog);

        return Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadTraining(text, "training", catalog, network));
    }

    private static void Written(Action<Utf8JsonWriter> body)
    {
        using var stream = new MemoryStream();
        using var writer = new Utf8JsonWriter(stream);

        writer.WriteStartObject();
        writer.WritePropertyName("part");
        body(writer);
        writer.WriteEndObject();
    }

    private static float[][] Numbers(Network network) => [.. network.Slots().Select(named => named.Slot.Value.Values.ToArray())];

    private static TrainingData Rows(int count, int width = 1)
    {
        var features = Enumerable.Range(0, count * width).Select(at => MathF.Sin(at) / 2).ToArray();
        var answers = Enumerable.Range(0, count).Select(row => 2 * features[row * width]).ToArray();

        return new TrainingData(Tensor.From(new Shape(count, width), features), Tensor.From(new Shape(count, 1), answers));
    }

    // A kind that is none of the four a file names.
    private sealed class Stray : ISaved<Stray>
    {
        public static string Name => "stray";

        public static Stray Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

        public void WriteSettings(Utf8JsonWriter writer)
        {
        }
    }

    // A layer registered under no kind.
    private sealed class Unnamed : Layer
    {
        protected override Tensor Compute(Tensor input, Pass pass) => input;
    }
}
