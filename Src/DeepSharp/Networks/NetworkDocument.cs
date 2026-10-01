// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Reflection;
using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// How a network is written into a file and read back: its layers by the kinds they are registered under and the settings
/// they are rebuilt from, every slot's numbers by its path, its loss by name — and, for a checkpoint, everything a run
/// needs to go on.
/// </summary>
/// <remarks>
/// Each part is written as an object into a file somebody else holds together — a network beside its pipeline, say — and
/// read back from that file's text by the key it stands under, so every fault is named at the line and column of the file
/// itself, all at once. The numbers are written as JSON numbers in the shortest form that reads back to the same float,
/// so a network read back is the same network to the last bit, and a value that is not a finite number is not written at
/// all. Each part names its own version; a part newer than this library is refused whole.
/// </remarks>
public static class NetworkDocument
{
    /// <summary>The version of the network's part of a file — and of a checkpoint's training part — this library writes and reads up to.</summary>
    /// <remarks>
    /// It goes up when a part can say something an older library would not understand, so that library names the newer
    /// version rather than the key it does not know. In the second, the network's part says which features held one value on
    /// every training row, and a convolution whose window is padded as 'same' says so with the word; a checkpoint's training
    /// part says the batch size, the early stopping and the engine its run went under. A part of the first, which 0.4.0
    /// wrote, is read as it was written and says nothing of them, every window's padding a number.
    /// </remarks>
    public const int Version = 2;

    /// <summary>
    /// The version 0.4.0 wrote, whose training part records neither the batch size nor the early stopping of its run: the one
    /// a part may hold neither in, and the one a checkpoint that records neither is written again with.
    /// </summary>
    internal const int FirstVersion = 1;

    internal const string KindKey = "kind";
    internal const string PackageKey = "package";
    internal const string VersionKey = "version";
    internal const string LayersKey = "layers";
    internal const string ParametersKey = "parameters";
    internal const string StateKey = "state";
    internal const string LossKey = "loss";
    internal const string ShapeKey = "shape";
    internal const string ValuesKey = "values";
    internal const string SeedKey = "seed";
    internal const string OptimizerKey = "optimizer";
    internal const string ScheduleKey = "schedule";
    internal const string MemoryKey = "memory";
    internal const string StepsKey = "steps";
    internal const string TensorsKey = "tensors";
    internal const string JudgementKey = "judgement";
    internal const string HistoryKey = "history";
    internal const string WaitKey = "wait";
    internal const string BestKey = "best";
    internal const string BestEpochKey = "bestEpoch";
    internal const string StopsKey = "stops";
    internal const string BestSlotsKey = "bestSlots";
    internal const string NumberKey = "number";
    internal const string EpochLossKey = "loss";
    internal const string ValidationLossKey = "validationLoss";
    internal const string LearningRateKey = "learningRate";
    internal const string TrainedOnKey = "trainedOn";
    internal const string FeaturesKey = "features";
    internal const string AnswersKey = "answers";
    internal const string OutputKey = "output";
    internal const string TrainedBehindKey = "trainedBehind";
    internal const string EpochKey = "epoch";
    internal const string UnvariedKey = "unvaried";
    internal const string BatchSizeKey = "batchSize";
    internal const string EarlyStoppingKey = "earlyStopping";
    internal const string PatienceKey = "patience";
    internal const string MinDeltaKey = "minDelta";
    internal const string RestoreBestKey = "restoreBest";
    internal const string EngineKey = "engine";
    internal const string NameKey = "name";
    internal const string DeviceKey = "device";

    private static readonly string[] NetworkKeys = [VersionKey, LayersKey, ParametersKey, StateKey, LossKey, TrainedOnKey];

    private static readonly string[] TrainingKeys =
        [VersionKey, SeedKey, BatchSizeKey, EarlyStoppingKey, EngineKey, OptimizerKey, ScheduleKey, MemoryKey, JudgementKey, HistoryKey];
    private static readonly Assembly OwnAssembly = typeof(NetworkDocument).Assembly;

    /// <summary>Writes a network and its loss as one object, and what it was trained on when that is said.</summary>
    /// <param name="writer">The writer, where the object goes: after the key it stands under, say.</param>
    /// <param name="network">The network: a stack, or a network written as code that is registered under a kind.</param>
    /// <param name="loss">What it was trained to bring down, whose output activation a prediction goes through.</param>
    /// <param name="trainedOn">What it was trained on, written beside it for whatever serves it to check; nothing, unless said.</param>
    /// <exception cref="InvalidOperationException">
    /// A layer is of no kind that can be written, or a slot holds a value that is not a finite number; or what it was
    /// trained on says a feature held one value on every training row that is no feature of it, or a value that is not a
    /// finite number.
    /// </exception>
    public static void WriteNetwork(Utf8JsonWriter writer, Network network, Loss loss, TrainedOn? trainedOn = null)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(loss);

        var slots = network.Slots().ToArray();

        writer.WriteStartObject();
        writer.WriteNumber(VersionKey, Version);
        writer.WritePropertyName(LayersKey);
        WriteKind(writer, network);
        WriteSlots(writer, slots, slots.ToDictionary(named => named.Path, named => named.Slot.Value));
        writer.WritePropertyName(LossKey);
        WriteKind(writer, loss);

        if (trainedOn is not null)
        {
            WriteTrainedOn(writer, trainedOn);
        }

        writer.WriteEndObject();
    }

    /// <summary>Writes what a run needs to go on from a checkpoint, as one object: the network's own part is written beside it.</summary>
    /// <param name="writer">The writer, where the object goes.</param>
    /// <param name="compiled">The network being trained, with its optimizer and its schedule.</param>
    /// <param name="checkpoint">The checkpoint.</param>
    /// <exception cref="InvalidOperationException">
    /// The network has moved on since the checkpoint was taken, so the network written beside it would not be the one the
    /// checkpoint goes on from; or a value is not a finite number.
    /// </exception>
    /// <remarks>
    /// Beside the seed it records the batch size and the early stopping the run went under — its patience, its least fall
    /// that counts and whether it restores the best, or null for a run that had none — and the engine it was on, by its name
    /// and, where the engine names them, its version and its device: what a run going on from it is handed again or refused.
    /// A checkpoint that records none of them, read from a file 0.4.0 wrote, is written as that version wrote it, still
    /// saying nothing of them, rather than with values nobody recorded.
    /// </remarks>
    public static void WriteTraining(Utf8JsonWriter writer, CompiledNetwork compiled, Checkpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentNullException.ThrowIfNull(compiled);
        ArgumentNullException.ThrowIfNull(checkpoint);

        var state = checkpoint.State;
        var slots = compiled.Network.Slots().ToArray();

        if (slots.Any(named => !state.Slots.TryGetValue(named.Path, out var kept) || !ReferenceEquals(kept, named.Slot.Value)))
        {
            throw new InvalidOperationException(
                "A checkpoint is written beside its network as the network stood when the checkpoint was taken, and this network has moved on since.");
        }

        writer.WriteStartObject();
        writer.WriteNumber(VersionKey, state.Pace is null ? FirstVersion : Version);
        writer.WriteNumber(SeedKey, state.Seed);

        if (state.Pace is { } pace)
        {
            WritePace(writer, pace);
        }

        if (state.Engine is { } engine)
        {
            WriteEngine(writer, engine);
        }

        writer.WritePropertyName(OptimizerKey);
        WriteKind(writer, compiled.Optimizer);
        writer.WritePropertyName(ScheduleKey);
        WriteKind(writer, compiled.Schedule);
        writer.WriteStartObject(MemoryKey);

        foreach (var (path, memory) in state.Memory)
        {
            writer.WriteStartObject(path);
            writer.WriteNumber(StepsKey, memory.Steps);
            writer.WriteStartObject(TensorsKey);

            foreach (var (name, tensor) in memory.Tensors)
            {
                WriteTensor(writer, name, tensor, $"{path} {name}");
            }

            writer.WriteEndObject();
            writer.WriteEndObject();
        }

        writer.WriteEndObject();

        if (state.Judgement is { } judgement)
        {
            WriteJudgement(writer, judgement, slots);
        }

        writer.WriteStartArray(HistoryKey);

        foreach (var epoch in state.History)
        {
            writer.WriteStartObject();
            writer.WriteNumber(NumberKey, epoch.Number);
            writer.WriteNumber(EpochLossKey, epoch.Loss);

            if (epoch.ValidationLoss is { } validation)
            {
                writer.WriteNumber(ValidationLossKey, validation);
            }

            writer.WriteNumber(LearningRateKey, epoch.LearningRate);
            writer.WriteEndObject();
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
    }

    /// <summary>Reads a network and its loss from the object under a key at the top of a file.</summary>
    /// <param name="json">The whole file.</param>
    /// <param name="property">The key the network stands under.</param>
    /// <param name="catalog">The kinds the network may be made of.</param>
    /// <returns>The network, every slot holding the numbers written, and its loss.</returns>
    /// <exception cref="NetworkFileException">Anything in the network's part is wrong: every fault, at its line and column in the file.</exception>
    public static SavedNetwork ReadNetwork(string json, string property, NetworkCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(property);
        ArgumentNullException.ThrowIfNull(catalog);

        var text = TextOf(json);
        using var document = text.Parsed();

        return NetworkIn(ReaderOf(document.RootElement, property, text, NetworkKeys), text, catalog);
    }

    /// <summary>
    /// Reads a checkpoint from a file that holds its network and what its run needs to go on under two keys at its top, the
    /// text read once for both.
    /// </summary>
    /// <param name="json">The whole file.</param>
    /// <param name="network">The key the network stands under.</param>
    /// <param name="training">The key the training part stands under.</param>
    /// <param name="catalog">The kinds the network, the optimizer and the schedule may be.</param>
    /// <returns>
    /// The network as the checkpoint left it, with its loss and what it was trained on, and the run to go on from it: as
    /// <see cref="ReadNetwork"/> and <see cref="ReadTraining"/> read them.
    /// </returns>
    /// <exception cref="NetworkFileException">
    /// Anything in the network's part is wrong, or — the network read — anything in the training part: every fault, at its
    /// line and column in the file, in the words each part's own door refuses it with.
    /// </exception>
    /// <remarks>
    /// Reading the two parts through their own doors reads the text twice. A checkpoint holds, beside every parameter, what
    /// the optimizer remembers of it, and the best epoch's slots when its run keeps them; read once, a checkpoint of four
    /// million parameters under Adam allocates, beside its own copy of the text, the numbers it holds and little else.
    /// </remarks>
    public static SavedCheckpoint ReadCheckpoint(string json, string network, string training, NetworkCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(training);
        ArgumentNullException.ThrowIfNull(catalog);

        var text = TextOf(json);
        using var document = text.Parsed();
        var saved = NetworkIn(ReaderOf(document.RootElement, network, text, NetworkKeys), text, catalog);

        return new SavedCheckpoint(saved, TrainingIn(ReaderOf(document.RootElement, training, text, TrainingKeys), text, catalog, saved));
    }

    // The text of a file, surveyed for the network's parts: a tensor's values read where they stand.
    private static NetworkText TextOf(string json) => new(json, PartReader.ReadsApart);

    // The network and its loss a reader of a network's part reads; refused when anything in that part is wrong.
    private static SavedNetwork NetworkIn(PartReader reader, NetworkText text, NetworkCatalog catalog)
    {
        var rebuilding = new Rebuilding(catalog, text);
        var network = reader.NetworkOf(rebuilding);
        var loss = reader.Rebuilt<Loss>(LossKey, NetworkCatalog.Role.Loss, rebuilding, "loss");

        var trainedOn = reader.TrainedOnOf();

        if (network is not null)
        {
            reader.Load(network);
        }

        text.ThrowIfFaulty();

        return new SavedNetwork(network!, loss!) { TrainedOn = trainedOn };
    }

    /// <summary>Reads what a run needs to go on from a checkpoint, from the object under a key at the top of a file.</summary>
    /// <param name="json">The whole file.</param>
    /// <param name="property">The key the training part stands under.</param>
    /// <param name="catalog">The kinds the optimizer and the schedule may be.</param>
    /// <param name="network">The network read from the same file, as the checkpoint left it.</param>
    /// <returns>
    /// The network compiled with the optimizer and schedule written — the optimizer remembering what the checkpoint says it
    /// did of each parameter — and the checkpoint to go on from, holding a run that goes on from it to the seed, the batch
    /// size, the early stopping and the engine it records.
    /// </returns>
    /// <exception cref="NetworkFileException">Anything in the training part is wrong: every fault, at its line and column in the file.</exception>
    /// <remarks>
    /// A training part records the batch size and the early stopping its run went under, and one that holds only one of them,
    /// or — from the second version — neither, is refused where it stands. A part of the first version, which 0.4.0 wrote, may
    /// hold neither: its checkpoint says nothing of them, and goes on under whatever it is handed. The engine its run was on
    /// is read where the part names one; a part that names none, as none 0.4.0 wrote does, says nothing of it, and its
    /// checkpoint goes on on whatever engine it is handed.
    /// </remarks>
    public static ResumedRun ReadTraining(string json, string property, NetworkCatalog catalog, SavedNetwork network)
    {
        ArgumentNullException.ThrowIfNull(property);
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentNullException.ThrowIfNull(network.Network);

        var text = TextOf(json);
        using var document = text.Parsed();

        return TrainingIn(ReaderOf(document.RootElement, property, text, TrainingKeys), text, catalog, network);
    }

    // What a run needs to go on, as a reader of a training part reads it beside the network read from the same file;
    // refused when anything in that part is wrong.
    private static ResumedRun TrainingIn(PartReader reader, NetworkText text, NetworkCatalog catalog, SavedNetwork network)
    {
        var rebuilding = new Rebuilding(catalog, text);
        var optimizer = reader.Rebuilt<Optimizer>(OptimizerKey, NetworkCatalog.Role.Optimizer, rebuilding, "optimizer");
        var schedule = reader.Rebuilt<LearningRateSchedule>(ScheduleKey, NetworkCatalog.Role.Schedule, rebuilding, "learning-rate schedule");
        var slots = network.Network.Slots().ToDictionary(named => named.Path, named => named.Slot);
        var seed = reader.Seed();
        var pace = reader.PaceOf();
        var engine = reader.EngineOf();
        var memory = reader.MemoryOf(slots);
        var history = reader.HistoryOf();
        var judgement = reader.JudgementOf(network.Network, history?.Count);

        if (optimizer is not null)
        {
            reader.Recall(optimizer, slots, memory);
        }

        text.ThrowIfFaulty();

        // Compiled as it was written: a file keeps the loss its network was compiled with, 0.4.0's pairs included.
        var compiled = new CompiledNetwork(network.Network, optimizer!, network.Loss, schedule);
        var resumable = new Resumable(seed, history!, slots.ToDictionary(named => named.Key, named => named.Value.Value), memory)
        {
            Judgement = judgement,
            Pace = pace,
            Engine = engine,
        };

        return new ResumedRun(compiled, new Checkpoint(resumable));
    }

    /// <summary>Writes a kind as an object: its name, the package it comes from when not this one, and its settings.</summary>
    /// <exception cref="InvalidOperationException">It is of no kind a file can name.</exception>
    internal static void WriteKind(Utf8JsonWriter writer, object thing)
    {
        if (thing is not ISaved saved)
        {
            throw new InvalidOperationException(
                $"A {thing.GetType().Name} cannot be written: a kind a network file names implements ISaved<T>, and is registered with the catalog that reads it.");
        }

        writer.WriteStartObject();
        writer.WriteString(KindKey, saved.Kind);

        if (thing.GetType().Assembly != OwnAssembly)
        {
            writer.WriteString(PackageKey, thing.GetType().Assembly.GetName().Name);
        }

        saved.WriteSettings(writer);
        writer.WriteEndObject();
    }

    // Every slot, the parameters under one key and the running statistics under the other, each by its path, holding the
    // value given for it.
    private static void WriteSlots(Utf8JsonWriter writer, NamedSlot[] slots, IReadOnlyDictionary<string, Tensor> values)
    {
        foreach (var learns in (bool[])[true, false])
        {
            writer.WriteStartObject(learns ? ParametersKey : StateKey);

            foreach (var named in slots.Where(named => named.Slot is Parameter == learns))
            {
                WriteTensor(writer, named.Path, values[named.Path], named.Path);
            }

            writer.WriteEndObject();
        }
    }

    private static void WriteTensor(Utf8JsonWriter writer, string key, Tensor tensor, string saying)
    {
        foreach (var value in tensor.Values)
        {
            if (!float.IsFinite(value))
            {
                throw new InvalidOperationException(
                    string.Create(CultureInfo.InvariantCulture, $"'{saying}' holds {value}, which is not a finite number, and a file cannot hold one."));
            }
        }

        writer.WriteStartObject(key);
        writer.WriteStartArray(ShapeKey);

        foreach (var length in tensor.Shape.Axes)
        {
            writer.WriteNumberValue(length);
        }

        writer.WriteEndArray();
        writer.WriteStartArray(ValuesKey);

        foreach (var value in tensor.Values)
        {
            writer.WriteNumberValue(value);
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
    }

    private static void WriteTrainedOn(Utf8JsonWriter writer, TrainedOn trainedOn)
    {
        var unvaried = trainedOn.Unvaried;

        if (unvaried is not null)
        {
            ThrowIfUnwritable(trainedOn.Features, unvaried);
        }

        writer.WriteStartObject(TrainedOnKey);
        WriteNames(writer, FeaturesKey, trainedOn.Features);
        WriteNames(writer, AnswersKey, trainedOn.Answers);
        writer.WriteString(OutputKey, trainedOn.Output);
        writer.WriteString(TrainedBehindKey, trainedOn.TrainedBehind);
        writer.WriteNumber(SeedKey, trainedOn.Seed);
        writer.WriteNumber(EpochKey, trainedOn.Epoch);

        // Written whenever it is said — that none held one value included — in the order the network takes the features: a
        // part without it is one that does not say.
        if (unvaried is not null)
        {
            writer.WriteStartObject(UnvariedKey);

            foreach (var feature in trainedOn.Features.Where(unvaried.ContainsKey))
            {
                writer.WriteNumber(feature, unvaried[feature]);
            }

            writer.WriteEndObject();
        }

        writer.WriteEndObject();
    }

    // A feature said to have held one value is one the network was trained on, and its value a finite number: what the file's
    // own reader would refuse is never written.
    private static void ThrowIfUnwritable(IReadOnlyList<string> features, IReadOnlyDictionary<string, double> unvaried)
    {
        foreach (var (feature, value) in unvaried)
        {
            if (!features.Contains(feature))
            {
                throw new InvalidOperationException(
                    $"'{feature}' is said to have held one value on every training row, and it is no feature the network was trained on.");
            }

            if (!double.IsFinite(value))
            {
                throw new InvalidOperationException(string.Create(
                    CultureInfo.InvariantCulture,
                    $"'{feature}' is said to have held {value} on every training row, which is not a finite number, and a file cannot hold one."));
            }
        }
    }

    private static void WriteNames(Utf8JsonWriter writer, string key, IReadOnlyList<string> names)
    {
        writer.WriteStartArray(key);

        foreach (var name in names)
        {
            writer.WriteStringValue(name);
        }

        writer.WriteEndArray();
    }

    // The batch size and the early stopping the run went under: its patience, its least fall that counts and whether it
    // restores the best — or null, which says the run had none, where a part without the key says nothing of it.
    private static void WritePace(Utf8JsonWriter writer, Pace pace)
    {
        writer.WriteNumber(BatchSizeKey, pace.BatchSize);

        if (pace.EarlyStopping is not { } stopping)
        {
            writer.WriteNull(EarlyStoppingKey);

            return;
        }

        writer.WriteStartObject(EarlyStoppingKey);
        writer.WriteNumber(PatienceKey, stopping.Patience);
        writer.WriteNumber(MinDeltaKey, stopping.MinDelta);
        writer.WriteBoolean(RestoreBestKey, stopping.RestoreBest);
        writer.WriteEndObject();
    }

    // The engine the run was on, as it names itself: its name, and its version and its device where it names them.
    private static void WriteEngine(Utf8JsonWriter writer, Engine engine)
    {
        writer.WriteStartObject(EngineKey);
        writer.WriteString(NameKey, engine.Name);

        if (engine.Version is { } version)
        {
            writer.WriteString(VersionKey, version);
        }

        if (engine.Device is { } device)
        {
            writer.WriteString(DeviceKey, device);
        }

        writer.WriteEndObject();
    }

    private static void WriteJudgement(Utf8JsonWriter writer, Judgement judgement, NamedSlot[] slots)
    {
        writer.WriteStartObject(JudgementKey);
        writer.WriteNumber(WaitKey, judgement.Wait);
        writer.WriteNumber(BestKey, judgement.Best);
        writer.WriteNumber(BestEpochKey, judgement.BestEpoch);
        writer.WriteBoolean(StopsKey, judgement.Stops);

        if (judgement.BestSlots is { } best)
        {
            writer.WriteStartObject(BestSlotsKey);
            WriteSlots(writer, slots, best);
            writer.WriteEndObject();
        }

        writer.WriteEndObject();
    }

    // The reader of the part of the file under its key: an object of a version this library reads, holding only the keys it
    // knows, read as a part of the version it names.
    private static PartReader ReaderOf(JsonElement root, string property, NetworkText text, string[] keys)
    {
        if (root.Member(property) is not { ValueKind: JsonValueKind.Object } part)
        {
            text.Fault([property], $"This file holds no object under '{property}'.");

            throw text.Refused();
        }

        if (part.Member(VersionKey)?.AsWhole() is not { } number || number < 1)
        {
            text.Fault([property, VersionKey], $"The '{property}' part names the whole number of the version it was written against, from 1, under '{VersionKey}'.");

            throw text.Refused();
        }

        if (number > Version)
        {
            text.Fault([property, VersionKey], string.Create(CultureInfo.InvariantCulture,
                $"The '{property}' part was written against version {number}, by a newer DeepSharp than this one, which reads up to version {Version}. Nothing in it is read: read it with that DeepSharp."));

            throw text.Refused();
        }

        foreach (var unknown in part.EnumerateObject().Select(each => each.Name).Where(name => !keys.Contains(name)))
        {
            text.Fault([property, unknown], $"The '{property}' part has no '{unknown.Quoted()}'. It holds: {string.Join(", ", keys)}.");
        }

        return new PartReader(text, property, part, number);
    }
}

/// <summary>A network read back from a file, with the loss it was trained to bring down.</summary>
/// <param name="Network">The network, every slot holding the numbers written.</param>
/// <param name="Loss">Its loss, whose output activation a prediction goes through.</param>
public readonly record struct SavedNetwork(Network Network, Loss Loss)
{
    /// <summary>What it was trained on, when the file says; nothing otherwise.</summary>
    public TrainedOn? TrainedOn { get; init; }
}

/// <summary>A run read back from a checkpoint: the network compiled as it was, and the checkpoint to go on from.</summary>
/// <param name="Compiled">The network with its optimizer, loss and schedule.</param>
/// <param name="Checkpoint">The checkpoint, handed to <see cref="FitOptions.ResumeFrom"/>.</param>
public readonly record struct ResumedRun(CompiledNetwork Compiled, Checkpoint Checkpoint);

/// <summary>A checkpoint read back from a file: its network as the checkpoint left it, and the run to go on from it.</summary>
/// <param name="Network">The network and its loss, with what it was trained on when the file says.</param>
/// <param name="Run">The same network compiled as it was, and the checkpoint to go on from.</param>
public readonly record struct SavedCheckpoint(SavedNetwork Network, ResumedRun Run);
