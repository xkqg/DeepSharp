// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Tensors;
using static DeepSharp.Networks.NetworkDocument;

namespace DeepSharp.Networks;

/// <summary>Reads one part of a network file — the network's, or the training's — noting every fault at its place in the file.</summary>
/// <remarks>
/// Every value is read as what it should be and nothing else, every object holds only the keys it is written with, and a
/// fault is noted where it stands and the reading goes on, so a file is refused once, with every fault in it. The part is
/// read as the version it names — one this library reads — says it is written.
/// </remarks>
internal sealed class PartReader(NetworkText text, string property, JsonElement part, int version)
{
    private readonly string[] _here = [property];

    /// <summary>
    /// Whether a list of a file is a tensor's values this reader reads — a slot's, under its part's parameters or running
    /// statistics, what the optimizer remembers of a parameter, or a slot of the best epoch's — whose numbers are read from
    /// the text where they stand, rather than parsed with the rest of the part.
    /// </summary>
    /// <param name="path">The list's path from the top of the file, the key of the part it stands in first.</param>
    public static bool ReadsApart(string[] path) =>
        path is [_, ParametersKey or StateKey, _, ValuesKey]
            or [_, MemoryKey, _, TensorsKey, _, ValuesKey]
            or [_, JudgementKey, BestSlotsKey, ParametersKey or StateKey, _, ValuesKey];

    /// <summary>The kind written under a key of the part, rebuilt; nothing, with its fault noted, when it cannot be.</summary>
    public TKind? Rebuilt<TKind>(string key, NetworkCatalog.Role role, Rebuilding rebuilding, string what)
        where TKind : class
    {
        if (part.Member(key) is { } written)
        {
            return (TKind?)rebuilding.Rebuild(written, [.. _here, key], role);
        }

        text.Fault(_here, $"The {what} is written under '{key}'.");

        return null;
    }

    /// <summary>The network written under its key: a stack, or a network written as code; nothing, with its fault noted, when it is not one.</summary>
    public Network? NetworkOf(Rebuilding rebuilding)
    {
        var layer = Rebuilt<Layer>(LayersKey, NetworkCatalog.Role.Layer, rebuilding, "network");

        if (layer is null or Network)
        {
            return (Network?)layer;
        }

        text.Fault([.. _here, LayersKey], $"What is written under '{LayersKey}' is no network: a stack, or a network written as code.");

        return null;
    }

    /// <summary>
    /// Reads every slot written into the network, each by its path, into the network: every slot the network holds, and none
    /// it does not — held to the rule a file of another framework's is held to by <see cref="Network.Load"/>.
    /// </summary>
    public void Load(Network network)
    {
        var load = SlotsUnder(part, _here, network);

        if (!text.Faulty)
        {
            load.Apply();
        }
    }

    /// <summary>The seed the run was worked out from.</summary>
    public long Seed()
    {
        if (part.Member(SeedKey)?.AsLong() is { } seed)
        {
            return seed;
        }

        text.Fault([.. _here, SeedKey], $"The seed is the whole number the run was worked out from, under '{SeedKey}'.");

        return 0;
    }

    /// <summary>
    /// The batch size and the early stopping the run went under; nothing when a part of the first version, which 0.4.0 wrote,
    /// holds neither, and so says nothing of them; nothing too, with every fault noted, when either cannot be read.
    /// </summary>
    /// <remarks>
    /// A part records both: one that holds only one of them, or — from the second version — neither, is refused. The norm the
    /// run's gradients were clipped to is read where the part holds it, from the third version on; a part that holds none says
    /// they were never clipped.
    /// </remarks>
    public Pace? PaceOf()
    {
        var batchSize = part.Member(BatchSizeKey);
        var earlyStopping = part.Member(EarlyStoppingKey);
        var clipNorm = part.Member(ClipNormKey);

        if (version == FirstVersion && batchSize is null && earlyStopping is null && clipNorm is null)
        {
            return null;
        }

        var rows = batchSize?.AsWhole() is { } whole && whole >= 1 ? whole : (int?)null;

        if (rows is null)
        {
            text.Fault([.. _here, BatchSizeKey], $"The batch size is the whole number of rows a batch of the run held, from one, under '{BatchSizeKey}'.");
        }

        var stopping = EarlyStoppingOf(earlyStopping);
        var clip = ClipOf(clipNorm);

        return rows is { } size ? new Pace(size, stopping) { Clip = clip } : null;
    }

    /// <summary>
    /// The engine the run was on, as it named itself; nothing when the part names none — as no part 0.4.0 wrote does — and
    /// so says nothing of it; nothing too, with its fault noted, when it is written as anything but its name and, where the
    /// engine named them, its version and its device, each as text.
    /// </summary>
    public Engine? EngineOf()
    {
        if (part.Member(EngineKey) is not { } written)
        {
            return null;
        }

        // Anything but an object holds no name, and is refused by that.
        string[] at = [.. _here, EngineKey];
        var version = written.Member(VersionKey);
        var device = written.Member(DeviceKey);

        if (written.Member(NameKey) is not { ValueKind: JsonValueKind.String } name
            || version is { ValueKind: not JsonValueKind.String }
            || device is { ValueKind: not JsonValueKind.String })
        {
            text.Fault(
                at,
                $"The engine the run was on is written under '{EngineKey}' as its '{NameKey}' and, where the engine names them, its '{VersionKey}' and its '{DeviceKey}', each as text.");

            return null;
        }

        Only(written, at, NameKey, VersionKey, DeviceKey);

        return new Engine(name.GetString()!, version?.GetString(), device?.GetString());
    }

    /// <summary>What the optimizer remembered of each parameter, by path: only what could be read whole.</summary>
    public Dictionary<string, SlotMemory> MemoryOf(Dictionary<string, Slot> slots)
    {
        var memory = new Dictionary<string, SlotMemory>(StringComparer.Ordinal);
        string[] under = [.. _here, MemoryKey];

        if (part.Member(MemoryKey) is not { ValueKind: JsonValueKind.Object } written)
        {
            text.Fault(under, $"What the optimizer remembers is written by path, as an object under '{MemoryKey}'.");

            return memory;
        }

        foreach (var entry in written.EnumerateObject())
        {
            string[] at = [.. under, entry.Name];

            if (!slots.TryGetValue(entry.Name, out var slot) || slot is not Parameter)
            {
                text.Fault(at, $"'{entry.Name.Quoted()}' is no parameter of this network.");
            }
            else if (entry.Value.Member(StepsKey)?.AsWhole() is not { } steps || entry.Value.Member(TensorsKey) is not { ValueKind: JsonValueKind.Object } tensors)
            {
                text.Fault(at, $"What is remembered of '{entry.Name}' is written as its '{StepsKey}' and its '{TensorsKey}'.");
            }
            else
            {
                Only(entry.Value, at, StepsKey, TensorsKey);

                if (TensorsOf(tensors, [.. at, TensorsKey], slot.Value.Shape, entry.Name) is { } kept)
                {
                    memory[entry.Name] = new SlotMemory(steps, kept);
                }
            }
        }

        return memory;
    }

    /// <summary>Puts what the optimizer remembered of each parameter back into it: the optimizer is the one that says what it keeps.</summary>
    public void Recall(Optimizer optimizer, Dictionary<string, Slot> slots, Dictionary<string, SlotMemory> memory)
    {
        foreach (var (path, remembered) in memory)
        {
            try
            {
                optimizer.PutBack((Parameter)slots[path], remembered);
            }
            catch (ArgumentException refusal)
            {
                text.Fault([.. _here, MemoryKey, path], refusal);
            }
        }
    }

    /// <summary>The epochs the run had gone through; nothing when they cannot all be read.</summary>
    public List<Epoch>? HistoryOf()
    {
        string[] under = [.. _here, HistoryKey];

        if (part.Member(HistoryKey) is not { ValueKind: JsonValueKind.Array } written)
        {
            text.Fault(under, $"The epochs so far are written as a list under '{HistoryKey}'.");

            return null;
        }

        var history = new List<Epoch>();
        var place = 0;

        foreach (var epoch in written.EnumerateArray())
        {
            string[] at = [.. under, place.ToString(CultureInfo.InvariantCulture)];
            var validation = epoch.Member(ValidationLossKey);

            if (epoch.Member(NumberKey)?.AsWhole() == place
                && epoch.Member(EpochLossKey)?.AsNumber() is { } loss
                && epoch.Member(LearningRateKey)?.AsNumber() is { } rate
                && (validation is null || validation.Value.AsNumber() is not null))
            {
                Only(epoch, at, NumberKey, EpochLossKey, ValidationLossKey, LearningRateKey);
                history.Add(new Epoch(place, loss, validation?.AsNumber(), rate));
            }
            else
            {
                text.Fault(
                    at,
                    $"An epoch is written as its '{NumberKey}' — its place in the history — its '{EpochLossKey}', its '{LearningRateKey}' and, with validation rows, its '{ValidationLossKey}'.");
            }

            place++;
        }

        return history.Count == place ? history : null;
    }

    /// <summary>How far early stopping had got; nothing when the run was not judged, or it cannot be read.</summary>
    /// <param name="network">The network read from the same file, whose slots the best epoch's are held to.</param>
    /// <param name="epochs">How many epochs the history holds; nothing when it cannot be read, and the best epoch cannot be held to it.</param>
    public Judgement? JudgementOf(Network network, int? epochs)
    {
        if (part.Member(JudgementKey) is not { } written)
        {
            return null;
        }

        string[] at = [.. _here, JudgementKey];

        if (written.Member(WaitKey)?.AsWhole() is not { } wait || wait < 0
            || written.Member(BestKey)?.AsNumber() is not { } best
            || written.Member(BestEpochKey)?.AsWhole() is not { } bestEpoch || bestEpoch < 0 || bestEpoch >= epochs
            || written.Member(StopsKey)?.AsTruth() is not { } stops)
        {
            text.Fault(
                at,
                $"How far early stopping had got is written as its '{WaitKey}', '{BestKey}', '{BestEpochKey}' — an epoch of the history — and '{StopsKey}'.");

            return null;
        }

        Only(written, at, WaitKey, BestKey, BestEpochKey, StopsKey, BestSlotsKey);

        IReadOnlyDictionary<string, Tensor>? bestSlots = null;

        if (written.Member(BestSlotsKey) is { } kept)
        {
            string[] under = [.. at, BestSlotsKey];

            if (kept.ValueKind == JsonValueKind.Object)
            {
                Only(kept, under, ParametersKey, StateKey);
            }

            bestSlots = SlotsUnder(kept, under, network).Values;
        }

        return new Judgement(wait, best, bestEpoch) { BestSlots = bestSlots, Stops = stops };
    }

    /// <summary>What the network was trained on, when the part says; nothing, with a fault noted when it says it wrongly.</summary>
    /// <remarks>
    /// Which features held one value on every training row is said by a part of the second version, and by none of the
    /// first: where it is not said, it is read as not said — never as saying that none did.
    /// </remarks>
    public TrainedOn? TrainedOnOf()
    {
        if (part.Member(TrainedOnKey) is not { } written)
        {
            return null;
        }

        string[] at = [.. _here, TrainedOnKey];

        if (Names(written.Member(FeaturesKey)) is not { } features
            || Names(written.Member(AnswersKey)) is not { } answers
            || written.Member(OutputKey) is not { ValueKind: JsonValueKind.String } output
            || written.Member(TrainedBehindKey) is not { ValueKind: JsonValueKind.String } behind
            || written.Member(SeedKey)?.AsLong() is not { } seed
            || written.Member(EpochKey)?.AsWhole() is not { } epoch || epoch < 0)
        {
            text.Fault(
                at,
                $"What the network was trained on is written as its '{FeaturesKey}' and '{AnswersKey}' — lists of names — its '{OutputKey}' and '{TrainedBehindKey}', its '{SeedKey}' and its '{EpochKey}'.");

            return null;
        }

        Only(written, at, FeaturesKey, AnswersKey, OutputKey, TrainedBehindKey, SeedKey, EpochKey, UnvariedKey);

        return new TrainedOn
        {
            Features = features,
            Answers = answers,
            Output = output.GetString()!,
            TrainedBehind = behind.GetString()!,
            Seed = seed,
            Epoch = epoch,
            Unvaried = UnvariedOf(written.Member(UnvariedKey), [.. at, UnvariedKey], features),
        };
    }

    // Each feature that held one value on every training row, with that value: nothing when the part does not say. Every name
    // is a feature of the network and every value a finite number; a fault is noted for any other, and what is read of the rest
    // is refused with the file.
    private Dictionary<string, double>? UnvariedOf(JsonElement? written, string[] at, string[] features)
    {
        if (written is not { } unvaried)
        {
            return null;
        }

        if (unvaried.ValueKind != JsonValueKind.Object)
        {
            text.Fault(at, $"Which features held one value on every training row is written as an object under '{UnvariedKey}': each feature by its name, with that value.");

            return null;
        }

        var held = new Dictionary<string, double>(StringComparer.Ordinal);

        foreach (var entry in unvaried.EnumerateObject())
        {
            string[] place = [.. at, entry.Name];

            if (!features.Contains(entry.Name))
            {
                text.Fault(place, $"'{entry.Name.Quoted()}' is no feature this network was trained on.");
            }
            else if (entry.Value.AsNumber() is { } value)
            {
                held[entry.Name] = value;
            }
            else
            {
                text.Fault(place, $"'{entry.Name}' is written with the one value it held on every training row, a finite number, and this is not one.");
            }
        }

        return held;
    }

    // The early stopping the run went under, as its patience, its least fall that counts and whether it restores the best;
    // nothing for a run that had none, written as null — and nothing, with its fault noted, when it is missing or written as
    // anything else, the ranges its settings take held as early stopping holds them, so reading one never throws.
    private EarlyStopping? EarlyStoppingOf(JsonElement? written)
    {
        if (written is { ValueKind: JsonValueKind.Null })
        {
            return null;
        }

        string[] at = [.. _here, EarlyStoppingKey];

        if (written is not { ValueKind: JsonValueKind.Object } stopping
            || stopping.Member(PatienceKey)?.AsWhole() is not { } patience || patience < 0
            || stopping.Member(MinDeltaKey)?.AsNumber() is not { } minDelta || minDelta < 0
            || stopping.Member(RestoreBestKey)?.AsTruth() is not { } restoreBest)
        {
            text.Fault(
                at,
                $"The early stopping the run went under is written under '{EarlyStoppingKey}' as its '{PatienceKey}' — a whole number from nought — its '{MinDeltaKey}' — a number from nought — and its '{RestoreBestKey}', true or false; or as null, for a run that had none.");

            return null;
        }

        Only(stopping, at, PatienceKey, MinDeltaKey, RestoreBestKey);

        return new EarlyStopping { Patience = patience, MinDelta = minDelta, RestoreBest = restoreBest };
    }

    // The clip the run's gradients went under: nothing where the part holds no norm — and nothing, with its fault noted, where
    // it holds one that is no number above nothing, or holds one at all in a part of a version that cannot say it.
    private GradientClip? ClipOf(JsonElement? written)
    {
        if (written is null)
        {
            return null;
        }

        string[] at = [.. _here, ClipNormKey];

        if (version < NetworkDocument.Version)
        {
            text.Fault(
                at,
                $"The norm a run's gradients were clipped to is written from the third version of a training part on, and this one is of the {(version == FirstVersion ? "first" : "second")}.");

            return null;
        }

        if (written.Value.AsNumber() is { } norm && norm > 0)
        {
            return new GradientClip(norm);
        }

        text.Fault(at, $"The norm the run's gradients were clipped to is written under '{ClipNormKey}', as a number above nothing.");

        return null;
    }

    // A list of names written here; nothing when it is anything else.
    private static string[]? Names(JsonElement? written) =>
        written is { ValueKind: JsonValueKind.Array } list && list.EnumerateArray().All(name => name.ValueKind == JsonValueKind.String)
            ? [.. list.EnumerateArray().Select(name => name.GetString()!)]
            : null;

    // Every slot of the network, written under a holder: the parameters under one key and the running statistics under the
    // other, each by its path — handed to the one load every file's numbers go into a network by, which holds them to it: all
    // of them, and no other. Which key a slot stands under is this file's own, and checked here; every other fault is the
    // load's, placed where the slot is written — or, for a slot not written, at the key its kind stands under, unless that
    // key's own fault already says nothing under it could be read.
    private SlotLoad SlotsUnder(JsonElement holder, string[] under, Network network)
    {
        var load = new SlotLoad(network.Slots());
        var places = new Dictionary<string, string[]>(StringComparer.Ordinal);
        var unreadable = new List<bool>();

        foreach (var learns in (bool[])[true, false])
        {
            var key = learns ? ParametersKey : StateKey;
            string[] at = [.. under, key];

            if (holder.Member(key) is not { ValueKind: JsonValueKind.Object } written)
            {
                text.Fault(at, $"The {(learns ? "parameters" : "running statistics")} are written by their paths, as an object under '{key}'.");
                unreadable.Add(learns);

                continue;
            }

            foreach (var entry in written.EnumerateObject())
            {
                string[] place = [.. at, entry.Name];
                var source = Pointer(place);
                places[source] = place;

                if (load.Slots.TryGetValue(entry.Name, out var slot) && slot is Parameter != learns)
                {
                    text.Fault(place, $"'{entry.Name}' is {(learns ? "a running statistic" : "a parameter")}, and is written under '{(learns ? StateKey : ParametersKey)}'.");
                }
                else if (load.Handed(entry.Name, source) is { } handed && TensorOf(entry.Value, place, handed.Value.Shape, entry.Name) is { } value)
                {
                    load.Keep(new SlotEntry(entry.Name, value, source));
                }
            }
        }

        foreach (var fault in load.Faults)
        {
            if (fault.Source is { } source)
            {
                text.Fault(places[source], fault.Message);

                continue;
            }

            var learns = load.Slots[fault.Slot] is Parameter;

            if (!unreadable.Contains(learns))
            {
                text.Fault([.. under, learns ? ParametersKey : StateKey], fault.Message);
            }
        }

        return load;
    }

    // Where a place stands in the file, as a JSON pointer spells it: the keys from the top, each with its '~' and '/' escaped.
    private static string Pointer(string[] place) =>
        string.Join('/', place.Select(key => key.Replace("~", "~0", StringComparison.Ordinal).Replace("/", "~1", StringComparison.Ordinal)));

    // The tensors remembered of one parameter, each of its shape; nothing when any of them cannot be read.
    private Dictionary<string, Tensor>? TensorsOf(JsonElement tensors, string[] at, Shape shape, string path)
    {
        var kept = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        var readable = true;

        foreach (var tensor in tensors.EnumerateObject())
        {
            if (TensorOf(tensor.Value, [.. at, tensor.Name], shape, $"{path} {tensor.Name.Quoted()}") is { } value)
            {
                kept[tensor.Name] = value;
            }
            else
            {
                readable = false;
            }
        }

        return readable ? kept : null;
    }

    // A tensor written as its shape and its values, which must be the shape the slot holds — in the words the load refuses
    // another shape with — and as many finite numbers as it has, read from the text where they stand.
    private Tensor? TensorOf(JsonElement written, string[] at, Shape expected, string saying)
    {
        if (written.Member(ShapeKey) is not { ValueKind: JsonValueKind.Array } shape || written.Member(ValuesKey) is not { ValueKind: JsonValueKind.Array })
        {
            text.Fault(at, $"'{saying}' is written as its '{ShapeKey}' and its '{ValuesKey}'.");

            return null;
        }

        Only(written, at, ShapeKey, ValuesKey);

        if (expected.Unlike([.. shape.EnumerateArray().Select(length => length.AsWhole())], saying) is { } unlike)
        {
            text.Fault([.. at, ShapeKey], unlike);

            return null;
        }

        string[] listed = [.. at, ValuesKey];
        var count = text.CountOf(listed);

        if (count != expected.Count)
        {
            text.Fault(
                listed,
                string.Create(CultureInfo.InvariantCulture, $"'{saying}' holds as many values as its {expected} shape, {expected.Count}, and the file holds {count}."));

            return null;
        }

        // Read where they stand, into the one array the tensor keeps: a network's file holds millions of them.
        return text.FloatsOf(listed, count, $"'{saying}' holds finite numbers, and this is not one.") is { } numbers ? Tensor.Wrap(expected, numbers) : null;
    }

    // An object holds only the keys it is written with: any other is refused where it stands, rather than read as nothing.
    private void Only(JsonElement written, string[] at, params string[] keys)
    {
        foreach (var name in written.EnumerateObject().Select(each => each.Name).Where(name => !keys.Contains(name)))
        {
            text.Fault([.. at, name], $"'{name.Quoted()}' is not written here: this holds {string.Join(", ", keys.Select(key => $"'{key}'"))}.");
        }
    }
}
