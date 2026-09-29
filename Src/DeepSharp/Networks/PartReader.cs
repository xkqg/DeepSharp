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
/// fault is noted where it stands and the reading goes on, so a file is refused once, with every fault in it.
/// </remarks>
internal sealed class PartReader(NetworkText text, string property, JsonElement part)
{
    private readonly string[] _here = [property];

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

    /// <summary>Reads every slot written into the network, each by its path: every slot the network holds, and none it does not.</summary>
    public void Load(Network network)
    {
        var slots = network.Slots().ToDictionary(named => named.Path, named => named.Slot);
        var read = SlotsUnder(part, _here, slots);

        if (!text.Faulty)
        {
            foreach (var (path, value) in read)
            {
                slots[path].Replace(value);
            }
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
                text.Fault(at, $"'{entry.Name}' is no parameter of this network.");
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
                optimizer.Recall((Parameter)slots[path], remembered);
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
    /// <param name="slots">The network's slots, by path.</param>
    /// <param name="epochs">How many epochs the history holds; nothing when it cannot be read, and the best epoch cannot be held to it.</param>
    public Judgement? JudgementOf(Dictionary<string, Slot> slots, int? epochs)
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

        Dictionary<string, Tensor>? bestSlots = null;

        if (written.Member(BestSlotsKey) is { } kept)
        {
            string[] under = [.. at, BestSlotsKey];

            if (kept.ValueKind == JsonValueKind.Object)
            {
                Only(kept, under, ParametersKey, StateKey);
            }

            bestSlots = SlotsUnder(kept, under, slots);
        }

        return new Judgement(wait, best, bestEpoch) { BestSlots = bestSlots, Stops = stops };
    }

    /// <summary>What the network was trained on, when the part says; nothing, with a fault noted when it says it wrongly.</summary>
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

        Only(written, at, FeaturesKey, AnswersKey, OutputKey, TrainedBehindKey, SeedKey, EpochKey);

        return new TrainedOn { Features = features, Answers = answers, Output = output.GetString()!, TrainedBehind = behind.GetString()!, Seed = seed, Epoch = epoch };
    }

    // A list of names written here; nothing when it is anything else.
    private static string[]? Names(JsonElement? written) =>
        written is { ValueKind: JsonValueKind.Array } list && list.EnumerateArray().All(name => name.ValueKind == JsonValueKind.String)
            ? [.. list.EnumerateArray().Select(name => name.GetString()!)]
            : null;

    // Every slot of the network, written under a holder: the parameters under one key and the running statistics under the
    // other, each by its path — all of them, and no other.
    private Dictionary<string, Tensor> SlotsUnder(JsonElement holder, string[] under, Dictionary<string, Slot> slots)
    {
        var read = new Dictionary<string, Tensor>(StringComparer.Ordinal);

        foreach (var learns in (bool[])[true, false])
        {
            var key = learns ? ParametersKey : StateKey;
            string[] at = [.. under, key];

            if (holder.Member(key) is not { ValueKind: JsonValueKind.Object } written)
            {
                text.Fault(at, $"The {(learns ? "parameters" : "running statistics")} are written by their paths, as an object under '{key}'.");

                continue;
            }

            foreach (var entry in written.EnumerateObject())
            {
                string[] place = [.. at, entry.Name];

                if (!slots.TryGetValue(entry.Name, out var slot))
                {
                    text.Fault(place, $"'{entry.Name}' is no slot of this network.");
                }
                else if (slot is Parameter != learns)
                {
                    text.Fault(place, $"'{entry.Name}' is {(learns ? "a running statistic" : "a parameter")}, and is written under '{(learns ? StateKey : ParametersKey)}'.");
                }
                else if (TensorOf(entry.Value, place, slot.Value.Shape, entry.Name) is { } value)
                {
                    read[entry.Name] = value;
                }
            }

            foreach (var missing in slots.Where(named => named.Value is Parameter == learns && written.Member(named.Key) is null))
            {
                text.Fault(at, $"'{missing.Key}' is missing: every slot of the network is written.");
            }
        }

        return read;
    }

    // The tensors remembered of one parameter, each of its shape; nothing when any of them cannot be read.
    private Dictionary<string, Tensor>? TensorsOf(JsonElement tensors, string[] at, Shape shape, string path)
    {
        var kept = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        var readable = true;

        foreach (var tensor in tensors.EnumerateObject())
        {
            if (TensorOf(tensor.Value, [.. at, tensor.Name], shape, $"{path} {tensor.Name}") is { } value)
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

    // A tensor written as its shape and its values, which must be the shape the slot holds and as many finite numbers as it has.
    private Tensor? TensorOf(JsonElement written, string[] at, Shape expected, string saying)
    {
        if (written.Member(ShapeKey) is not { ValueKind: JsonValueKind.Array } shape || written.Member(ValuesKey) is not { ValueKind: JsonValueKind.Array } values)
        {
            text.Fault(at, $"'{saying}' is written as its '{ShapeKey}' and its '{ValuesKey}'.");

            return null;
        }

        Only(written, at, ShapeKey, ValuesKey);

        int?[] lengths = [.. shape.EnumerateArray().Select(length => length.AsWhole())];

        if (!lengths.SequenceEqual(expected.Axes.ToArray().Select(length => (int?)length)))
        {
            var said = string.Join('x', lengths.Select(length => length?.ToString(CultureInfo.InvariantCulture) ?? "?"));
            text.Fault([.. at, ShapeKey], $"'{saying}' is a {expected} slot here, and is written as {said}.");

            return null;
        }

        var elements = values.EnumerateArray().ToArray();

        if (elements.Length != expected.Count)
        {
            text.Fault(
                [.. at, ValuesKey],
                string.Create(CultureInfo.InvariantCulture, $"'{saying}' holds as many values as its {expected} shape, {expected.Count}, and the file holds {elements.Length}."));

            return null;
        }

        var numbers = new float[elements.Length];
        var readable = true;

        for (var place = 0; place < elements.Length; place++)
        {
            if (elements[place].AsFloat() is { } number)
            {
                numbers[place] = number;
            }
            else
            {
                text.Fault([.. at, ValuesKey, place.ToString(CultureInfo.InvariantCulture)], $"'{saying}' holds finite numbers, and this is not one.");
                readable = false;
            }
        }

        return readable ? Tensor.From(expected, numbers) : null;
    }

    // An object holds only the keys it is written with: any other is refused where it stands, rather than read as nothing.
    private void Only(JsonElement written, string[] at, params string[] keys)
    {
        foreach (var name in written.EnumerateObject().Select(each => each.Name).Where(name => !keys.Contains(name)))
        {
            text.Fault([.. at, name], $"'{name}' is not written here: this holds {string.Join(", ", keys.Select(key => $"'{key}'"))}.");
        }
    }
}
