// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Tensors;

using DeepSharp.Networks;

namespace DeepSharp.Import.Keras;

/// <summary>
/// What one layer of a Keras model becomes here: the words it is written in, the activation it carries after them, the
/// slots its numbers go into, in the order Keras keeps them, and the sides of the window it walks, when it has one.
/// </summary>
/// <param name="Words">What it writes into a description in Keras's words; nothing for a layer that is its activation alone.</param>
/// <param name="Activation">The activation after it, by Keras's name for it: <c>linear</c> for none.</param>
internal readonly record struct KerasLayer(Action<Sequential>? Words, string Activation)
{
    /// <summary>The slots its numbers go into, in the order Keras keeps its numbers.</summary>
    public IReadOnlyList<KerasSlot> Slots { get; init; } = [];

    /// <summary>
    /// The slots a file may leave without a number because Keras saves them merged into another: a low-rank pair, whose product
    /// the archive Keras 3 saves adds into the kernel. The layer begins them afresh, and a file that holds them gives their numbers.
    /// </summary>
    public IReadOnlyList<KerasSlot> Fresh { get; init; } = [];

    /// <summary>
    /// The slot each number goes into by the name the file gives it, for a layer whose numbers the HDF5 file Keras saved before
    /// does not list in the order the layer keeps them: it lists a layer's trainable numbers and then those that do not train, so
    /// a layer with a held kernel lists its bias and its pair before its kernel. A name no entry here ends in is placed by its place.
    /// </summary>
    public IReadOnlyDictionary<string, KerasSlot> Called { get; init; } = new Dictionary<string, KerasSlot>();

    /// <summary>The slot a number goes into by the name the file gives it, when this layer places its numbers by name.</summary>
    /// <param name="name">The name the file gives the number: a dataset's path, whose last part is the number's own name.</param>
    /// <returns>The slot; nothing when no name here is the name's last part.</returns>
    public KerasSlot? SlotCalled(string name) =>
        Called.Count > 0 && Called.TryGetValue(name[(name.LastIndexOf('/') + 1)..], out var slot) ? (KerasSlot?)slot : null;

    /// <summary>
    /// What the layer's numbers become, for a layer whose numbers the file holds otherwise than the slots keep them: in another order, split in
    /// two, or fewer than the slots. It takes the numbers the file holds, in the order the file lists them, and gives the slots' numbers.
    /// </summary>
    public Func<IReadOnlyList<HeldNumber>, IReadOnlyList<RemadeNumber>>? Remade { get; init; }

    /// <summary>How many layers of a description this one Keras layer writes; one, unless it is more than one layer here.</summary>
    /// <remarks>Keras's separable convolution is a depthwise convolution and a convolution of a one-by-one window, which are two layers here.</remarks>
    public int Places { get; init; } = 1;

    /// <summary>The length of each side of the window it walks, outermost first, for a convolution: none for any other layer.</summary>
    public int[] Sizes { get; init; } = [];

    /// <summary>
    /// The shape of an example the network takes in place of the model's input, for a first layer that takes another: an
    /// embedding, whose category Keras is handed as its place and the network here as its ones and noughts. Nothing for any other.
    /// </summary>
    public Shape? Takes { get; init; }

    /// <summary>
    /// The shape a number of this layer is handed to its slot in: a convolution's kernel, which Keras writes as one length
    /// for each side of its window, then the channels in and the channels out, as the window's places times the channels in
    /// by the channels out — the same numbers in the same order — when the lengths it begins with are the window's sides;
    /// any other number as the file writes it.
    /// </summary>
    public Shape LaidOut(ulong[] lengths, KerasSlot slot)
    {
        var written = new Shape([.. lengths.Select(length => checked((int)length))]);

        if (slot.Kept == KerasKeeping.Flat)
        {
            return new Shape(written.Count);
        }

        var sides = slot.Sides.Length > 0 ? slot.Sides : Sizes;

        if (sides.Length == 0 || written.Rank != sides.Length + 2 || !written.Axes[..sides.Length].SequenceEqual(sides))
        {
            return written;
        }

        var places = sides.Aggregate(1, (all, side) => checked(all * side));
        var channels = written[sides.Length];
        var made = written[written.Rank - 1];

        return slot.Kept switch
        {
            // A depthwise kernel: the window's places by the channels it is handed times the channels it makes of each.
            KerasKeeping.Apart => new Shape(places, checked(channels * made)),

            // A transposed kernel: a row for every channel in, holding the window's places by the channels out.
            KerasKeeping.GivenBack => new Shape(made, checked(places * channels)),
            _ => new Shape(checked(places * channels), made),
        };
    }

    /// <summary>
    /// A number as the library keeps it, where Keras keeps the same numbers in another order: a transposed convolution's
    /// kernel, which Keras writes the window's places by the channels out by the channels in.
    /// </summary>
    /// <param name="values">The numbers as the file holds them.</param>
    /// <param name="lengths">The lengths the file writes them under.</param>
    /// <param name="slot">The slot they go into.</param>
    /// <returns>The same numbers, in the order the slot keeps them.</returns>
    public float[] Turned(float[] values, ulong[] lengths, KerasSlot slot)
    {
        var sides = slot.Sides.Length > 0 ? slot.Sides : Sizes;

        if (slot.Kept != KerasKeeping.GivenBack || lengths.Length != sides.Length + 2)
        {
            return values;
        }

        var places = sides.Aggregate(1, (all, side) => checked(all * side));
        var made = checked((int)lengths[^2]);
        var channels = checked((int)lengths[^1]);
        var turned = new float[values.Length];

        for (var place = 0; place < places; place++)
        {
            for (var output = 0; output < made; output++)
            {
                for (var input = 0; input < channels; input++)
                {
                    turned[(input * places * made) + (place * made) + output] = values[(((place * made) + output) * channels) + input];
                }
            }
        }

        return turned;
    }
}

/// <summary>A number of a layer as its slot keeps it, made of the numbers the file holds.</summary>
/// <param name="Slot">The slot's name in the layer.</param>
/// <param name="Shape">The shape the numbers are handed to the slot in.</param>
/// <param name="Values">The numbers, row-major.</param>
/// <param name="Source">Where the file holds the numbers they were made of, for a fault to name.</param>
internal readonly record struct RemadeNumber(string Slot, Shape Shape, float[] Values, string Source);

/// <summary>How Keras keeps a number that a layer here keeps in another shape, or in another order.</summary>
internal enum KerasKeeping
{
    /// <summary>A convolution's kernel: the window's places by the channels in by the channels out, which is what a layer here keeps.</summary>
    AsWritten,

    /// <summary>A depthwise kernel: the window's places by the channels it is handed by the channels it makes of each, which are its channels out together.</summary>
    Apart,

    /// <summary>A transposed kernel: the window's places by the channels out by the channels in, which a layer here keeps the other way about.</summary>
    GivenBack,

    /// <summary>
    /// A parametric rectifier's shares: one for each channel, Keras writing a length of one for every axis they are shared over,
    /// which a layer here keeps as the channels alone, in the same order.
    /// </summary>
    Flat,
}

/// <summary>Where one of a Keras layer's numbers goes: the slot's name, how many layers on from this one it stands, and the window its lengths begin with.</summary>
/// <param name="Named">The slot's name.</param>
/// <param name="Away">How many layers on from this one the slot stands; none, for this layer's own.</param>
/// <param name="Sides">The length of each side of the window its lengths begin with; none, to read the layer's own.</param>
internal readonly record struct KerasSlot(string Named, int Away, int[] Sides)
{
    /// <summary>How Keras keeps the number that goes into it; as a layer here keeps it, unless said.</summary>
    public KerasKeeping Kept { get; init; }

    /// <summary>A slot of the layer's own, whose lengths begin with the layer's own window.</summary>
    /// <param name="named">The slot's name.</param>
    public static implicit operator KerasSlot(string named) => new(named, 0, []);
}

/// <summary>
/// A kind of Keras layer read here: what its settings are written into, in Keras's words, and what of them is refused.
/// </summary>
/// <remarks>The kinds listed here are every kind read; a layer of any other is refused, naming them.</remarks>
internal abstract class KerasKind
{
    // Every kind read, in the order a refusal names them.
    private static readonly KerasKind[] Covered =
    [
        new DenseKind(),
        new ConvolutionKind("Conv1D", 1, (filters, window, start) => description => description.Conv1D(filters, window.Line(), window.Groups, start)),
        new ConvolutionKind("Conv2D", 2, (filters, window, start) => description => description.Conv2D(filters, window.Plane(), window.Groups, start)),
        new ConvolutionKind("Conv3D", 3, (filters, window, start) => description => description.Conv3D(filters, window.Volume(), window.Groups, start)),
        new TransposedConvolutionKind("Conv1DTranspose", 1),
        new TransposedConvolutionKind("Conv2DTranspose", 2),
        new TransposedConvolutionKind("Conv3DTranspose", 3),
        new DepthwiseConvolutionKind("DepthwiseConv1D", 1),
        new DepthwiseConvolutionKind("DepthwiseConv2D", 2),
        new SeparableConvolutionKind("SeparableConv1D", 1),
        new SeparableConvolutionKind("SeparableConv2D", 2),
        new PaddingKind("ZeroPadding1D", 1),
        new PaddingKind("ZeroPadding2D", 2),
        new PaddingKind("ZeroPadding3D", 3),
        new PoolingKind("MaxPooling1D", 1, window => description => description.MaxPool1D(window.Line())),
        new PoolingKind("MaxPooling2D", 2, window => description => description.MaxPool2D(window.Plane())),
        new PoolingKind("MaxPooling3D", 3, window => description => description.MaxPool3D(window.Volume())),
        new PoolingKind("AveragePooling1D", 1, window => description => description.AvgPool1D(window.Line())),
        new PoolingKind("AveragePooling2D", 2, window => description => description.AvgPool2D(window.Plane())),
        new PoolingKind("AveragePooling3D", 3, window => description => description.AvgPool3D(window.Volume())),
        new GlobalPoolingKind("GlobalMaxPooling1D", 1, keepsAxes => description => description.GlobalMaxPool1D(keepsAxes)),
        new GlobalPoolingKind("GlobalMaxPooling2D", 2, keepsAxes => description => description.GlobalMaxPool2D(keepsAxes)),
        new GlobalPoolingKind("GlobalMaxPooling3D", 3, keepsAxes => description => description.GlobalMaxPool3D(keepsAxes)),
        new GlobalPoolingKind("GlobalAveragePooling1D", 1, keepsAxes => description => description.GlobalAvgPool1D(keepsAxes)),
        new GlobalPoolingKind("GlobalAveragePooling2D", 2, keepsAxes => description => description.GlobalAvgPool2D(keepsAxes)),
        new GlobalPoolingKind("GlobalAveragePooling3D", 3, keepsAxes => description => description.GlobalAvgPool3D(keepsAxes)),
        new BatchNormalizationKind(), new LayerNormalizationKind(), new DropoutKind(),
        new SpatialDropoutKind("SpatialDropout1D", 1, rate => description => description.SpatialDropout1D(rate)),
        new SpatialDropoutKind("SpatialDropout2D", 2, rate => description => description.SpatialDropout2D(rate)),
        new SpatialDropoutKind("SpatialDropout3D", 3, rate => description => description.SpatialDropout3D(rate)),
        new GroupNormalizationKind(), new RmsNormalizationKind(), new EmbeddingKind(),
        new RecurrentKind("LSTM"), new RecurrentKind("GRU"), new BidirectionalKind(),
        new FlattenKind(), new ReshapeKind(), new ActivationKind(), new ReluKind(),
        new LeakyReluKind(), new PReluKind(), new EluKind(), new SoftmaxKind(),
    ];

    /// <summary>Every kind read, by the class name Keras writes it under.</summary>
    public static IReadOnlyDictionary<string, KerasKind> ByName { get; } = Covered.ToDictionary(kind => kind.Name, StringComparer.Ordinal);

    /// <summary>The kinds read, named as a refusal names them: <c>Dense, Conv1D, … or ReLU</c>.</summary>
    public static string Listed { get; } = $"{string.Join(", ", Covered[..^1].Select(kind => kind.Name))} or {Covered[^1].Name}";

    /// <summary>The class name Keras writes the kind under.</summary>
    public abstract string Name { get; }

    /// <summary>Notes a fault when a layer works in another precision than single precision, as a network here does.</summary>
    /// <remarks>Keras writes a layer's precision as a policy's name, or as the name alone.</remarks>
    public static void RequireSinglePrecision(KerasSettings layer)
    {
        var policy = layer.Setting("dtype", default(JsonElement));
        var precision = policy.ValueKind == JsonValueKind.Object && policy.TryGetProperty("config", out var config) && config.TryGetProperty("name", out var name) ? name : policy;

        if (precision.ValueKind != JsonValueKind.Undefined && precision.ToString() != "float32")
        {
            layer.Refuse($"it works in {precision.ToString().Quoted()}, and a network here works in single precision, float32.");
        }
    }

    /// <summary>A layer of this kind, read from its settings; each fault of them noted there.</summary>
    public abstract KerasLayer Read(KerasSettings layer);

    /// <summary>The activation a layer carries, by Keras's name for it: <c>linear</c>, unless it says another.</summary>
    protected static string ActivationOf(KerasSettings layer) => layer.Setting("activation", "linear");

    /// <summary>
    /// The rules a Keras layer puts on its numbers, as a description says them: the penalty and the bound of its weights and of its
    /// bias, and whether it is held still.
    /// </summary>
    /// <param name="Weights">The rules for its kernel — or its scale.</param>
    /// <param name="Bias">The rules for its bias — or its shift.</param>
    /// <param name="Held">Whether the layer does not train.</param>
    internal readonly record struct Rules(SlotRules Weights, SlotRules Bias, bool Held)
    {
        /// <summary>The description with the rules put on the layer its last word makes.</summary>
        /// <param name="description">The description.</param>
        /// <returns>The same description.</returns>
        public Sequential Put(Sequential description)
        {
            if (Weights != default || Bias != default)
            {
                description.Ruled(Weights, Bias);
            }

            return Held ? description.Held() : description;
        }
    }

    /// <summary>Whether a layer is held still, as Keras's <c>trainable = False</c> holds it; a layer that penalises its outputs is refused.</summary>
    /// <param name="layer">The layer's settings.</param>
    /// <returns>Whether it does not train.</returns>
    protected static bool Held(KerasSettings layer)
    {
        if (layer.Says("activity_regularizer"))
        {
            layer.Refuse($"it penalises what it makes, activity_regularizer {layer.Written("activity_regularizer")!.Quoted()}, and a penalty here is on a layer's numbers, never on what it makes.");
        }

        return !layer.Setting("trainable", true);
    }

    /// <summary>The penalty and the bound a layer puts on its number of a kind, read in Keras's words.</summary>
    /// <param name="layer">The layer's settings.</param>
    /// <param name="prefix">What Keras calls the number in the settings: <c>kernel</c>, <c>bias</c>, <c>depthwise</c>, <c>gamma</c>.</param>
    /// <param name="rank">How many axes the number has in Keras's layout.</param>
    /// <param name="bounded">
    /// The axes a bound over each unit's numbers is over, in Keras's layout; nothing for a number a bound cannot be held on, which
    /// refuses the bound.
    /// </param>
    /// <returns>Its rules.</returns>
    protected static SlotRules Governing(KerasSettings layer, string prefix, int rank, int[]? bounded)
    {
        var rules = new SlotRules();
        var regularizer = $"{prefix}_regularizer";
        var constraint = $"{prefix}_constraint";

        if (layer.Says(regularizer))
        {
            var penalty = layer.Within(regularizer);
            var kind = penalty.Setting("class_name", string.Empty);
            var numbers = penalty.Within("config");
            var (l1, l2) = kind switch
            {
                "L1" => (numbers.Setting("l1", 0.01), 0.0),
                "L2" => (0.0, numbers.Setting("l2", 0.01)),
                "L1L2" => (numbers.Setting("l1", 0.0), numbers.Setting("l2", 0.0)),
                _ => (double.NaN, double.NaN),
            };

            if (double.IsNaN(l1))
            {
                layer.Refuse($"its {regularizer} is a {kind.Quoted()}, and the penalties a layer's numbers are put under here are L1, L2 and L1L2.");
            }
            else if (l1 < 0 || l2 < 0 || !double.IsFinite(l1) || !double.IsFinite(l2))
            {
                layer.Refuse($"its {regularizer} is a penalty of {l1} and {l2}, and a penalty is a finite number of at least nothing.");
            }
            else
            {
                rules = rules with { L1 = l1, L2 = l2 };
            }
        }

        return layer.Says(constraint) && Bound(layer, constraint, rank, bounded) is { } bound ? rules with { MaxNorm = bound } : rules;
    }

    // The bound of a MaxNorm constraint, over the axes a unit's numbers lie along; any other constraint, or another axis, is refused
    // and bounds nothing.
    private static double? Bound(KerasSettings layer, string key, int rank, int[]? bounded)
    {
        var constraint = layer.Within(key);
        var kind = constraint.Setting("class_name", string.Empty);

        if (kind != "MaxNorm")
        {
            layer.Refuse($"its {key} is a {kind.Quoted()}, and the only constraint a layer's numbers are put under here is MaxNorm.");

            return null;
        }

        if (bounded is null)
        {
            layer.Refuse($"its {key} bounds the norm of its numbers, and the kernel of a transposed convolution is laid out by the channels it takes, not by each unit it makes, so a bound by unit cannot be held on it.");

            return null;
        }

        var config = constraint.Within("config");
        var value = config.Setting("max_value", 2.0);
        var axis = config.Setting("axis", default(JsonElement));
        int[]? axes = axis.ValueKind switch
        {
            JsonValueKind.Undefined => [0],
            JsonValueKind.Number when axis.TryGetInt32(out var one) => [one],
            JsonValueKind.Array when axis.EnumerateArray().All(item => item.ValueKind == JsonValueKind.Number && item.TryGetInt32(out _)) => axis.EnumerateArray().Select(item => item.GetInt32()).ToArray(),
            _ => null,
        };
        var named = axes?.Select(each => each < 0 ? each + rank : each).Order().Distinct().ToArray();

        if (named is null || !named.SequenceEqual(bounded.Order()))
        {
            layer.Refuse(
                $"its {key} bounds the norm over axis {(axes is null ? axis.Written() : axis.ValueKind == JsonValueKind.Undefined ? "0" : axis.Written())}, and a bound here is over each unit's numbers: over axis {(bounded.Length == 1 ? bounded[0].ToString(CultureInfo.InvariantCulture) : $"[{string.Join(", ", bounded)}]")} of the {rank} axes of this number.");

            return null;
        }

        if (value <= 0 || !double.IsFinite(value))
        {
            layer.Refuse($"its {key} bounds the norm to {value}, and a bound is a finite number above nothing.");

            return null;
        }

        return value;
    }

    /// <summary>Whether a dense layer or a convolution adds a bias: Keras's <c>use_bias</c>, true unless it says otherwise.</summary>
    /// <param name="layer">The layer's settings.</param>
    /// <returns>Whether it adds one.</returns>
    protected static bool KeepsBias(KerasSettings layer) => layer.Setting("use_bias", true);

    /// <summary>How a layer starts, by whether it adds a bias.</summary>
    /// <param name="bias">Whether it adds one.</param>
    /// <returns>A beginning without draws, since the file's numbers take the place of any start.</returns>
    protected static Beginning Starting(bool bias) => new() { WithoutBias = !bias };

    /// <summary>The slots of a layer that has a kernel and may add a bias: the kernel, and the bias when there is one.</summary>
    /// <param name="kernel">The kernel's slot.</param>
    /// <param name="bias">Whether the layer adds a bias.</param>
    /// <returns>The slots, in the order Keras keeps its numbers.</returns>
    protected static KerasSlot[] WithBias(KerasSlot kernel, bool bias) => bias ? [kernel, "bias"] : [kernel];

    /// <summary>Notes a fault when a layer lays its images out other than with their channels last, as images here are.</summary>
    protected static void RequireChannelsLast(KerasSettings layer) => RequireChannelsLast(layer, "images");

    /// <summary>Notes a fault when a layer lays what it walks out other than with its channels last, as series, images and volumes here are.</summary>
    /// <param name="layer">The layer's settings.</param>
    /// <param name="things">What the layer walks, plural, as the fault says it: <c>series</c>, <c>images</c> or <c>volumes</c>.</param>
    protected static void RequireChannelsLast(KerasSettings layer, string things)
    {
        var format = layer.Setting("data_format", "channels_last");

        if (format != "channels_last")
        {
            layer.Refuse($"it lays its {things} out '{format.Quoted()}', and {things} here are laid out with their channels last.");
        }
    }

    /// <summary>What a layer that walks so many axes walks, plural: <c>series</c>, <c>images</c> or <c>volumes</c>.</summary>
    /// <param name="axes">How many axes: one, two or three.</param>
    protected static string ThingsWalkedAlong(int axes) => axes switch { 1 => "series", 2 => "images", _ => "volumes" };

    /// <summary>Notes a fault when a normalisation works over another axis than the last.</summary>
    protected static void RequireLastAxis(KerasSettings layer)
    {
        if (layer.Written("axis") is { } axis and not ("-1" or "[-1]"))
        {
            layer.Refuse($"it normalises over axis {axis.Quoted()}, and a normalisation here works over the last axis.");
        }
    }

    /// <summary>What a normalisation keeps: Keras's <c>scale</c> and <c>center</c>, both true unless it says otherwise.</summary>
    /// <param name="layer">The layer's settings.</param>
    /// <returns>Whether it multiplies by a scale of its own and adds a shift of its own.</returns>
    protected static Affine Keeping(KerasSettings layer) => new()
    {
        WithoutScale = !layer.Setting("scale", true),
        WithoutShift = !layer.Setting("center", true),
    };

    /// <summary>The slots of a normalisation's scale and shift, in the order Keras keeps them, of those it keeps.</summary>
    /// <param name="keeping">What it keeps.</param>
    /// <returns>The scale's slot and the shift's, of the ones it keeps.</returns>
    protected static IEnumerable<KerasSlot> Kept(Affine keeping)
    {
        if (!keeping.WithoutScale)
        {
            yield return "weight";
        }

        if (!keeping.WithoutShift)
        {
            yield return "bias";
        }
    }
}

/// <summary>Keras's <c>Dense</c>: a dense layer, its kernel inputs by outputs as the slot keeps it.</summary>
internal sealed class DenseKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Dense";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var units = layer.Setting<int>("units");
        var bias = KeepsBias(layer);
        var pair = layer.Says("lora_rank") ? PairOf(layer) : null;
        var start = Starting(bias) with { LowRank = pair };
        var rules = new Rules(Governing(layer, "kernel", 2, [0]), bias ? Governing(layer, "bias", 1, [0]) : default, Held(layer));

        // Keras saves a layer with a pair as the archive and as the HDF5 file it saved before, and they hold it differently: the
        // file holds the kernel, the bias and both matrices, each as it is; the archive holds the kernel merged with the pair's
        // product and neither matrix, so the layer begins its pair afresh.
        KerasSlot[] slots = pair is null ? WithBias("weight", bias) : [.. WithBias("weight", bias), "lora_a", "lora_b"];

        return new KerasLayer(description => rules.Put(description.Dense(units, start)), ActivationOf(layer))
        {
            Slots = slots,
            Fresh = pair is null ? [] : ["lora_a", "lora_b"],
            Called = pair is null
                ? new Dictionary<string, KerasSlot>()
                : new Dictionary<string, KerasSlot> { ["kernel"] = "weight", ["bias"] = "bias", ["lora_kernel_a"] = "lora_a", ["lora_kernel_b"] = "lora_b" },
        };
    }

    // The pair a Dense layer was given with enable_lora: its rank, and its alpha, which is the rank where Keras leaves it out.
    private static LowRank? PairOf(KerasSettings layer)
    {
        var rank = layer.Setting<int>("lora_rank");
        var alpha = layer.Setting("lora_alpha", (double)rank);

        if (rank < 1 || !double.IsFinite(alpha) || alpha <= 0)
        {
            layer.Refuse($"its low-rank pair is of rank {rank} and alpha {alpha}, and a pair here is of a rank from one and an alpha above nothing.");

            return null;
        }

        return new LowRank(rank, alpha);
    }
}

/// <summary>
/// Keras's <c>BatchNormalization</c>, in Keras's meaning: its momentum the share of the running statistics a batch leaves
/// as they were, and its epsilon.
/// </summary>
internal sealed class BatchNormalizationKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "BatchNormalization";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        RequireLastAxis(layer);

        var momentum = layer.Setting<double>("momentum");
        var epsilon = layer.Setting<double>("epsilon");
        var keeping = Keeping(layer);
        var rules = new Rules(Governing(layer, "gamma", 1, [0]), Governing(layer, "beta", 1, [0]), Held(layer));

        // Keras keeps the scale, then the shift, then the running statistics: the ones it leaves out are not in the file.
        return new KerasLayer(description => rules.Put(description.BatchNorm(momentum, epsilon, keeping)), "linear")
        {
            Slots = [.. Kept(keeping), "running_mean", "running_var"],
        };
    }
}

/// <summary>Keras's <c>LayerNormalization</c>, with its epsilon.</summary>
internal sealed class LayerNormalizationKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "LayerNormalization";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        RequireLastAxis(layer);

        if (layer.Setting("rms_scaling", false))
        {
            layer.Refuse(
                "it scales by rms_scaling, which Keras works out from the variance of the values rather than the root of the mean of their squares, "
                + "and deprecates: a normalisation by the root mean square is Keras's RMSNormalization, which is read.");
        }

        var epsilon = layer.Setting<double>("epsilon");

        var keeping = Keeping(layer);
        var rules = new Rules(Governing(layer, "gamma", 1, [0]), Governing(layer, "beta", 1, [0]), Held(layer));

        return new KerasLayer(description => rules.Put(description.LayerNorm(epsilon, keeping)), "linear") { Slots = [.. Kept(keeping)] };
    }
}

/// <summary>Keras's <c>Dropout</c>, leaving out each value on its own.</summary>
internal sealed class DropoutKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Dropout";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var rate = layer.Setting<double>("rate");

        if (layer.Written("noise_shape") is { } noise)
        {
            layer.Refuse($"it leaves values out by a noise shape, {noise.Quoted()}, and a dropout here leaves out each value on its own.");
        }

        return new KerasLayer(description => description.Dropout(rate), "linear");
    }
}

/// <summary>Keras's <c>Flatten</c>, of images laid out with their channels last.</summary>
internal sealed class FlattenKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Flatten";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        RequireChannelsLast(layer);

        return new KerasLayer(description => description.Flatten(), "linear");
    }
}

/// <summary>Keras's <c>Reshape</c>, stating every length.</summary>
internal sealed class ReshapeKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Reshape";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var target = layer.Setting<int[]?>("target_shape");

        if (target is not null && target.Contains(-1))
        {
            layer.Refuse($"its target shape, {layer.Written("target_shape")!.Quoted()}, leaves a length to be worked out, and a reshape here states every length.");
        }

        return new KerasLayer(description => description.Reshape(new Shape(target!)), "linear");
    }
}

/// <summary>Keras's <c>Activation</c>: an activation alone.</summary>
internal sealed class ActivationKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Activation";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer) => new(null, layer.Setting<string>("activation"));
}

/// <summary>Keras's <c>ReLU</c>, the plain one: no cap, no slope below nothing, no threshold.</summary>
internal sealed class ReluKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "ReLU";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        if (layer.Says("max_value") || layer.Setting("negative_slope", 0.0) != 0 || layer.Setting("threshold", 0.0) != 0)
        {
            layer.Refuse(
                "it caps its values, lets some below nothing through or starts above nothing, and a relu here keeps each value above nothing as it is and makes the rest nothing.");
        }

        return new KerasLayer(null, "relu");
    }
}
