// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Import.Keras;

/// <summary>Keras's <c>LeakyReLU</c>, letting the share Keras 3 writes as <c>negative_slope</c> through below nothing.</summary>
internal sealed class LeakyReluKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "LeakyReLU";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var slope = layer.Setting("negative_slope", 0.3);

        return new KerasLayer(description => description.LeakyRelu(slope), "linear");
    }
}

/// <summary>
/// Keras's <c>PReLU</c>, a share for each channel: over a row, one for each feature, as Keras keeps it when it shares nothing; over a
/// series, an image or a volume, one for each channel, shared over every place, which Keras writes as <c>shared_axes</c>.
/// </summary>
/// <remarks>
/// Its shares go in as Keras holds them, a length of one for every axis they are shared over. A model whose shares are another
/// number than its channels — one for every value of an image, or one for them all — is refused where its number goes in.
/// </remarks>
internal sealed class PReluKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "PReLU";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        if (layer.Says("alpha_regularizer") || layer.Says("alpha_constraint"))
        {
            layer.Refuse("it penalises or bounds its shares, and a parametric rectifier here learns its shares free.");
        }

        var held = Held(layer);

        return new KerasLayer(description => new Rules(default, default, held).Put(description.PRelu(eachChannel: true)), "linear")
        {
            Slots = [new KerasSlot("weight", 0, []) { Kept = KerasKeeping.Flat }],
        };
    }
}

/// <summary>Keras's <c>ELU</c>, bending towards minus its alpha.</summary>
internal sealed class EluKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "ELU";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var alpha = layer.Setting("alpha", 1.0);

        return new KerasLayer(description => description.Elu(alpha), "linear");
    }
}

/// <summary>Keras's <c>Softmax</c> over the last axis: the softmax as a layer, lifted into a loss that applies it itself when it ends the model.</summary>
internal sealed class SoftmaxKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Softmax";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        if (layer.Written("axis") is { } axis and not ("-1" or "[-1]"))
        {
            layer.Refuse($"it shares out axis {axis.Quoted()}, and a softmax here shares out the last axis.");
        }

        return new KerasLayer(null, "softmax");
    }
}

/// <summary>
/// Keras's <c>GroupNormalization</c>: the channels of each example in groups, measured over every place it holds — and, with a
/// group for every channel, Keras's <c>groups=-1</c>, an instance normalisation.
/// </summary>
internal sealed class GroupNormalizationKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "GroupNormalization";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        RequireLastAxis(layer);

        var groups = layer.Setting("groups", 32);
        var epsilon = layer.Setting("epsilon", 1e-3);
        var keeping = Keeping(layer);
        var rules = new Rules(Governing(layer, "gamma", 1, [0]), Governing(layer, "beta", 1, [0]), Held(layer));

        if (groups is 0 or < -1)
        {
            layer.Refuse(string.Create(CultureInfo.InvariantCulture, $"it puts the channels in {groups} groups, and a group normalisation puts them in one group at least, or one for each channel with -1."));
        }

        return new KerasLayer(
            description => rules.Put(groups == -1 ? description.InstanceNorm(epsilon, keeping) : description.GroupNorm(groups, epsilon, keeping)),
            "linear")
        {
            Slots = [.. Kept(keeping)],
        };
    }
}

/// <summary>Keras's <c>RMSNormalization</c>: each row of the last axis over the root of the mean of its squares, and scaled.</summary>
internal sealed class RmsNormalizationKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "RMSNormalization";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        RequireLastAxis(layer);

        var epsilon = layer.Setting("epsilon", 1e-6);
        var held = Held(layer);

        return new KerasLayer(description => new Rules(default, default, held).Put(description.RmsNorm(epsilon)), "linear") { Slots = ["weight"] };
    }
}

/// <summary>
/// Keras's <c>Embedding</c>, as the first layer over one category's place per example: the network here takes the category's ones
/// and noughts instead, which its vectors are multiplied by, and its vectors are Keras's in the order of the places.
/// </summary>
internal sealed class EmbeddingKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Embedding";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var categories = layer.Setting<int>("input_dim");
        var width = layer.Setting<int>("output_dim");
        var rules = new Rules(Governing(layer, "embeddings", 2, [1]), default, Held(layer));

        if (layer.Setting("mask_zero", false))
        {
            layer.Refuse("it masks the place nought, mask_zero, and an embedding here gives every category a vector of its own, the first one too.");
        }

        if (layer.Written("lora_rank") is { } rank)
        {
            layer.Refuse($"its vectors are low-rank factors, lora_rank {rank}, and an embedding here keeps its vectors whole.");
        }

        return new KerasLayer(description => rules.Put(description.Embedding([.. Enumerable.Range(0, Math.Max(categories, 2))], width)), "linear")
        {
            Slots = ["weight"],
            Takes = new Shape(Math.Max(categories, 1)),
        };
    }
}
