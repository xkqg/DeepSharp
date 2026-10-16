// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

using DeepSharp.Networks;

namespace DeepSharp.Import.Keras;

/// <summary>The loss a Keras model was compiled with, as a loss here, and whether it took the network's logits.</summary>
/// <param name="Loss">The loss here that is Keras's.</param>
/// <param name="FromLogits">
/// Whether Keras's loss took the network's raw outputs; otherwise it took the chances a last activation gave, which a loss
/// here applies itself.
/// </param>
internal readonly record struct KerasLoss(Loss Loss, bool FromLogits)
{
    private const string Kept = "sum_over_batch_size";

    // The losses read here, by every name Keras writes each under: its own, the short one, and its class's — each made from the
    // settings Keras writes beside it.
    private static readonly Dictionary<string, Func<KerasSettings, Loss>> Named = new(StringComparer.Ordinal)
    {
        ["binary_crossentropy"] = Binary,
        ["BinaryCrossentropy"] = Binary,
        ["categorical_crossentropy"] = Categorical,
        ["CategoricalCrossentropy"] = Categorical,
        ["mean_squared_error"] = _ => new MeanSquaredError(),
        ["mse"] = _ => new MeanSquaredError(),
        ["MeanSquaredError"] = _ => new MeanSquaredError(),
        ["mean_absolute_error"] = _ => new MeanAbsoluteError(),
        ["mae"] = _ => new MeanAbsoluteError(),
        ["MeanAbsoluteError"] = _ => new MeanAbsoluteError(),
        ["huber"] = Bent,
        ["Huber"] = Bent,
        ["poisson"] = Counted,
        ["Poisson"] = Counted,
        ["hinge"] = _ => new Hinge(),
        ["Hinge"] = _ => new Hinge(),
        ["binary_focal_crossentropy"] = Focused,
        ["BinaryFocalCrossentropy"] = Focused,
    };

    /// <summary>
    /// The loss as Keras writes it — its name, the function of that name, or the object with its settings — read; nothing,
    /// with its fault noted, when there is none or it is none read here.
    /// </summary>
    /// <param name="written">The loss as the file writes it; nothing when the model was saved without one.</param>
    /// <param name="where">Where the file writes it.</param>
    /// <param name="faults">The faults of the file.</param>
    public static KerasLoss? Read(JsonElement? written, string where, List<string> faults)
    {
        if (written is not { } loss)
        {
            faults.Add($"{where}: the model was saved without the loss it was trained with, and a network here answers through its loss.");

            return null;
        }

        var said = new KerasSettings(loss, where, faults);
        var className = said.Setting<string?>("class_name", null);
        var name = loss.ValueKind == JsonValueKind.String ? loss.GetString()!
            : className == "function" ? said.Setting<string>("config")
            : className ?? loss.Written();

        if (!Named.TryGetValue(name, out var made))
        {
            faults.Add(
                $"{where}: the loss is {name.Quoted()}, and a Keras model is read here when it was trained with a binary, a binary focal or a categorical cross-entropy, "
                + "a mean squared or a mean absolute error, a Huber, a Poisson or a hinge loss.");

            return null;
        }

        var settings = new KerasSettings(said.Setting("config", default(JsonElement)), where, faults);
        var reduction = settings.Setting("reduction", Kept);

        if (reduction != Kept)
        {
            settings.Refuse($"the loss reduces a batch's losses by '{reduction.Quoted()}', and a loss here takes their mean.");
        }

        return new KerasLoss(made(settings), settings.Setting("from_logits", false));
    }

    // A binary cross-entropy, its labels moved towards a half as Keras's label_smoothing moves them.
    private static Loss Binary(KerasSettings settings) => new BinaryCrossEntropy { Smoothing = Smoothing(settings) };

    // A categorical cross-entropy, its labels spread over the classes as Keras's label_smoothing spreads them.
    private static Loss Categorical(KerasSettings settings) => new CrossEntropy { Smoothing = Smoothing(settings) };

    // Keras's Huber loss, bending at its delta.
    private static Loss Bent(KerasSettings settings) => new Huber(settings.Setting("delta", 1.0));

    // Keras's Poisson loss: on the rates themselves, a ten-millionth added before the logarithm is taken, as Keras adds it.
    private static Loss Counted(KerasSettings settings) => new Poisson(logInput: false) { Epsilon = 1e-7 };

    // Keras's binary focal cross-entropy: its gamma, and its alpha only where it balances the classes; it softens no labels here.
    private static Loss Focused(KerasSettings settings)
    {
        if (settings.Setting("label_smoothing", 0.0) is var smoothing and not 0)
        {
            settings.Refuse(string.Create(CultureInfo.InvariantCulture, $"the focal loss smooths its labels by {smoothing}, and a focal loss here is of labels of nought or one, as torchvision's is."));
        }

        return new Focal(settings.Setting("gamma", 2.0)) { Alpha = settings.Setting("apply_class_balancing", false) ? settings.Setting("alpha", 0.25) : null };
    }

    // The share Keras's label_smoothing softens the labels by: refused, and none taken, where it is no share below one.
    private static double Smoothing(KerasSettings settings)
    {
        var smoothing = settings.Setting("label_smoothing", 0.0);

        if (smoothing is >= 0 and < 1)
        {
            return smoothing;
        }

        settings.Refuse(string.Create(CultureInfo.InvariantCulture, $"the loss smooths its labels by {smoothing}, and labels are softened by a share of nothing or more, below one."));

        return 0;
    }
}
