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

    // The losses read here, by every name Keras writes each under: its own, the short one, and its class's.
    private static readonly Dictionary<string, Func<Loss>> Named = new(StringComparer.Ordinal)
    {
        ["binary_crossentropy"] = () => new BinaryCrossEntropy(),
        ["BinaryCrossentropy"] = () => new BinaryCrossEntropy(),
        ["categorical_crossentropy"] = () => new CrossEntropy(),
        ["CategoricalCrossentropy"] = () => new CrossEntropy(),
        ["mean_squared_error"] = () => new MeanSquaredError(),
        ["mse"] = () => new MeanSquaredError(),
        ["MeanSquaredError"] = () => new MeanSquaredError(),
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
            faults.Add($"{where}: the loss is {name.Quoted()}, and a Keras model is read here when it was trained with a binary or a categorical cross-entropy or a mean squared error.");

            return null;
        }

        var settings = new KerasSettings(said.Setting("config", default(JsonElement)), where, faults);
        var smoothing = settings.Setting("label_smoothing", 0.0);
        var reduction = settings.Setting("reduction", Kept);

        if (smoothing != 0)
        {
            settings.Refuse(string.Create(CultureInfo.InvariantCulture, $"the loss smooths its labels by {smoothing}, and no loss here smooths them."));
        }

        if (reduction != Kept)
        {
            settings.Refuse($"the loss reduces a batch's losses by '{reduction.Quoted()}', and a loss here takes their mean.");
        }

        return new KerasLoss(made(), settings.Setting("from_logits", false));
    }
}
