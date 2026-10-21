// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;
using Tensor = DeepSharp.Tensors.Tensor;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DeepSharp.Tests.Export.Onnx;

/// <summary>ONNX Runtime, asked what a graph answers, on one thread so the order of its arithmetic is the same every time it is asked.</summary>
internal static class Runtime
{
    /// <summary>What the graph answers for the rows, as single-precision numbers.</summary>
    /// <param name="graph">The graph, as the bytes of an ONNX file.</param>
    /// <param name="rows">The rows, laid out as the graph takes them.</param>
    /// <returns>The answers, row-major.</returns>
    public static float[] Answers(byte[] graph, Tensor rows)
    {
        using var options = new SessionOptions { IntraOpNumThreads = 1, InterOpNumThreads = 1, LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
        using var session = new InferenceSession(graph, options);
        var given = new DenseTensor<float>(rows.Values.ToArray(), [.. rows.Shape.Axes.ToArray()]);

        using var results = session.Run([NamedOnnxValue.CreateFromTensor("input", given)]);

        return [.. results.Single().AsEnumerable<float>()];
    }

    /// <summary>
    /// A batch of series, images or volumes with their channels first, as the graph takes them, from the same batch with its channels last,
    /// as the network takes them.
    /// </summary>
    /// <param name="batch">The batch, the channels last.</param>
    public static Tensor ChannelsFirst(Tensor batch)
    {
        var axes = batch.Shape.Axes.ToArray();
        var channels = axes[^1];
        var places = batch.Shape.Count / (axes[0] * channels);
        var moved = new float[batch.Shape.Count];

        for (var example = 0; example < axes[0]; example++)
        {
            for (var place = 0; place < places; place++)
            {
                for (var channel = 0; channel < channels; channel++)
                {
                    moved[(((example * channels) + channel) * places) + place] = batch.Values[(((example * places) + place) * channels) + channel];
                }
            }
        }

        return Tensor.From(new Shape([axes[0], channels, .. axes[1..^1]]), moved);
    }

    /// <summary>A batch of examples of the shape given, numbers worked out from their places, the same on every machine.</summary>
    /// <param name="count">How many examples.</param>
    /// <param name="each">The shape of one example.</param>
    public static Tensor Batch(int count, Shape each) =>
        Tensor.From(new Shape([count, .. each.Axes.ToArray()]), [.. Enumerable.Range(0, count * each.Count).Select(at => (float)(Math.Sin((at * 0.37) + 0.3) * 0.8))]);

    /// <summary>A batch of rows of numbers worked out from their places, the same on every machine.</summary>
    /// <param name="count">How many rows.</param>
    /// <param name="width">How many values in a row.</param>
    public static Tensor Rows(int count, int width) =>
        Tensor.From(new Shape(count, width), [.. Enumerable.Range(0, count * width).Select(at => (float)(Math.Sin((at * 0.7) + 0.3) * 0.8))]);

    /// <summary>
    /// The network with every number replaced by one worked out from its place — so no layer is left at the start that makes it
    /// answer what a freshly lowered one would — and the variance of a normalisation kept above nought.
    /// </summary>
    /// <param name="network">The network.</param>
    /// <returns>The same network.</returns>
    public static LayerStack Filled(LayerStack network)
    {
        var entries = network.Slots().Select((named, place) =>
        {
            var count = named.Slot.Value.Shape.Count;
            var variance = named.Path.EndsWith("running_var", StringComparison.Ordinal);
            var values = Enumerable.Range(0, count).Select(at => (float)(variance ? 0.6 + (0.4 * Math.Abs(Math.Sin(at + place))) : Math.Sin((at * 1.3) + place) * 0.6)).ToArray();

            return new SlotEntry(named.Path, Tensor.From(named.Slot.Value.Shape, values), "a number worked out from its place");
        }).ToArray();

        network.Load(entries);

        return network;
    }
}
