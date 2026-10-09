// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using Onnx;

namespace DeepSharp.Import.Onnx;

/// <summary>
/// ONNX's <c>ReduceMean</c> or <c>ReduceMax</c> over every place of each channel of a series, an image or a volume — the axes
/// after the batch's and the channels' — and over nothing else: a global pooling, as PyTorch's default exporter writes an
/// adaptive pooling to one place. The axes may be an input the graph holds, from ONNX 18, or an attribute, before it; negative
/// ones are counted from the end. The axes it reduces over are kept as axes of one place, unless it says <c>keepdims</c> 0, and
/// then a row of channels is what it gives.
/// </summary>
/// <param name="name">ONNX's name for the operator.</param>
/// <param name="pooling">The global pooling it is read as.</param>
internal sealed class ReduceKind(string name, GlobalPoolKind pooling) : OnnxKind
{
    /// <inheritdoc />
    public override string Name => name;

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        if (!TakesImages(node, reaching) || node.Axes(numbers) is not { } axes)
        {
            return new OnnxLayer(null, Flow.Images);
        }

        if (axes.Length == 0)
        {
            node.Refuse(
                node.Whole("noop_with_empty_axes", 0) == 0
                    ? "it reduces over every axis when it names none, the batch and the channels among them, and a global pooling here reduces over the places of each channel alone."
                    : "it reduces over nothing when it names no axes, noop_with_empty_axes 1, and a global pooling here reduces over the places of each channel.");

            return new OnnxLayer(null, Flow.Images);
        }

        // A series made of one more axis by an Unsqueeze has four, an image four, a volume five: the batch, the channels, the places.
        var rank = reaching.Axes + (reaching.Lifted ? 3 : 2);
        var places = Enumerable.Range(2, rank - 2).ToArray();

        if (!axes.Select(axis => axis < 0 ? axis + rank : axis).Order().SequenceEqual(places))
        {
            node.Refuse(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"it reduces over axes [{string.Join(", ", axes)}] of a value of {rank} axes, and a global pooling here reduces over every place of each channel and nothing else, axes [{string.Join(", ", places)}]."));

            return new OnnxLayer(null, Flow.Images);
        }

        var keeps = node.Whole("keepdims", 1) != 0;

        return new OnnxLayer(pooling.Pool(reaching.Axes, keeps), keeps ? Flow.Images : Flow.Rows);
    }
}

/// <summary>What the Unsqueeze and the Squeeze round the pooling of a series are.</summary>
internal static class LiftedPooling
{
    extension(GraphProto graph)
    {
        /// <summary>
        /// Whether the node at a place opens the three nodes PyTorch's default exporter writes round the pooling of a series to
        /// one place, and no others: an Unsqueeze making an axis of one place third of four, the ReduceMean or ReduceMax that
        /// keeps the axes it reduces over, straight after it, and a Squeeze taking that axis away, straight after the reduce —
        /// each taking the value the one before it made, and that value nothing else. The exporter writes the Unsqueeze's and the
        /// Squeeze's axes as <c>-2</c>; <c>2</c> says the same.
        /// </summary>
        /// <param name="place">The node's place among the graph's nodes; it is taken to be an Unsqueeze.</param>
        /// <param name="numbers">The values the graph holds, the nodes' axes among them.</param>
        internal bool OpensLiftedPooling(int place, GraphNumbers numbers)
        {
            if (place + 2 >= graph.Node.Count)
            {
                return false;
            }

            // What reading the three says is not the pattern's to say: each node says it when it is read.
            var faults = new List<string>();
            var unsqueeze = new OnnxNode(graph.Node[place], place, faults, []);
            var reduce = new OnnxNode(graph.Node[place + 1], place + 1, faults, []);
            var squeeze = new OnnxNode(graph.Node[place + 2], place + 2, faults, []);

            return reduce.Operator is "ReduceMean" or "ReduceMax"
                && squeeze.Operator == "Squeeze"
                && reduce.Input(0) == unsqueeze.Output
                && squeeze.Input(0) == reduce.Output
                && graph.Takers(unsqueeze.Output) == 1
                && graph.Takers(reduce.Output) == 1
                && reduce.Whole("keepdims", 1) != 0
                && unsqueeze.Axes(numbers) is [2 or -2]
                && squeeze.Axes(numbers) is [2 or -2];
        }

        // How many times a value is taken: as an input of a node, or as what the graph gives.
        private int Takers(string value) => graph.Node.Sum(node => node.Input.Count(input => input == value)) + graph.Output.Count(output => output.Name == value);
    }
}
