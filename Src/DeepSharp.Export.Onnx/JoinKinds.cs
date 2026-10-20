// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Export.Onnx;

/// <summary>The steps added up value by value: ONNX's Add, one for each step beyond the first, in the order the layer adds them in.</summary>
internal sealed class SumKind : ExportKind<Sum>
{
    /// <inheritdoc />
    protected override Value Write(Sum layer, Writing at)
    {
        if (at.Inputs.Any(input => input.Image != at.Input.Image))
        {
            throw new NotSupportedException(
                "A sum adds a row a flatten made of an image to a row it did not, or to one made of another image, and ONNX flattens an image channel by channel where the network flattens it place by place: the two rows do not hold their values in one order.");
        }

        var sum = at.Input.Name;

        for (var next = 1; next < at.Inputs.Count; next++)
        {
            var last = next == at.Inputs.Count - 1;

            sum = at.Graph.Node(at.Node("Add", 7, last ? null : $"Add.{next}"), [sum, at.Inputs[next].Name], last ? at.Path : $"{at.Path}/Add.{next}");
        }

        return at.Keeping(sum);
    }
}

/// <summary>The steps laid side by side along their last axis: ONNX's Concat along the axis after the batch, which is a row's last.</summary>
internal sealed class ConcatenationKind : ExportKind<Concatenation>
{
    /// <inheritdoc />
    protected override Value Write(Concatenation layer, Writing at)
    {
        RequireNoFlattenedImage(at, "laid side by side");

        return at.Giving(at.Graph.Node(
            at.Node("Concat", 4) with { Attributes = node => node.State("axis", 1L) },
            [.. at.Inputs.Select(input => input.Name)],
            at.Path));
    }

    // Rows a flatten made of an image keep their values channel by channel in the graph: laid beside others, the row would hold its values in two orders.
    internal static void RequireNoFlattenedImage(Writing at, string what)
    {
        if (at.Inputs.Any(input => input.Image is not null))
        {
            throw new NotSupportedException(
                $"A row a flatten made of an image is {what}, and ONNX flattens an image channel by channel where the network flattens it place by place: the row would hold its values in two orders.");
        }
    }
}

/// <summary>A block of a row's columns: ONNX's Slice along the axis after the batch, from the tenth operator set where the bounds are inputs.</summary>
internal sealed class ColumnsKind : ExportKind<Columns>
{
    /// <inheritdoc />
    protected override Value Write(Columns layer, Writing at)
    {
        ConcatenationKind.RequireNoFlattenedImage(at, "cut into columns");

        var starts = at.Graph.Wholes(at.Number("starts"), layer.First);
        var ends = at.Graph.Wholes(at.Number("ends"), layer.First + layer.Count);
        var axes = at.Graph.Wholes(at.Number("axes"), 1);

        return at.Giving(at.Graph.Node(at.Node("Slice", 10), [at.Input.Name, starts, ends, axes], at.Path));
    }
}

/// <summary>A stack of steps inside a step: its steps written one after another, each under the name of the step it stands in and then its own.</summary>
internal sealed class StackKind : ExportKind<LayerStack>
{
    /// <inheritdoc />
    protected override Value Write(LayerStack layer, Writing at) => new StackWriting(at.Graph).Write(layer, at.Inputs, at.Path);
}
