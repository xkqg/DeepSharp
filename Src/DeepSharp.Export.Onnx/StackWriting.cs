// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Export.Onnx;

/// <summary>A stack of steps written one after another, each taking the values of the steps it names.</summary>
/// <param name="graph">The graph being written.</param>
internal sealed class StackWriting(GraphDraft graph)
{
    /// <summary>Writes the stack's steps.</summary>
    /// <param name="stack">The stack.</param>
    /// <param name="inputs">The value the stack is handed, and, where its steps take the mask of the padding, that mask besides.</param>
    /// <param name="inside">The path of the step the stack stands in; nothing for the network's own stack.</param>
    /// <returns>The value of the last step.</returns>
    /// <exception cref="NotSupportedException">A layer is of no kind written.</exception>
    public Value Write(LayerStack stack, IReadOnlyList<Value> inputs, string inside)
    {
        var input = inputs[0];
        var shapes = inputs.Count > 1 ? stack.Shapes(input.Each, inputs[1].Each) : stack.Shapes(input.Each);
        var made = new Dictionary<string, Value>(StringComparer.Ordinal) { [Step.Input] = input };
        var last = input;

        if (inputs.Count > 1)
        {
            made[Step.Mask] = inputs[1];
        }

        for (var at = 0; at < stack.Steps.Count; at++)
        {
            var step = stack.Steps[at];
            var path = inside.Length == 0 ? step.Name : $"{inside}.{step.Name}";
            var each = new Shape([.. shapes[at].Made.Axes[1..]]);

            last = ExportKinds.For(step.Layer).Write(new Writing(graph, step.Layer, path, [.. step.Takes.Select(name => made[name])], each));
            made[step.Name] = last;
        }

        return last;
    }
}
