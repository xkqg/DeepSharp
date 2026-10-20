// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using DeepSharp.Tensors;
using DeepSharp.Tests.Tensors;

namespace DeepSharp.Measure;

/// <summary>One product a network makes: what it is for, and the shapes it multiplies.</summary>
/// <param name="Name">Where the product comes from.</param>
/// <param name="Left">The left matrix's shape.</param>
/// <param name="Right">The right matrix's shape.</param>
internal readonly record struct ProductWorkload(string Name, Shape Left, Shape Right)
{
    /// <summary>The multiply-adds the product makes.</summary>
    public long MultiplyAdds => (long)Left[0] * Left[1] * Right[1];
}

/// <summary>
/// The matrix products the light engine is asked for — the forward and backward products of the networks a team trains, from
/// a layer of a hundred and twenty-eight to a product of one value — timed as 0.13.0 released them and as this build makes them.
/// </summary>
internal sealed class ProductWorkloads(Measurement measurement)
{
    private static readonly ProductWorkload[] Shapes =
    [
        new("one value", new(1, 1), new(1, 1)),
        new("a row by a column", new(1, 16), new(16, 1)),
        new("rows by a column of one", new(32, 16), new(16, 1)),
        new("a row against a wide layer", new(1, 16), new(16, 128)),
        new("titanic forward, 14 -> 16", new(32, 14), new(14, 16)),
        new("titanic forward, 16 -> 1", new(32, 16), new(16, 1)),
        new("titanic backward, X turned by G", new(14, 32), new(32, 16)),
        new("histogram forward, 70 -> 128", new(32, 70), new(70, 128)),
        new("histogram forward, 128 -> 70", new(32, 128), new(128, 70)),
        new("histogram backward, X turned by G", new(70, 32), new(32, 128)),
        new("histogram backward, G by W turned", new(32, 70), new(70, 128)),
        new("a convolution's patches by its kernel", new(6272, 9), new(9, 8)),
        new("a layer of 256", new(64, 256), new(256, 256)),
        new("a product of 2^24 multiply-adds", new(256, 256), new(256, 256)),
        new("a product of 2^27 multiply-adds", new(512, 512), new(512, 512)),
    ];

    /// <summary>Times every product, 0.13.0's arithmetic against this build's.</summary>
    /// <returns>A table, one row to a product, in Markdown.</returns>
    public string Table()
    {
        var table = new StringBuilder();
        var backend = new CpuBackend();
        var random = new Random(7);

        table.AppendLine("| product | shape | multiply-adds | 0.13.0 ns | this build ns | speed-up | 0.13.0 bytes | this build bytes | same bits |");
        table.AppendLine("|---|---|---:|---:|---:|---:|---:|---:|---|");

        foreach (var workload in Shapes)
        {
            var left = Filled(workload.Left, random);
            var right = Filled(workload.Right, random);
            var comparison = measurement.Between(
                () => ReleasedCpuArithmetic.MatMul(left.Values, right.Values, left.Shape, right.Shape),
                () => backend.MatMul(left, right));
            var same = ReleasedCpuArithmetic.MatMul(left.Values, right.Values, left.Shape, right.Shape)
                .AsSpan().SequenceEqual(backend.MatMul(left, right).Values);

            table.AppendLine(string.Create(
                CultureInfo.InvariantCulture,
                $"| {workload.Name} | {workload.Left}·{workload.Right} | {workload.MultiplyAdds} | {comparison.Before.Nanoseconds:F0} | {comparison.After.Nanoseconds:F0} | {comparison.Speedup:F2} | {comparison.Before.Bytes} | {comparison.After.Bytes} | {(same ? "yes" : "NO")} |"));
        }

        return table.ToString();
    }

    private static Tensor Filled(Shape shape, Random random) =>
        Tensor.From(shape, [.. Enumerable.Range(0, shape.Count).Select(_ => (float)((random.NextDouble() * 2) - 1))]);
}
