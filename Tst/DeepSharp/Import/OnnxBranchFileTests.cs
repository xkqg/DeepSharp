// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Google.Protobuf;
using Onnx;

namespace DeepSharp.Tests.Import;

/// <summary>
/// A graph whose paths meet — a value added back after the layers that worked on it, laid side by side with another, cut
/// apart by columns — read as the steps a network of steps is: each node a step that takes the value it names, so the network
/// answers as the graph's own arithmetic does.
/// </summary>
public class OnnxBranchFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static Tensor Rows(int count, int width) =>
        Tensor.From(new Shape(count, width), [.. Enumerable.Range(0, count * width).Select(at => (float)(Math.Sin(at * 0.7) / 2))]);

    private static float[] Values(LayerStack stack, string path) => stack.Slots().Single(slot => slot.Path == path).Slot.Value.Values.ToArray();

    [Fact]
    public void ALayerWhoseInputIsAddedBackAfterIt_IsReadAsAResidualStep_AndAnswersAsItsGraphDoes()
    {
        using var graph = OnnxFixtures.Open("onnx-branch.onnx");
        var saved = new OnnxFile(new BinaryCrossEntropy()).Read(graph);
        var stack = Assert.IsType<LayerStack>(saved.Network);

        Assert.Equal(["0", "1", "2", "3"], stack.Steps.Select(step => step.Name));
        Assert.Equal(["input", "0", "1", "2"], stack.Steps.Select(step => step.Takes[0]));
        Assert.Equal(["1", "input"], stack.Steps[2].Takes);
        Assert.IsType<Sum>(stack.Steps[2].Layer);

        var rows = Rows(5, 14);
        var inner = Values(stack, "0.weight");
        var innerBias = Values(stack, "0.bias");
        var outer = Values(stack, "3.weight");
        var outerBias = Values(stack, "3.bias")[0];
        var answers = saved.Network.Predict(rows, saved.Loss, Engine).Values.ToArray();

        for (var row = 0; row < 5; row++)
        {
            var logit = (double)outerBias;

            for (var unit = 0; unit < 14; unit++)
            {
                var hidden = (double)innerBias[unit];

                for (var input = 0; input < 14; input++)
                {
                    hidden += rows.Values[(row * 14) + input] * inner[(input * 14) + unit];
                }

                logit += (Math.Max(hidden, 0) + rows.Values[(row * 14) + unit]) * outer[unit];
            }

            Assert.Equal(1 / (1 + Math.Exp(-logit)), answers[row], 5);
        }
    }

    [Fact]
    public void ABranchRead_IsWrittenToItsFileAsTheEighthVersion_AndReadBackTheSame()
    {
        using var graph = OnnxFixtures.Open("onnx-branch.onnx");
        var saved = new OnnxFile(new BinaryCrossEntropy()).Read(graph);
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, saved.Network, saved.Loss);
            writer.WriteEndObject();
        }

        var written = Encoding.UTF8.GetString(stream.ToArray());
        var rebuilt = NetworkDocument.ReadNetwork(written, "network", NetworkCatalog.BuiltIn());
        var rows = Rows(3, 14);

        Assert.Equal(NetworkDocument.EighthVersion, JsonDocument.Parse(written).RootElement.GetProperty("network").GetProperty("version").GetInt32());
        Assert.Equal(
            saved.Network.Predict(rows, saved.Loss, Engine).Values.ToArray(),
            rebuilt.Network.Predict(rows, rebuilt.Loss, Engine).Values.ToArray());
    }

    // A layer's numbers worked out from their places, written outputs by inputs as a Gemm with transB takes them.
    private sealed record Linear(string Name, int Inputs, int Outputs, int Seed)
    {
        public float[] Weights => [.. Enumerable.Range(0, Inputs * Outputs).Select(at => ((((7 * (at / Inputs)) + (3 * (at % Inputs)) + Seed) % 11) - 5) / 20f)];

        public float[] Bias => [.. Enumerable.Range(0, Outputs).Select(at => (((((5 * at) + Seed) % 7) - 3) / 40f) + 0.0053f)];

        public TensorProto[] Initializers => [OnnxFixtures.Floats(Name + ".weight", [Outputs, Inputs], Weights), OnnxFixtures.Floats(Name + ".bias", [Outputs], Bias)];

        public NodeProto Node(string input, string output)
        {
            var node = OnnxFixtures.NodeOf("Gemm", Name, [input, Name + ".weight", Name + ".bias"], output);
            node.SetWhole("transB", 1);

            return node;
        }

        public double[][] Apply(double[][] rows) =>
            [.. rows.Select(row => Enumerable.Range(0, Outputs).Select(unit => Enumerable.Range(0, Inputs).Sum(input => row[input] * Weights[(unit * Inputs) + input]) + Bias[unit]).ToArray())];
    }

    private static double[][] RowsOf(Tensor rows) =>
        [.. Enumerable.Range(0, rows.Shape[0]).Select(row => Enumerable.Range(0, rows.Shape[1]).Select(column => (double)rows.Values[(row * rows.Shape[1]) + column]).ToArray())];

    private static double[][] Relu(double[][] rows) => [.. rows.Select(row => row.Select(value => Math.Max(value, 0)).ToArray())];

    private static MemoryStream Graph(int width, string output, IEnumerable<TensorProto> numbers, params NodeProto[] nodes)
    {
        var model = new ModelProto { IrVersion = 8 };
        model.OpsetImport.Add(new OperatorSetIdProto { Version = 17 });
        model.Graph = new GraphProto { Name = "branches" };

        var shape = new TensorShapeProto();
        shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimParam = "batch" });
        shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = width });
        model.Graph.Input.Add(new ValueInfoProto { Name = "x", Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = (int)TensorProto.Types.DataType.Float, Shape = shape } } });
        model.Graph.Output.Add(new ValueInfoProto { Name = output });
        model.Graph.Initializer.Add(numbers);
        model.Graph.Node.Add(nodes);

        return new MemoryStream(model.ToByteArray());
    }

    private static NodeProto Concat(string name, string[] inputs, string output, long axis = 1)
    {
        var node = OnnxFixtures.NodeOf("Concat", name, inputs, output);
        node.SetWhole("axis", axis);

        return node;
    }

    private static void AnswersAs(Stream graph, Tensor rows, double[][] expected)
    {
        var saved = new OnnxFile(new MeanSquaredError()).Read(graph);
        var answers = saved.Network.Predict(rows, saved.Loss, Engine).Values.ToArray();
        var width = expected[0].Length;

        for (var row = 0; row < expected.Length; row++)
        {
            for (var column = 0; column < width; column++)
            {
                Assert.Equal(expected[row][column], answers[(row * width) + column], 5);
            }
        }
    }

    [Fact]
    public void AValueLaidBesideTheInput_ByAConcat_IsAConcatenationOfTwoSteps_AndAnswersAsItsGraphDoes()
    {
        var inner = new Linear("inner", 6, 4, 1);
        var outer = new Linear("outer", 10, 2, 4);
        var rows = Rows(4, 6);
        var graph = Graph(
            6, "y", [.. inner.Initializers, .. outer.Initializers],
            inner.Node("x", "h"), OnnxFixtures.NodeOf("Relu", "act", ["h"], "r"), Concat("join", ["x", "r"], "c"), outer.Node("c", "y"));
        var hidden = Relu(inner.Apply(RowsOf(rows)));

        AnswersAs(graph, rows, outer.Apply([.. RowsOf(rows).Zip(hidden, (row, made) => row.Concat(made).ToArray())]));
    }

    [Fact]
    public void TwoInputsHandedAsTheColumnsOfOneRow_AreCutApartBySplit_AndEachReachesItsOwnLayer()
    {
        var left = new Linear("left", 2, 3, 2);
        var right = new Linear("right", 4, 3, 5);
        var rows = Rows(4, 6);
        var split = OnnxFixtures.NodeOf("Split", "cut", ["x", "sizes"], "a");
        split.Output.Add("b");
        split.SetWhole("axis", 1);
        var graph = Graph(
            6, "y", [.. left.Initializers, .. right.Initializers, OnnxFixtures.Wholes("sizes", 2, 4)],
            split, left.Node("a", "la"), right.Node("b", "lb"), OnnxFixtures.NodeOf("Add", "meet", ["la", "lb"], "y"));
        var all = RowsOf(rows);
        var a = left.Apply([.. all.Select(row => row[..2])]);
        var b = right.Apply([.. all.Select(row => row[2..])]);

        AnswersAs(graph, rows, [.. a.Zip(b, (one, other) => one.Zip(other, (x, y) => x + y).ToArray())]);
    }

    [Fact]
    public void AColumnsSlice_ReadsTheColumnsItSays_AndASliceToTheEndOfAnInputRowIsThroughToTheWidthTheGraphStates()
    {
        var layer = new Linear("tail", 4, 2, 3);
        var rows = Rows(3, 6);
        var slice = OnnxFixtures.NodeOf("Slice", "tail", ["x", "starts", "ends", "axes"], "t");
        var graph = Graph(6, "y", [.. layer.Initializers, OnnxFixtures.Wholes("starts", 2), OnnxFixtures.Wholes("ends", long.MaxValue), OnnxFixtures.Wholes("axes", 1)], slice, layer.Node("t", "y"));

        AnswersAs(graph, rows, layer.Apply([.. RowsOf(rows).Select(row => row[2..])]));
    }

    [Fact]
    public void ASliceOfTheBatchAxis_OrOfSomeColumnsSteppedOver_IsRefusedAtItsNode()
    {
        var layer = new Linear("tail", 2, 2, 3);
        var slice = OnnxFixtures.NodeOf("Slice", "cut", ["x", "starts", "ends", "axes"], "t");
        var onTheBatch = Graph(6, "y", [.. layer.Initializers, OnnxFixtures.Wholes("starts", 0), OnnxFixtures.Wholes("ends", 2), OnnxFixtures.Wholes("axes", 0)], slice, layer.Node("t", "y"));
        var stepped = OnnxFixtures.NodeOf("Slice", "cut", ["x", "starts", "ends", "axes", "steps"], "t");
        var skipping = Graph(
            6, "y", [.. layer.Initializers, OnnxFixtures.Wholes("starts", 0), OnnxFixtures.Wholes("ends", 4), OnnxFixtures.Wholes("axes", 1), OnnxFixtures.Wholes("steps", 2)],
            stepped, layer.Node("t", "y"));

        Assert.Contains("node 'cut' (Slice): it cuts axis 0", Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(onTheBatch)).Message, StringComparison.Ordinal);
        Assert.Contains("node 'cut' (Slice): it steps over", Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(skipping)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AConcatOfTheBatchAxis_OrOfAValueNoNodeMade_IsRefusedAtItsNode()
    {
        var layer = new Linear("a", 6, 2, 1);
        var onTheBatch = Graph(6, "y", layer.Initializers, layer.Node("x", "h"), Concat("join", ["x", "h"], "c", axis: 0));
        var unmade = Graph(6, "y", layer.Initializers, layer.Node("x", "h"), Concat("join", ["x", "ghost"], "y"));

        Assert.Contains("node 'join' (Concat): it lays its values side by side along axis 0", Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(onTheBatch)).Message, StringComparison.Ordinal);
        Assert.Contains("'ghost', which is neither the input nor a value a node before it made", Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(unmade)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANodeWhoseWorkNothingTakes_IsRefusedAsAStepTakenByNoLaterStep()
    {
        var one = new Linear("a", 6, 2, 1);
        var two = new Linear("b", 6, 2, 2);
        var graph = Graph(6, "y", [.. one.Initializers, .. two.Initializers], one.Node("x", "unused"), two.Node("x", "y"));
        var refused = Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(graph));

        Assert.Contains("taken by no later step", refused.Message, StringComparison.Ordinal);
    }
    [Fact]
    public void ASplitIntoEqualParts_OrBySizesTheNodeStatesAsAnAttribute_CutsTheColumnsAsItSays()
    {
        var rows = Rows(4, 6);
        var all = RowsOf(rows);
        var equal = OnnxFixtures.NodeOf("Split", "cut", ["x"], "a");
        equal.Output.Add("b");
        equal.SetWhole("axis", 1);
        var left = new Linear("left", 3, 3, 2);
        var right = new Linear("right", 3, 3, 5);
        var evenly = Graph(6, "y", [.. left.Initializers, .. right.Initializers], equal, left.Node("a", "la"), right.Node("b", "lb"), OnnxFixtures.NodeOf("Add", "meet", ["la", "lb"], "y"));

        AnswersAs(
            evenly,
            rows,
            [.. left.Apply([.. all.Select(row => row[..3])]).Zip(right.Apply([.. all.Select(row => row[3..])]), (one, other) => one.Zip(other, (x, y) => x + y).ToArray())]);

        var stated = OnnxFixtures.NodeOf("Split", "cut", ["x"], "a");
        stated.Output.Add("b");
        stated.SetWhole("axis", 1);
        stated.SetWholes("split", 2, 4);
        var narrow = new Linear("left", 2, 3, 2);
        var wide = new Linear("right", 4, 3, 5);
        var bySizes = Graph(6, "y", [.. narrow.Initializers, .. wide.Initializers], stated, narrow.Node("a", "la"), wide.Node("b", "lb"), OnnxFixtures.NodeOf("Add", "meet", ["la", "lb"], "y"));

        AnswersAs(
            bySizes,
            rows,
            [.. narrow.Apply([.. all.Select(row => row[..2])]).Zip(wide.Apply([.. all.Select(row => row[2..])]), (one, other) => one.Zip(other, (x, y) => x + y).ToArray())]);
    }

    [Fact]
    public void ASliceThatStatesItsColumnsAsAttributes_ReadsTheColumnsItSays()
    {
        var layer = new Linear("mid", 3, 2, 3);
        var rows = Rows(3, 6);
        var slice = OnnxFixtures.NodeOf("Slice", "mid", ["x"], "t");
        slice.SetWholes("starts", 1);
        slice.SetWholes("ends", 4);
        slice.SetWholes("axes", 1);
        var graph = Graph(6, "y", layer.Initializers, slice, layer.Node("t", "y"));

        AnswersAs(graph, rows, layer.Apply([.. RowsOf(rows).Select(row => row[1..4])]));
    }

    [Theory]
    [InlineData("split sizes of 2 and 5", "node 'cut' (Split): it cuts into [2, 5]")]
    [InlineData("split sizes with a nought", "node 'cut' (Split): its sizes")]
    [InlineData("a slice of two runs", "node 'cut' (Slice): it cuts one run of columns")]
    [InlineData("a slice of no column", "node 'cut' (Slice): it cuts columns 4 to 2, and that is no column")]
    public void ACutThatNoNetworkHereMakes_IsRefusedAtItsNode(string cut, string refusal)
    {
        var layer = new Linear("tail", 2, 2, 3);
        var slice = OnnxFixtures.NodeOf("Slice", "cut", ["x", "starts", "ends", "axes"], "t");
        var split = OnnxFixtures.NodeOf("Split", "cut", ["x", "sizes"], "t");

        split.Output.Add("u");
        split.SetWhole("axis", 1);

        var graph = cut switch
        {
            "split sizes of 2 and 5" => Graph(6, "y", [.. layer.Initializers, OnnxFixtures.Wholes("sizes", 2, 5)], split, layer.Node("t", "y")),
            "split sizes with a nought" => Graph(6, "y", [.. layer.Initializers, OnnxFixtures.Wholes("sizes", 0, 6)], split, layer.Node("t", "y")),
            "a slice of two runs" => Graph(6, "y", [.. layer.Initializers, OnnxFixtures.Wholes("starts", 0, 1), OnnxFixtures.Wholes("ends", 2, 3), OnnxFixtures.Wholes("axes", 1)], slice, layer.Node("t", "y")),
            _ => Graph(6, "y", [.. layer.Initializers, OnnxFixtures.Wholes("starts", 4), OnnxFixtures.Wholes("ends", 2), OnnxFixtures.Wholes("axes", 1)], slice, layer.Node("t", "y")),
        };

        Assert.Contains(refusal, Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(graph)).Message, StringComparison.Ordinal);
    }
}
