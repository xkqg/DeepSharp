// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Export.Onnx;
using DeepSharp.Import.Keras;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Export.Onnx;

/// <summary>
/// The two packages as one road: a model Keras saved, read here and written as an ONNX graph, is answered through by ONNX Runtime as Keras
/// answered it. Each package is held to its own side by its own suite — the reader to Keras's numbers, the writer to the network's — and
/// this is the only place the two meet, so a layer one reads and the other cannot write is found here and nowhere else.
/// </summary>
public class OnnxFromKerasTests
{
    private static readonly string Folder = Path.Join(Repository.Root, "Tst", "DeepSharp", "Import", "Fixtures");

    private static readonly JsonElement Keras = JsonDocument.Parse(File.ReadAllText(Path.Join(Folder, "keras-attention.json"))).RootElement;

    public static TheoryData<string, string> EveryFile
    {
        get
        {
            var files = new TheoryData<string, string>();

            foreach (var name in new[] { "attention", "attention-causal", "attention-block" })
            {
                files.Add(name, "keras");
                files.Add(name, "h5");
            }

            return files;
        }
    }

    [Theory]
    [MemberData(nameof(EveryFile))]
    public void AKerasModelOfAttention_IsWrittenAsOnnx_AndAnsweredThroughAsKerasAnswers(string name, string kind)
    {
        var read = new KerasFile().Read(new MemoryStream(File.ReadAllBytes(Path.Join(Folder, $"keras-{name}.{kind}"))));
        var written = Keras.GetProperty(name).GetProperty("examples");
        var shape = new Shape([.. written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]);
        var rows = Tensor.From(shape, [.. written.GetProperty("values").EnumerateArray().Select(value => (float)value.GetDouble())]);
        var keras = Keras.GetProperty(name).GetProperty("outputs").EnumerateArray().Select(value => (float)value.GetDouble()).ToArray();

        var graph = new OnnxExport(new Shape([.. shape.Axes.ToArray()[1..]])).ToBytes(read.Network, read.Loss);
        var there = Runtime.Answers(graph, Runtime.ChannelsFirst(rows));

        Assert.Equal(keras.Length, there.Length);
        Assert.InRange(keras.Zip(there, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
    }
}
