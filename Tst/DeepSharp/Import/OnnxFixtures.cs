// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;
using Google.Protobuf;
using Onnx;

namespace DeepSharp.Tests.Import;

/// <summary>
/// The ONNX graphs PyTorch exported that the ONNX reader's tests read (Fixtures/onnx-fixtures.py made them), what PyTorch
/// answered through each network, and graphs made from them the way a graph comes to say something else: a node changed,
/// added or taken out, a number written otherwise.
/// </summary>
internal static class OnnxFixtures
{
    /// <summary>The folder the fixtures stand in, beside the Keras reader's.</summary>
    public static string Folder => KerasFixtures.Folder;

    /// <summary>What PyTorch answered, and the rows and images it answered for.</summary>
    public static JsonElement Answers { get; } = JsonDocument.Parse(File.ReadAllText(Path.Join(Folder, "onnx-fixtures.json"))).RootElement;

    /// <summary>What TensorFlow answered through the network tf2onnx converted, and the images it answered for.</summary>
    public static JsonElement TensorFlow { get; } = JsonDocument.Parse(File.ReadAllText(Path.Join(Folder, "tf2onnx-fixtures.json"))).RootElement;

    /// <summary>A fixture, opened as the file it is, so the numbers its exporter kept beside it are found there.</summary>
    public static FileStream Open(string file) => File.OpenRead(Path.Join(Folder, file));

    /// <summary>A fixture's graph as its messages, to be changed.</summary>
    public static ModelProto Model(string file)
    {
        using var stream = Open(file);

        return ModelProto.Parser.ParseFrom(stream);
    }

    /// <summary>A fixture whose every number stands in the graph itself, changed as said and written to memory.</summary>
    public static MemoryStream Edited(string file, Action<ModelProto> edit)
    {
        var model = Model(file);
        edit(model);

        return new MemoryStream(model.ToByteArray());
    }

    /// <summary>A graph's node, by its name.</summary>
    public static NodeProto Node(this ModelProto model, string name) => model.Graph.Node.Single(node => node.Name == name);

    /// <summary>A graph's initializer, by its name.</summary>
    public static TensorProto Initializer(this ModelProto model, string name) => model.Graph.Initializer.Single(tensor => tensor.Name == name);

    /// <summary>A node's attribute, by its name.</summary>
    public static AttributeProto Attribute(this NodeProto node, string name) => node.Attribute.Single(attribute => attribute.Name == name);

    /// <summary>Sets a node's whole-number attribute, adding it when the node writes none.</summary>
    public static void SetWhole(this NodeProto node, string name, long value) =>
        node.Set(new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Int, I = value });

    /// <summary>Sets a node's list of whole numbers, adding it when the node writes none.</summary>
    public static void SetWholes(this NodeProto node, string name, params long[] values)
    {
        var attribute = new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Ints };
        attribute.Ints.Add(values);
        node.Set(attribute);
    }

    /// <summary>Sets a node's number attribute, adding it when the node writes none.</summary>
    public static void SetNumber(this NodeProto node, string name, float value) =>
        node.Set(new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Float, F = value });

    /// <summary>Sets a node's text attribute, adding it when the node writes none.</summary>
    public static void SetText(this NodeProto node, string name, string value) =>
        node.Set(new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.String, S = ByteString.CopyFromUtf8(value) });

    /// <summary>Takes a node's attribute out.</summary>
    public static void Unset(this NodeProto node, string name) => node.Attribute.Remove(node.Attribute(name));

    /// <summary>An initializer of single-precision numbers, written as ONNX writes one: little-endian bytes.</summary>
    public static TensorProto Floats(string name, long[] dims, params float[] values)
    {
        var tensor = new TensorProto { Name = name, DataType = (int)TensorProto.Types.DataType.Float };
        tensor.Dims.Add(dims);
        tensor.RawData = ByteString.CopyFrom([.. values.SelectMany(BitConverter.GetBytes)]);

        return tensor;
    }

    /// <summary>An initializer of whole numbers, as ONNX writes a reshape's target.</summary>
    public static TensorProto Wholes(string name, params long[] values)
    {
        var tensor = new TensorProto { Name = name, DataType = (int)TensorProto.Types.DataType.Int64 };
        tensor.Dims.Add(values.Length);
        tensor.RawData = ByteString.CopyFrom([.. values.SelectMany(BitConverter.GetBytes)]);

        return tensor;
    }

    /// <summary>A node of the default domain, taking and giving the values named.</summary>
    public static NodeProto NodeOf(string op, string name, string[] inputs, string output)
    {
        var node = new NodeProto { OpType = op, Name = name };
        node.Input.Add(inputs);
        node.Output.Add(output);

        return node;
    }

    /// <summary>The values of a tensor PyTorch answered, as single-precision numbers.</summary>
    public static float[] Values(JsonElement numbers) => [.. numbers.EnumerateArray().Select(value => (float)value.GetDouble())];

    /// <summary>The images PyTorch was handed, as a batch here: image, row, column, channel.</summary>
    public static Tensor Images(JsonElement images) =>
        Tensor.From(new Shape([.. images.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), Values(images.GetProperty("values")));

    /// <summary>A batch of rows of fourteen features, as PyTorch was handed them.</summary>
    public static Tensor Passengers(JsonElement features)
    {
        var values = Values(features);

        return Tensor.From(new Shape(values.Length / 14, 14), values);
    }

    private static void Set(this NodeProto node, AttributeProto attribute)
    {
        var at = node.Attribute.ToList().FindIndex(each => each.Name == attribute.Name);

        if (at < 0)
        {
            node.Attribute.Add(attribute);
        }
        else
        {
            node.Attribute[at] = attribute;
        }
    }
}

/// <summary>A folder of its own for graphs written in a test, and the files beside them; gone when the test is done.</summary>
internal sealed class GraphFolder : IDisposable
{
    /// <summary>A new, empty folder.</summary>
    public GraphFolder() => Directory.CreateDirectory(Path);

    /// <summary>Where it stands.</summary>
    public string Path { get; } = System.IO.Path.Join(System.IO.Path.GetTempPath(), $"deepsharp-onnx-{Guid.NewGuid():N}");

    /// <summary>Writes a graph into the folder and opens it as its file.</summary>
    public FileStream Written(ModelProto model, string name = "model.onnx")
    {
        File.WriteAllBytes(System.IO.Path.Join(Path, name), model.ToByteArray());

        return File.OpenRead(System.IO.Path.Join(Path, name));
    }

    /// <summary>Writes a file of bytes beside the graphs.</summary>
    public void Beside(string name, byte[] bytes) => File.WriteAllBytes(System.IO.Path.Join(Path, name), bytes);

    /// <inheritdoc />
    public void Dispose() => Directory.Delete(Path, recursive: true);
}
