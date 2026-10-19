// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A folder of pictures read as rows, a folder to a class: each picture a cell of the size and the channels the step says, the pixels
/// as the picture holds them, and a picture of another size refused instead of being resized into one the person never chose.
/// </summary>
public class ReadImagesTests
{
    private readonly record struct Pixel(byte R, byte G, byte B);

    // A picture as an uncompressed 24-bit BMP: rows bottom-up, blue green red, each row padded to four bytes.
    private static byte[] Bmp(int width, int height, Func<int, int, Pixel> pixel)
    {
        var stride = ((width * 3) + 3) / 4 * 4;
        var bytes = new byte[54 + (stride * height)];

        bytes[0] = (byte)'B';
        bytes[1] = (byte)'M';
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(2), bytes.Length);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(10), 54);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(14), 40);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(18), width);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(22), height);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(26), 1);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(28), 24);

        for (var row = 0; row < height; row++)
        {
            for (var column = 0; column < width; column++)
            {
                var (r, g, b) = pixel(height - 1 - row, column);
                var at = 54 + (row * stride) + (column * 3);

                bytes[at] = b;
                bytes[at + 1] = g;
                bytes[at + 2] = r;
            }
        }

        return bytes;
    }

    // A picture as a PNG of eight-bit red green blue.
    private static byte[] Png(int width, int height, Func<int, int, Pixel> pixel)
    {
        var raw = new List<byte>();

        for (var row = 0; row < height; row++)
        {
            raw.Add(0);

            for (var column = 0; column < width; column++)
            {
                var (r, g, b) = pixel(row, column);

                raw.AddRange([r, g, b]);
            }
        }

        using var packed = new MemoryStream();

        using (var zlib = new ZLibStream(packed, CompressionLevel.Optimal, leaveOpen: true))
        {
            zlib.Write(raw.ToArray());
        }

        var header = new byte[13];

        BinaryPrimitives.WriteInt32BigEndian(header, width);
        BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), height);
        header[8] = 8;
        header[9] = 2;

        using var png = new MemoryStream();

        png.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        Chunk(png, "IHDR", header);
        Chunk(png, "IDAT", packed.ToArray());
        Chunk(png, "IEND", []);

        return png.ToArray();
    }

    private static void Chunk(MemoryStream into, string kind, byte[] data)
    {
        var length = new byte[4];
        var body = Encoding.ASCII.GetBytes(kind).Concat(data).ToArray();
        var crc = 0xFFFFFFFFu;

        foreach (var value in body)
        {
            crc ^= value;

            for (var bit = 0; bit < 8; bit++)
            {
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
            }
        }

        BinaryPrimitives.WriteInt32BigEndian(length, data.Length);
        into.Write(length);
        into.Write(body);
        BinaryPrimitives.WriteUInt32BigEndian(length, ~crc);
        into.Write(length);
    }

    // Three pictures of 4 wide and 2 high in two classes: a red one, a grey one and a blue one.
    private static ScratchFolder Pictures()
    {
        var folder = new ScratchFolder("deepsharp-images-");

        Directory.CreateDirectory(folder.At("cold"));
        Directory.CreateDirectory(folder.At("warm"));
        File.WriteAllBytes(folder.At(Path.Join("warm", "red.png")), Png(4, 2, (row, column) => new Pixel((byte)(200 + row), (byte)column, 10)));
        File.WriteAllBytes(folder.At(Path.Join("cold", "blue.bmp")), Bmp(4, 2, (row, column) => new Pixel(10, (byte)column, (byte)(200 + row))));
        File.WriteAllBytes(folder.At(Path.Join("cold", "grey.bmp")), Bmp(4, 2, (_, column) => new Pixel((byte)(40 * column), (byte)(40 * column), (byte)(40 * column))));
        File.WriteAllText(folder.At(Path.Join("cold", "notes.txt")), "not a picture");

        return folder;
    }

    [Fact]
    public void EachPicture_IsARow_InTheOrderOfItsPath_AsACellOfTheShapeTheStepSays()
    {
        using var folder = Pictures();
        var source = new ReadImagesStep(folder.Path, new ImageShape(4, 2, 3)).Open(SourceFolder.WorkingDirectory);
        var rows = source.Rows.Select(row => row.ToArray()).ToArray();

        Assert.Equal(["image", "label", "cold", "warm"], source.ColumnNames);
        Assert.Equal(["cold", "cold", "warm"], rows.Select(row => row[1]));

        var blue = MediaCell.Parse(rows[0][0]!);
        var red = MediaCell.Parse(rows[2][0]!);

        Assert.Equal([2, 4, 3], blue.Shape);
        Assert.Equal(24, blue.Values.Length);

        // Rows top to bottom, each pixel red green blue, a byte v read as v / 127.5 - 1.
        Assert.Equal((10 / 127.5f) - 1, blue.Values[0], 4);
        Assert.Equal((200 / 127.5f) - 1, blue.Values[2], 4);
        Assert.Equal((201 / 127.5f) - 1, blue.Values[(4 * 3) + 2], 4);
        Assert.Equal((200 / 127.5f) - 1, red.Values[0], 4);
        Assert.Equal((3 / 127.5f) - 1, red.Values[(3 * 3) + 1], 4);
    }

    [Fact]
    public void OneChannel_ReadsAGreyPicture_AsTheGreyItHolds()
    {
        using var folder = Pictures();
        var source = new ReadImagesStep(folder.Path, new ImageShape(4, 2, 1)).Open(SourceFolder.WorkingDirectory);
        var grey = MediaCell.Parse(source.Rows.Select(row => row.ToArray()).ElementAt(1)[0]!);

        Assert.Equal([2, 4, 1], grey.Shape);
        Assert.Equal(-1, grey.Values[0], 4);
        Assert.Equal((40 / 127.5f) - 1, grey.Values[1], 4);
        Assert.Equal((120 / 127.5f) - 1, grey.Values[3], 4);
    }

    [Fact]
    public void APictureOfAnotherSize_IsRefused_NamingTheFileAndBothSizes_AndNeverResized()
    {
        using var folder = Pictures();
        var refused = Assert.Throws<FormatException>(() => new ReadImagesStep(folder.Path, new ImageShape(8, 8, 3)).Open(SourceFolder.WorkingDirectory));

        Assert.Contains("blue.bmp", refused.Message, StringComparison.Ordinal);
        Assert.Contains("4 by 2", refused.Message, StringComparison.Ordinal);
        Assert.Contains("8 by 8", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileThatIsNoPicture_IsRefused_NamingTheFile()
    {
        using var folder = new ScratchFolder("deepsharp-images-");

        Directory.CreateDirectory(folder.At("a"));
        File.WriteAllText(folder.At(Path.Join("a", "broken.png")), "this is no picture");

        var refused = Assert.Throws<FormatException>(() => new ReadImagesStep(folder.Path, new ImageShape(4, 2, 3)).Open(SourceFolder.WorkingDirectory));

        Assert.Contains("broken.png", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheShape_NeedsASizeAndOneThreeOrFourChannels()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ImageShape(0, 2, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ImageShape(4, 0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ImageShape(4, 2, 2));
        Assert.Equal(1, new ImageShape(4, 2, 1).Channels);
        Assert.Equal(4, new ImageShape(4, 2, 4).Channels);
    }

    [Fact]
    public void TheStep_IsWrittenAndReadBackAsItIs_AndNamesItsVerb()
    {
        var step = new ReadImagesStep("pictures", new ImageShape(28, 28, 1));

        Assert.Equal("read.images", step.Verb);
        Assert.Equal(step, ReadImagesStep.ReadFrom(JsonDocument.Parse(Encoding.UTF8.GetString(step.Canonical())).RootElement));
        Assert.Throws<ArgumentException>(() => new ReadImagesStep(" ", new ImageShape(28, 28, 1)));
        Assert.Throws<ArgumentNullException>(() => step.Open(null!));
        Assert.Throws<ArgumentNullException>(() => ImagesSourceExtensions.ReadImages(null!, "pictures", new ImageShape(28, 28, 1)));
        Assert.Throws<ArgumentNullException>(() => ImagesSourceExtensions.WithImages(null!));
    }

    [Fact]
    public void TheVerbIsReadBackWhereTheCatalogIsTaught_AndRefusedWhereItIsNot_NamingThePackage()
    {
        var json = Pdd.Create().ReadImages("pictures", new ImageShape(4, 2, 3)).Declaration.ToJson();
        var taught = StepCatalog.BuiltIn().WithImages();

        Assert.Equal("DeepSharp.Pipelines.Images", StepCatalog.PackageThatBrings("read.images"));
        Assert.Contains("DeepSharp.Pipelines.Images", Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn())).Message, StringComparison.Ordinal);
        Assert.Equal("read.images", PipelineDeclaration.FromJson(json, taught).Steps[0].Verb);
    }

    [Fact]
    public void ThePictures_AreAMediaColumn_ThePipelineSplitsAndHandsOverAsFeatures()
    {
        using var folder = Pictures();
        var prepared = Pdd.Create()
            .ReadImages(folder.Path, new ImageShape(4, 2, 3))
            .Declare(schema => schema.Media("image").Category("label").Integer("cold", "warm"))
            .SplitAtRandom(0.50, 0.25, 1)
            .Labels(["cold", "warm"], ones: 1)
            .Drop("label")
            .Build()
            .Run();
        var batch = prepared.Batch(Part.Train, Needs.OneScale);

        Assert.Equal(24, batch.Width);
        Assert.Equal("image.0", batch.FeatureNames[0]);
        Assert.Equal([2, 4, 3], batch.ExampleShape);
    }

    [Fact]
    public void FourChannels_ReadAPictureWithHowMuchShowsThrough_WhichAPictureWithoutOneShowsWhole()
    {
        using var folder = Pictures();
        var source = new ReadImagesStep(folder.Path, new ImageShape(4, 2, 4)).Open(SourceFolder.WorkingDirectory);
        var cell = MediaCell.Parse(source.Rows.First()[0]!);

        Assert.Equal([2, 4, 4], cell.Shape);
        Assert.Equal(1f, cell.Values[3], 4);
    }

}
