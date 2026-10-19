// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A folder of sounds read as rows, a folder to a class: each recording a cell of the length, the channels and the rate the step says,
/// the samples as the recording holds them, and a recording of another length, channel count or rate refused instead of being cut,
/// mixed or resampled into the one the person never chose.
/// </summary>
public class ReadSoundsTests
{
    // A recording as a WAV file of sixteen-bit samples, the channels interleaved.
    private static byte[] Wav(int rate, int channels, params short[] samples)
    {
        var bytes = new byte[44 + (samples.Length * 2)];

        Encoding.ASCII.GetBytes("RIFF").CopyTo(bytes, 0);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(4), bytes.Length - 8);
        Encoding.ASCII.GetBytes("WAVEfmt ").CopyTo(bytes, 8);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(16), 16);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(20), 1);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(22), (short)channels);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(24), rate);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(28), rate * channels * 2);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(32), (short)(channels * 2));
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(34), 16);
        Encoding.ASCII.GetBytes("data").CopyTo(bytes, 36);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(40), samples.Length * 2);

        for (var at = 0; at < samples.Length; at++)
        {
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(44 + (at * 2)), samples[at]);
        }

        return bytes;
    }

    // Three recordings of four samples at 8000 a second, in two classes.
    private static ScratchFolder Recordings()
    {
        var folder = new ScratchFolder("deepsharp-sounds-");

        Directory.CreateDirectory(folder.At("quiet"));
        Directory.CreateDirectory(folder.At("loud"));
        File.WriteAllBytes(folder.At(Path.Join("loud", "a.wav")), Wav(8000, 1, 32767, -32768, 16384, 0));
        File.WriteAllBytes(folder.At(Path.Join("quiet", "b.wav")), Wav(8000, 1, 100, -100, 50, 0));
        File.WriteAllBytes(folder.At(Path.Join("quiet", "c.wav")), Wav(8000, 1, 0, 0, 1, -1));
        File.WriteAllText(folder.At(Path.Join("quiet", "notes.txt")), "not a recording");

        return folder;
    }

    [Fact]
    public void EachRecording_IsARow_InTheOrderOfItsPath_AsACellOfTheShapeTheStepSays()
    {
        using var folder = Recordings();
        var source = new ReadSoundsStep(folder.Path, new SoundShape(8000, 4, 1)).Open(SourceFolder.WorkingDirectory);
        var rows = source.Rows.Select(row => row.ToArray()).ToArray();

        Assert.Equal(["sound", "label", "loud", "quiet"], source.ColumnNames);
        Assert.Equal(["loud", "quiet", "quiet"], rows.Select(row => row[1]));

        var loud = MediaCell.Parse(rows[0][0]!);
        var quiet = MediaCell.Parse(rows[1][0]!);

        // A sixteen-bit sample v is read as v / 32768, so full scale is one and the cell holds what the recording held.
        Assert.Equal([4, 1], loud.Shape);
        Assert.Equal(32767 / 32768f, loud.Values[0], 6);
        Assert.Equal(-1f, loud.Values[1], 6);
        Assert.Equal(0.5f, loud.Values[2], 6);
        Assert.Equal(100 / 32768f, quiet.Values[0], 6);
    }

    [Fact]
    public void TwoChannels_AreHeldBesideEachOther_SampleByInterleavedSample()
    {
        using var folder = new ScratchFolder("deepsharp-sounds-");

        Directory.CreateDirectory(folder.At("a"));
        File.WriteAllBytes(folder.At(Path.Join("a", "s.wav")), Wav(44100, 2, 1000, -1000, 2000, -2000, 3000, -3000));

        var cell = MediaCell.Parse(new ReadSoundsStep(folder.Path, new SoundShape(44100, 3, 2)).Open(SourceFolder.WorkingDirectory).Rows.Single()[0]!);

        Assert.Equal([3, 2], cell.Shape);
        Assert.Equal([1000 / 32768f, -1000 / 32768f, 2000 / 32768f, -2000 / 32768f, 3000 / 32768f, -3000 / 32768f], cell.Values);
    }

    [Theory]
    [InlineData(8000, 5, 1, "4 samples")]
    [InlineData(8000, 4, 2, "1 channel")]
    [InlineData(16000, 4, 1, "8000 a second")]
    public void ARecordingOfAnotherLengthChannelsOrRate_IsRefused_NamingTheFileAndBothValues_AndNeverCutMixedOrResampled(int rate, int length, int channels, string found)
    {
        using var folder = Recordings();
        var refused = Assert.Throws<FormatException>(() => new ReadSoundsStep(folder.Path, new SoundShape(rate, length, channels)).Open(SourceFolder.WorkingDirectory));

        Assert.Contains("a.wav", refused.Message, StringComparison.Ordinal);
        Assert.Contains(found, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileThatIsNoRecording_IsRefused_NamingTheFile()
    {
        using var folder = new ScratchFolder("deepsharp-sounds-");

        Directory.CreateDirectory(folder.At("a"));
        File.WriteAllText(folder.At(Path.Join("a", "broken.wav")), "this is no recording");

        var refused = Assert.Throws<FormatException>(() => new ReadSoundsStep(folder.Path, new SoundShape(8000, 4, 1)).Open(SourceFolder.WorkingDirectory));

        Assert.Contains("broken.wav", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheShape_NeedsARateALengthAndAChannel()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SoundShape(0, 4, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SoundShape(8000, 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SoundShape(8000, 4, 0));
        Assert.Equal(8000, new SoundShape(8000, 4, 1).Rate);
    }

    [Fact]
    public void TheStep_IsWrittenAndReadBackAsItIs_AndNamesItsVerb()
    {
        var step = new ReadSoundsStep("recordings", new SoundShape(16000, 16000, 1));

        Assert.Equal("read.sounds", step.Verb);
        Assert.Equal(step, ReadSoundsStep.ReadFrom(JsonDocument.Parse(Encoding.UTF8.GetString(step.Canonical())).RootElement));
        Assert.Throws<ArgumentException>(() => new ReadSoundsStep(" ", new SoundShape(16000, 16000, 1)));
        Assert.Throws<ArgumentNullException>(() => step.Open(null!));
        Assert.Throws<ArgumentNullException>(() => SoundsSourceExtensions.ReadSounds(null!, "recordings", new SoundShape(16000, 16000, 1)));
        Assert.Throws<ArgumentNullException>(() => SoundsSourceExtensions.WithSounds(null!));
    }

    [Fact]
    public void TheVerbIsReadBackWhereTheCatalogIsTaught_AndRefusedWhereItIsNot_NamingThePackage()
    {
        var json = Pdd.Create().ReadSounds("recordings", new SoundShape(8000, 4, 1)).Declaration.ToJson();
        var taught = StepCatalog.BuiltIn().WithSounds();

        Assert.Equal("DeepSharp.Pipelines.Sound", StepCatalog.PackageThatBrings("read.sounds"));
        Assert.Contains("DeepSharp.Pipelines.Sound", Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn())).Message, StringComparison.Ordinal);
        Assert.Equal("read.sounds", PipelineDeclaration.FromJson(json, taught).Steps[0].Verb);
    }

    [Fact]
    public void TheRecordings_AreAMediaColumn_ThePipelineSplitsAndHandsOverAsFeatures()
    {
        using var folder = Recordings();
        var prepared = Pdd.Create()
            .ReadSounds(folder.Path, new SoundShape(8000, 4, 1))
            .Declare(schema => schema.Media("sound").Category("label").Integer("loud", "quiet"))
            .SplitAtRandom(0.50, 0.25, 1)
            .Labels(["loud", "quiet"], ones: 1)
            .Drop("label")
            .Build()
            .Run();
        var batch = prepared.Batch(Part.Train, Needs.OneScale);

        Assert.Equal(4, batch.Width);
        Assert.Equal("sound.0", batch.FeatureNames[0]);
        Assert.Equal([4, 1], batch.ExampleShape);
    }
}
