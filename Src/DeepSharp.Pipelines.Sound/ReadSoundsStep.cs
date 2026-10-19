// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using NAudio.Wave;

namespace DeepSharp.Pipelines;

/// <summary>
/// The length, the channels and the rate every recording of a folder is read as.
/// </summary>
/// <param name="Rate">How many samples a second, in each channel; one or more.</param>
/// <param name="Length">How many samples each channel holds; one or more.</param>
/// <param name="Channels">How many channels: one for mono, two for stereo; one or more.</param>
public readonly record struct SoundShape(int Rate, int Length, int Channels)
{
    /// <summary>How many samples a second, in each channel.</summary>
    public int Rate { get; } = Rate >= 1
        ? Rate
        : throw new ArgumentOutOfRangeException(nameof(Rate), Rate, "A recording has at least one sample a second.");

    /// <summary>How many samples each channel holds.</summary>
    public int Length { get; } = Length >= 1
        ? Length
        : throw new ArgumentOutOfRangeException(nameof(Length), Length, "A recording is at least one sample long.");

    /// <summary>How many channels.</summary>
    public int Channels { get; } = Channels >= 1
        ? Channels
        : throw new ArgumentOutOfRangeException(nameof(Channels), Channels, "A recording has at least one channel.");
}

/// <summary>
/// Reads the rows from a folder of recordings, a folder to a class: a recording a row, the folder it stands in the answer.
/// </summary>
/// <remarks>
/// The columns are <c>sound</c> — the samples as one media cell of the shape the step says, a row for each sample in turn and a column
/// for each channel — and <c>label</c>, the name of the folder, and then a column of nought and one for each folder, named for it, which a
/// learner is handed the answer by. A recording of another length, another number of channels or another rate is refused, naming it:
/// nothing is cut, padded, mixed down or resampled into a shape the person did not choose. WAV files only; whatever sample format the file
/// holds, a sample is kept as sixteen bits. Nothing is opened until the pipeline runs.
/// </remarks>
public sealed record ReadSoundsStep : IPipelineStep<ReadSoundsStep>, IOpensRows, IDescribesColumns
{
    /// <summary>The name of the column that holds each recording's samples.</summary>
    public const string SoundColumn = "sound";

    private static readonly string[] Extensions = [".wav"];

    private static readonly FilePathParameter PathKey = new(
        "path", "The folder the recordings will be in, when the pipeline runs: a folder to each class, files of .wav in each.", "recordings");

    private static readonly WholeNumberParameter RateKey = new("rate", "How many samples a second every recording has; one that has another rate is refused, never resampled.", 16000) { AtLeast = 1 };

    private static readonly WholeNumberParameter LengthKey = new("length", "How many samples each channel of every recording holds; a longer or shorter one is refused, never cut or padded.", 16000) { AtLeast = 1 };

    private static readonly WholeNumberParameter ChannelsKey = new("channels", "How many channels every recording has; one that has another number is refused, never mixed.", 1) { AtLeast = 1 };

    /// <summary>Declares that the rows come from the recordings of this folder.</summary>
    /// <param name="path">Where the folder will be, when the pipeline runs.</param>
    /// <param name="shape">The length, the channels and the rate every recording is read as.</param>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    public ReadSoundsStep(string path, SoundShape shape)
    {
        Path = PathKey.Require(path);
        Shape = shape;
    }

    /// <summary>Where the folder will be, when the pipeline runs.</summary>
    public string Path { get; }

    /// <summary>The length, the channels and the rate every recording is read as.</summary>
    public SoundShape Shape { get; }

    /// <inheritdoc />
    public static string Name => "read.sounds";

    /// <inheritdoc />
    public static string Purpose =>
        "Reads the rows from a folder of WAV recordings, a folder to a class: a recording a row at the length, channels and rate the step says, the folder it stands in the answer, and a column of nought and one for each class.";

    /// <inheritdoc />
    public static StepParameters<ReadSoundsStep> Parameters { get; } = new StepParameters<ReadSoundsStep>()
        .With(PathKey, step => step.Path)
        .With(RateKey, step => step.Shape.Rate)
        .With(LengthKey, step => step.Shape.Length)
        .With(ChannelsKey, step => step.Shape.Channels);

    /// <inheritdoc />
    public static int Since => 12;

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <exception cref="DirectoryNotFoundException">There is no such folder.</exception>
    /// <exception cref="FormatException">
    /// A file stands beside the folders, none is a recording, a class is named as a column of the rows is, or a recording cannot be read
    /// or is not the shape the step says — each named with the file.
    /// </exception>
    public IRowSource Open(SourceFolder folder)
    {
        ArgumentNullException.ThrowIfNull(folder);

        return new FilesAsRows(folder.Resolve(Path), SoundColumn, Extensions, CellOf);
    }

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A key is missing or is of the wrong kind.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The rate, the length or the channels are below one.</exception>
    public static ReadSoundsStep ReadFrom(JsonElement element) =>
        new(PathKey.Read(element), new SoundShape(RateKey.Read(element), LengthKey.Read(element), ChannelsKey.Read(element)));

    private string CellOf(string file)
    {
        int rate;
        int channels;
        float[] samples;

        try
        {
            using var reader = new WaveFileReader(file);
            var provider = reader.ToSampleProvider();

            rate = provider.WaveFormat.SampleRate;
            channels = provider.WaveFormat.Channels;
            samples = new float[checked((int)Math.Min(int.MaxValue, reader.SampleCount * channels))];

            var read = 0;

            while (read < samples.Length)
            {
                var more = provider.Read(samples, read, samples.Length - read);

                if (more == 0)
                {
                    break;
                }

                read += more;
            }

            Array.Resize(ref samples, read);
        }
        catch (Exception fault) when (fault is InvalidOperationException or EndOfStreamException or OverflowException or IOException and not FileNotFoundException)
        {
            throw new FormatException($"it is no recording of a kind that is read here: {fault.Message}", fault);
        }

        var length = samples.Length / channels;

        if (rate != Shape.Rate || channels != Shape.Channels || length != Shape.Length)
        {
            throw new FormatException(string.Create(
                CultureInfo.InvariantCulture,
                $"it holds {length} samples of {channels} {(channels == 1 ? "channel" : "channels")} at {rate} a second and the step says {Shape.Length} samples of {Shape.Channels} {(Shape.Channels == 1 ? "channel" : "channels")} at {Shape.Rate} a second: recordings are never cut, mixed or resampled, so make them one shape first."));
        }

        return MediaCell.Samples([Shape.Length, Shape.Channels], [.. samples.Select(sample => (short)Math.Clamp(Math.Round(sample * 32768f), short.MinValue, short.MaxValue))]);
    }
}

/// <summary>Reading the rows from a folder of recordings, and teaching a catalog the verb a pipeline file names it by.</summary>
/// <remarks>
/// A reader is an extension method shipped by the package that brings what the format needs, so the verb exists exactly
/// when this package is referenced, and a project that reads no recording carries none of it.
/// </remarks>
public static class SoundsSourceExtensions
{
    extension(PipelineBuilder pipeline)
    {
        /// <summary>Declares that the rows come from a folder of WAV recordings, a folder to each class.</summary>
        /// <param name="path">Where the folder will be, when the pipeline runs; a relative path is read from the pipeline's folder.</param>
        /// <param name="shape">The length, the channels and the rate every recording is read as.</param>
        /// <returns>The pipeline, so the next verb can be written after it.</returns>
        /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
        /// <remarks>Nothing is opened until the pipeline runs. A recording of another shape is refused then, never cut, mixed or resampled.</remarks>
        public PipelineBuilder ReadSounds(string path, SoundShape shape)
        {
            ArgumentNullException.ThrowIfNull(pipeline);

            return pipeline.Add(new ReadSoundsStep(path, shape));
        }
    }

    extension(StepCatalog catalog)
    {
        /// <summary>Teaches a catalog to read <c>read.sounds</c> back out of a pipeline file.</summary>
        /// <returns>The same catalog, so registration reads as one sentence.</returns>
        public StepCatalog WithSounds()
        {
            ArgumentNullException.ThrowIfNull(catalog);

            catalog.Register<ReadSoundsStep>();

            return catalog;
        }
    }
}

/// <summary>The recording reader's verb, offered to whichever catalog an application builds.</summary>
public sealed class SoundsSteps : IStepContribution
{
    /// <inheritdoc />
    public void AddTo(StepCatalog catalog) => catalog.WithSounds();
}
