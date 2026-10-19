// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using StbImageSharp;

namespace DeepSharp.Pipelines;

/// <summary>
/// The size and the channels every picture of a folder is read as: its width and height in pixels, and how many numbers each pixel is.
/// </summary>
/// <param name="Width">How many pixels across; one or more.</param>
/// <param name="Height">How many pixels down; one or more.</param>
/// <param name="Channels">One for grey, three for red green blue, four for red green blue and how much shows through.</param>
public readonly record struct ImageShape(int Width, int Height, int Channels)
{
    /// <summary>How many pixels across.</summary>
    public int Width { get; } = Width >= 1
        ? Width
        : throw new ArgumentOutOfRangeException(nameof(Width), Width, "A picture is at least one pixel across.");

    /// <summary>How many pixels down.</summary>
    public int Height { get; } = Height >= 1
        ? Height
        : throw new ArgumentOutOfRangeException(nameof(Height), Height, "A picture is at least one pixel down.");

    /// <summary>How many numbers each pixel is: one, three or four.</summary>
    public int Channels { get; } = Channels is 1 or 3 or 4
        ? Channels
        : throw new ArgumentOutOfRangeException(nameof(Channels), Channels, "A pixel is one number for grey, three for red green blue, or four with how much shows through.");
}

/// <summary>
/// Reads the rows from a folder of pictures, a folder to a class: a picture a row, the folder it stands in the answer.
/// </summary>
/// <remarks>
/// The columns are <c>image</c> — the pixels as one media cell of the shape the step says, rows top to bottom, each pixel its channels
/// in turn — and <c>label</c>, the name of the folder, and then a column of nought and one for each folder, named for it, which a learner
/// is handed the answer by. A picture of another size is refused, naming it: nothing is resized, cropped or padded into a size the person
/// did not choose. A picture of another kind of pixel is converted to the channels the step says, which is a choice the step states.
/// Nothing is opened until the pipeline runs.
/// </remarks>
public sealed record ReadImagesStep : IPipelineStep<ReadImagesStep>, IOpensRows, IDescribesColumns
{
    /// <summary>The name of the column that holds each picture's pixels.</summary>
    public const string ImageColumn = "image";

    private static readonly string[] Extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tga"];

    private static readonly FilePathParameter PathKey = new(
        "path", "The folder the pictures will be in, when the pipeline runs: a folder to each class, pictures of .png, .jpg, .bmp, .gif or .tga in each.", "pictures");

    private static readonly WholeNumberParameter WidthKey = new("width", "How many pixels across every picture is; one that is not is refused, never resized.", 28) { AtLeast = 1 };

    private static readonly WholeNumberParameter HeightKey = new("height", "How many pixels down every picture is; one that is not is refused, never resized.", 28) { AtLeast = 1 };

    private static readonly WholeNumberParameter ChannelsKey = new("channels", "How many numbers each pixel is: 1 for grey, 3 for red green blue, 4 with how much shows through.", 3) { AtLeast = 1 };

    /// <summary>Declares that the rows come from the pictures of this folder.</summary>
    /// <param name="path">Where the folder will be, when the pipeline runs.</param>
    /// <param name="shape">The size and the channels every picture is read as.</param>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    public ReadImagesStep(string path, ImageShape shape)
    {
        Path = PathKey.Require(path);
        Shape = shape;
    }

    /// <summary>Where the folder will be, when the pipeline runs.</summary>
    public string Path { get; }

    /// <summary>The size and the channels every picture is read as.</summary>
    public ImageShape Shape { get; }

    /// <inheritdoc />
    public static string Name => "read.images";

    /// <inheritdoc />
    public static string Purpose =>
        "Reads the rows from a folder of pictures, a folder to a class: a picture a row at the size and channels the step says, the folder it stands in the answer, and a column of nought and one for each class.";

    /// <inheritdoc />
    public static StepParameters<ReadImagesStep> Parameters { get; } = new StepParameters<ReadImagesStep>()
        .With(PathKey, step => step.Path)
        .With(WidthKey, step => step.Shape.Width)
        .With(HeightKey, step => step.Shape.Height)
        .With(ChannelsKey, step => step.Shape.Channels);

    /// <inheritdoc />
    public static int Since => 12;

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <exception cref="DirectoryNotFoundException">There is no such folder.</exception>
    /// <exception cref="FormatException">
    /// A file stands beside the folders, none is a picture, a class is named as a column of the rows is, or a picture cannot be read
    /// or is not the size the step says — each named with the file.
    /// </exception>
    public IRowSource Open(SourceFolder folder)
    {
        ArgumentNullException.ThrowIfNull(folder);

        return new FilesAsRows(folder.Resolve(Path), ImageColumn, Extensions, CellOf);
    }

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A key is missing or is of the wrong kind.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The size is below one, or the channels are not one, three or four.</exception>
    public static ReadImagesStep ReadFrom(JsonElement element) =>
        new(PathKey.Read(element), new ImageShape(WidthKey.Read(element), HeightKey.Read(element), ChannelsKey.Read(element)));

    private string CellOf(string file)
    {
        var wanted = Shape.Channels switch
        {
            1 => ColorComponents.Grey,
            3 => ColorComponents.RedGreenBlue,
            _ => ColorComponents.RedGreenBlueAlpha,
        };
        ImageResult image;

        try
        {
            image = ImageResult.FromMemory(File.ReadAllBytes(file), wanted);
        }
        catch (InvalidOperationException fault)
        {
            throw new FormatException($"it is no picture of a kind that is read here: {fault.Message}", fault);
        }

        return image.Width == Shape.Width && image.Height == Shape.Height
            ? MediaCell.Pixels([Shape.Height, Shape.Width, Shape.Channels], image.Data)
            : throw new FormatException(string.Create(
                CultureInfo.InvariantCulture,
                $"it is {image.Width} by {image.Height} pixels and the step says {Shape.Width} by {Shape.Height}: pictures are never resized, so make them one size first."));
    }
}

/// <summary>Reading the rows from a folder of pictures, and teaching a catalog the verb a pipeline file names it by.</summary>
/// <remarks>
/// A reader is an extension method shipped by the package that brings what the format needs, so the verb exists exactly
/// when this package is referenced, and a project that reads no picture carries none of it.
/// </remarks>
public static class ImagesSourceExtensions
{
    extension(PipelineBuilder pipeline)
    {
        /// <summary>Declares that the rows come from a folder of pictures, a folder to each class.</summary>
        /// <param name="path">Where the folder will be, when the pipeline runs; a relative path is read from the pipeline's folder.</param>
        /// <param name="shape">The size and the channels every picture is read as.</param>
        /// <returns>The pipeline, so the next verb can be written after it.</returns>
        /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
        /// <remarks>Nothing is opened until the pipeline runs. A picture of another size is refused then, never resized.</remarks>
        public PipelineBuilder ReadImages(string path, ImageShape shape)
        {
            ArgumentNullException.ThrowIfNull(pipeline);

            return pipeline.Add(new ReadImagesStep(path, shape));
        }
    }

    extension(StepCatalog catalog)
    {
        /// <summary>Teaches a catalog to read <c>read.images</c> back out of a pipeline file.</summary>
        /// <returns>The same catalog, so registration reads as one sentence.</returns>
        public StepCatalog WithImages()
        {
            ArgumentNullException.ThrowIfNull(catalog);

            catalog.Register<ReadImagesStep>();

            return catalog;
        }
    }
}

/// <summary>The picture reader's verb, offered to whichever catalog an application builds.</summary>
public sealed class ImagesSteps : IStepContribution
{
    /// <inheritdoc />
    public void AddTo(StepCatalog catalog) => catalog.WithImages();
}
