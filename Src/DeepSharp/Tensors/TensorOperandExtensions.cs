// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Tensors;

/// <summary>
/// What the seam refuses, said once: each check an operation of <see cref="ITensorBackend"/> makes of what it is handed,
/// before any arithmetic, in the words its refusal is given.
/// </summary>
/// <remarks>
/// Every engine calls these first, the light one included, so an engine of your own refuses what the light one refuses —
/// the same exception, the same words, naming the same argument — rather than whatever its own library does with two
/// shapes that differ: stretch one over the other, answer with something that is not a number, or throw an exception of
/// its own. A model that one engine refuses is refused by every engine, and cannot tell them apart by what either lets
/// through.
/// </remarks>
public static class TensorOperandExtensions
{
    extension(Tensor left)
    {
        /// <summary>
        /// What <see cref="ITensorBackend.Add"/>, <see cref="ITensorBackend.Subtract"/>, <see cref="ITensorBackend.Multiply"/>
        /// and <see cref="ITensorBackend.Divide"/> ask of their tensors: both handed, of one shape.
        /// </summary>
        /// <param name="right">The second.</param>
        /// <param name="operation">The operation's name, for the refusal's words.</param>
        /// <exception cref="ArgumentNullException">Either is not handed.</exception>
        /// <exception cref="ArgumentException">The two shapes differ.</exception>
        public void RequireSameShape(Tensor right, string operation)
        {
            ArgumentNullException.ThrowIfNull(left);
            ArgumentNullException.ThrowIfNull(right);

            if (left.Shape != right.Shape)
            {
                throw new ArgumentException(
                    $"{operation} needs two tensors of the same shape, and was given {left.Shape} and {right.Shape}.", nameof(right));
            }
        }

        /// <summary>What <see cref="ITensorBackend.MatMul"/> asks of its tensors: two matrices, the left as wide as the right is tall.</summary>
        /// <param name="right">The right matrix.</param>
        /// <exception cref="ArgumentNullException">Either is not handed.</exception>
        /// <exception cref="ArgumentException">Either is not a matrix, or the left is not as wide as the right is tall.</exception>
        public void RequireMatrixProduct(Tensor right)
        {
            Matrix(left, nameof(ITensorBackend.MatMul), nameof(left));
            Matrix(right, nameof(ITensorBackend.MatMul), nameof(right));

            if (right.Shape[0] != left.Shape[1])
            {
                throw new ArgumentException(
                    $"MatMul needs the left matrix as wide as the right one is tall, and was given {left.Shape} and {right.Shape}.",
                    nameof(right));
            }
        }
    }

    extension(Tensor matrix)
    {
        /// <summary>
        /// What <see cref="ITensorBackend.Transpose"/>, <see cref="ITensorBackend.SumRows"/> and
        /// <see cref="ITensorBackend.LogSoftmax"/> ask of their tensor: a matrix.
        /// </summary>
        /// <param name="operation">The operation's name, for the refusal's words.</param>
        /// <exception cref="ArgumentNullException">It is not handed.</exception>
        /// <exception cref="ArgumentException">It is not a matrix.</exception>
        public void RequireMatrix(string operation) => Matrix(matrix, operation, nameof(matrix));

        /// <summary>What <see cref="ITensorBackend.FirstLargest"/> asks of its tensor: a matrix whose rows hold a value at least.</summary>
        /// <param name="operation">The operation's name, for the refusal's words.</param>
        /// <exception cref="ArgumentNullException">It is not handed.</exception>
        /// <exception cref="ArgumentException">It is not a matrix, or its rows hold no value.</exception>
        public void RequireRowsToPickFrom(string operation)
        {
            Matrix(matrix, operation, nameof(matrix));

            if (matrix.Shape[1] < 1)
            {
                throw new ArgumentException($"{operation} picks one value of each row, and a {matrix.Shape} matrix has rows of none.", nameof(matrix));
            }
        }

        /// <summary>What <see cref="ITensorBackend.AddRow"/> asks of its tensors: a matrix, and a row as long as it is wide.</summary>
        /// <param name="row">The row.</param>
        /// <exception cref="ArgumentNullException">Either is not handed.</exception>
        /// <exception cref="ArgumentException">
        /// The first is not a matrix, the second is not a row, or the row is not as long as the matrix is wide.
        /// </exception>
        public void RequireRow(Tensor row)
        {
            Matrix(matrix, nameof(ITensorBackend.AddRow), nameof(matrix));
            ArgumentNullException.ThrowIfNull(row);

            if (row.Shape.Rank != 1)
            {
                throw new ArgumentException($"AddRow needs a row, a tensor of one axis, and was given a {row.Shape} one.", nameof(row));
            }

            if (row.Shape[0] != matrix.Shape[1])
            {
                throw new ArgumentException(
                    $"AddRow needs a row as long as the matrix is wide, and was given {matrix.Shape} and {row.Shape}.", nameof(row));
            }
        }
    }

    extension(Tensor values)
    {
        /// <summary>What <see cref="ITensorBackend.Mean"/> asks of its tensor: at least one value.</summary>
        /// <exception cref="ArgumentNullException">It is not handed.</exception>
        /// <exception cref="ArgumentException">It holds no values.</exception>
        public void RequireValues()
        {
            ArgumentNullException.ThrowIfNull(values);

            if (values.Shape.Count == 0)
            {
                throw new ArgumentException($"A mean needs at least one value, and a {values.Shape} tensor holds none.", nameof(values));
            }
        }

        /// <summary>What <see cref="ITensorBackend.Scale"/> asks of its tensors: values, and a factor of one value with no axes.</summary>
        /// <param name="factor">The factor.</param>
        /// <exception cref="ArgumentNullException">Either is not handed.</exception>
        /// <exception cref="ArgumentException">The factor has axes.</exception>
        public void RequireFactor(Tensor factor)
        {
            ArgumentNullException.ThrowIfNull(values);
            ArgumentNullException.ThrowIfNull(factor);

            if (factor.Shape.Rank != 0)
            {
                throw new ArgumentException(
                    $"Scale multiplies by one value, a tensor with no axes, and was given a {factor.Shape} one.", nameof(factor));
            }
        }

        /// <summary>What <see cref="ITensorBackend.Reshape"/> asks of its tensor and shape: as many values in the one as the other holds.</summary>
        /// <param name="shape">The shape they are to be laid out along.</param>
        /// <exception cref="ArgumentNullException">The values are not handed.</exception>
        /// <exception cref="ArgumentException">The shape holds another number of values.</exception>
        public void RequireSameCount(Shape shape)
        {
            ArgumentNullException.ThrowIfNull(values);

            if (shape.Count != values.Shape.Count)
            {
                throw new ArgumentException(
                    $"Reshape keeps every value, and a {values.Shape} tensor holds {values.Shape.Count} where a {shape} one holds {shape.Count}.",
                    nameof(shape));
            }
        }
    }

    extension(Tensor images)
    {
        /// <summary>What <see cref="ITensorBackend.Unfold"/> asks of its images and window: a batch of images the window stands on.</summary>
        /// <param name="window">The window.</param>
        /// <exception cref="ArgumentNullException">The images are not handed.</exception>
        /// <exception cref="ArgumentException">
        /// The images are not a batch of rows, columns and channels, or the window cannot stand on them: a side or a stride
        /// below one, a border below nothing, a border given to a window padded as 'same', or a window larger than the image
        /// with its border.
        /// </exception>
        public void RequireImagesFor(Window window)
        {
            ArgumentNullException.ThrowIfNull(images);
            Images(images.Shape, window, nameof(ITensorBackend.Unfold), nameof(images));
        }
    }

    extension(Tensor patches)
    {
        /// <summary>
        /// What <see cref="ITensorBackend.Fold"/> asks of its patches, images and window: the patches such a window takes of
        /// such images.
        /// </summary>
        /// <param name="images">The shape of the images they were taken from, channels last.</param>
        /// <param name="window">The window that took them.</param>
        /// <exception cref="ArgumentNullException">The patches are not handed.</exception>
        /// <exception cref="ArgumentException">
        /// The shape is not one of images, the window cannot stand on them, or the patches are not the ones such a window takes
        /// of such images.
        /// </exception>
        public void RequirePatchesOf(Shape images, Window window)
        {
            ArgumentNullException.ThrowIfNull(patches);
            Images(images, window, nameof(ITensorBackend.Fold), nameof(images));

            var taken = window.PatchesOver(images);

            if (patches.Shape != taken)
            {
                throw new ArgumentException(
                    $"Fold puts back the {taken} patches a {window} takes of {images} images, and was given {patches.Shape}.",
                    nameof(patches));
            }
        }
    }

    extension(Window window)
    {
        /// <summary>
        /// Refuses a window that cannot stand anywhere: a side or the stride below one, or the border below nothing — or, padded
        /// as 'same', given a border of its own besides the one it works out.
        /// </summary>
        /// <remarks>The one rule for the window, whether a convolution is being built or its window is being walked.</remarks>
        internal void RequireStanding()
        {
            if (window.Height < 1 || window.Width < 1 || window.Stride < 1 || window.Padding < 0
                || (window.Border is { } sides && (sides.Top < 0 || sides.Bottom < 0 || sides.Left < 0 || sides.Right < 0)))
            {
                throw new ArgumentException(
                    $"A {window} cannot stand anywhere: its sides and its stride are at least one, and its border at least nothing.", nameof(window));
            }

            if (window.Border is not null && (window.Padding != 0 || window.PaddingMode != PaddingMode.Stated))
            {
                throw new ArgumentException($"A {window} states its border for each side, and cannot be given a padding or a way to work one out besides.", nameof(window));
            }

            if (window.PaddingMode != PaddingMode.Stated && window.Padding != 0)
            {
                throw new ArgumentException(
                    string.Create(CultureInfo.InvariantCulture, $"A {window} works its border out from each image it stands on, and cannot be given one of {window.Padding} besides."),
                    nameof(window));
            }
        }

        /// <summary>
        /// The patches a window takes of a batch of images: a row for every image and every place it stands, each as long as the
        /// window holds values.
        /// </summary>
        internal Shape PatchesOver(Shape images) =>
            new(images[0] * window.RowsOver(images[1]) * window.ColumnsOver(images[2]), window.Height * window.Width * images[3]);
    }

    // A tensor that is not a matrix, refused in the words of the operation that needs one.
    private static void Matrix(Tensor tensor, string operation, string parameter)
    {
        ArgumentNullException.ThrowIfNull(tensor, parameter);

        if (tensor.Shape.Rank != 2)
        {
            throw new ArgumentException($"{operation} works on matrices, and was given a {tensor.Shape} tensor.", parameter);
        }
    }

    // A shape that is not a batch of images, or a window that cannot stand on them.
    private static void Images(Shape images, Window window, string operation, string parameter)
    {
        if (images.Rank != 4)
        {
            throw new ArgumentException(
                $"{operation} works on a batch of images, image by row by column by channel, and was given a {images} one.", parameter);
        }

        window.RequireStanding();

        if (window.RowsOver(images[1]) < 1 || window.ColumnsOver(images[2]) < 1)
        {
            throw new ArgumentException($"A {window} is larger than a {images} image with its border.", nameof(window));
        }
    }
}
