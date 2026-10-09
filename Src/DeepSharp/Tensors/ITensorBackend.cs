// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// Where a tensor's arithmetic actually happens.
/// </summary>
/// <remarks>
/// This seam exists from the first line for one reason: a model written against it does not know, and never
/// has to learn, whether its arithmetic runs on this machine's vector registers or somewhere else entirely.
/// A backend is handed to whatever needs one; nothing reaches for a shared instance of its own accord.
/// <para>
/// The operations are the ones a network, its loss, its optimizer and its backward pass need, and no more: the backward
/// pass of each is written in these same operations, so working out a gradient never needs anything a backend does not
/// offer. A layer, a loss or a convolution is not an operation here: each is written in these, so a backend has nothing
/// to add for it.
/// </para>
/// <para>
/// A backend that keeps its values somewhere of its own hands back what it makes on a <see cref="TensorStorage"/> of its
/// own, with <see cref="Tensor.On(Shape, TensorStorage)"/>, and takes in a tensor on any other storage where an operation
/// reads it; nothing here says where values live, so nothing here changes for it. Every operation asks first what
/// <see cref="TensorOperandExtensions"/> asks of its tensors, so every backend refuses what the light one refuses, in the
/// same words.
/// </para>
/// </remarks>
public interface ITensorBackend
{
    /// <summary>
    /// Which backend this is, for a log line or an error message, and what a checkpoint records its run was on, with the version
    /// and the device a backend names through <see cref="INamesItsVersionAndDevice"/>. Lowercase, one word.
    /// </summary>
    string Name { get; }

    /// <summary>Adds two tensors of the same shape, value by value.</summary>
    /// <param name="left">The first tensor.</param>
    /// <param name="right">A tensor of the same shape.</param>
    /// <returns>A new tensor; neither argument is changed.</returns>
    /// <exception cref="ArgumentException">The two shapes differ.</exception>
    Tensor Add(Tensor left, Tensor right);

    /// <summary>Subtracts one tensor from another of the same shape, value by value.</summary>
    /// <param name="left">The tensor subtracted from.</param>
    /// <param name="right">A tensor of the same shape, subtracted from it.</param>
    /// <returns>A new tensor; neither argument is changed.</returns>
    /// <exception cref="ArgumentException">The two shapes differ.</exception>
    Tensor Subtract(Tensor left, Tensor right);

    /// <summary>Multiplies two tensors of the same shape, value by value.</summary>
    /// <param name="left">The first tensor.</param>
    /// <param name="right">A tensor of the same shape.</param>
    /// <returns>A new tensor; neither argument is changed.</returns>
    /// <exception cref="ArgumentException">The two shapes differ.</exception>
    Tensor Multiply(Tensor left, Tensor right);

    /// <summary>Multiplies two matrices: each row of the left against each column of the right.</summary>
    /// <param name="left">A matrix of so many rows, as wide as the right one is tall.</param>
    /// <param name="right">A matrix as tall as the left one is wide.</param>
    /// <returns>A new matrix as tall as the left one and as wide as the right one.</returns>
    /// <exception cref="ArgumentException">Either is not a matrix, or the left is not as wide as the right is tall.</exception>
    Tensor MatMul(Tensor left, Tensor right);

    /// <summary>Turns a matrix's rows into its columns.</summary>
    /// <param name="matrix">The matrix.</param>
    /// <returns>A new matrix as tall as the given one is wide.</returns>
    /// <exception cref="ArgumentException">The tensor is not a matrix.</exception>
    Tensor Transpose(Tensor matrix);

    /// <summary>Adds one row to every row of a matrix, as a layer adds its bias to every example of a batch.</summary>
    /// <param name="matrix">The matrix.</param>
    /// <param name="row">A tensor of one axis, as long as the matrix is wide.</param>
    /// <returns>A new matrix of the same shape.</returns>
    /// <exception cref="ArgumentException">
    /// The first is not a matrix, the second is not a row, or the row is not as long as the matrix is wide.
    /// </exception>
    Tensor AddRow(Tensor matrix, Tensor row);

    /// <summary>Adds up the rows of a matrix: one total for each column.</summary>
    /// <param name="matrix">The matrix.</param>
    /// <returns>A new tensor of one axis, as long as the matrix is wide.</returns>
    /// <exception cref="ArgumentException">The tensor is not a matrix.</exception>
    Tensor SumRows(Tensor matrix);

    /// <summary>The average of every value, as one value with no axes: what a loss is.</summary>
    /// <param name="values">The values.</param>
    /// <returns>A new tensor with no axes.</returns>
    /// <exception cref="ArgumentException">The tensor holds no values.</exception>
    Tensor Mean(Tensor values);

    /// <summary>Multiplies every value by one value.</summary>
    /// <param name="values">The values.</param>
    /// <param name="factor">A tensor with no axes: the one value to multiply by.</param>
    /// <returns>A new tensor of the shape of the values.</returns>
    /// <exception cref="ArgumentException">The factor has axes.</exception>
    Tensor Scale(Tensor values, Tensor factor);

    /// <summary>A tensor of a shape with every value the same.</summary>
    /// <param name="shape">The shape.</param>
    /// <param name="value">The value every position holds.</param>
    /// <returns>A new tensor.</returns>
    Tensor Fill(Shape shape, float value);

    /// <summary>Keeps every value above nothing and makes the rest nothing: the rectifier a layer bends its output with.</summary>
    /// <param name="values">The values.</param>
    /// <returns>A new tensor of the same shape.</returns>
    Tensor Relu(Tensor values);

    /// <summary>One where a value is above nothing and nought everywhere else, nothing itself included.</summary>
    /// <param name="values">The values.</param>
    /// <returns>A new tensor of the same shape.</returns>
    /// <remarks>The step the rectifier's gradient is made of: a gradient passes where the value was above nothing.</remarks>
    Tensor Positive(Tensor values);

    /// <summary>One where a value is the first largest of its row and nought everywhere else: which value of each row a maximum is.</summary>
    /// <param name="matrix">The matrix, one row for each choice to make.</param>
    /// <returns>A new matrix of the same shape, with one one in each row.</returns>
    /// <exception cref="ArgumentException">The tensor is not a matrix, or its rows hold no value to pick.</exception>
    /// <remarks>
    /// The step a maximum's gradient is made of, as <see cref="Positive"/> is the rectifier's: the values times this, added up
    /// along each row, are the row's largest value, and the gradient of that goes to the value picked and to no other. Of values
    /// that tie the first is picked, and a value that is not a number counts as the largest, the first of them the one picked,
    /// as libtorch's argmax has it. Like <see cref="Positive"/> it is flat either side of where it steps, so it sends nothing back.
    /// </remarks>
    Tensor FirstLargest(Tensor matrix);

    /// <summary>The hyperbolic tangent of every value, between minus one and one.</summary>
    /// <param name="values">The values.</param>
    /// <returns>A new tensor of the same shape.</returns>
    Tensor Tanh(Tensor values);

    /// <summary>The logistic function of every value, between nothing and one: one over one plus the exponential of minus it.</summary>
    /// <param name="values">The values.</param>
    /// <returns>A new tensor of the same shape.</returns>
    Tensor Sigmoid(Tensor values);

    /// <summary>The exponential of every value.</summary>
    /// <param name="values">The values.</param>
    /// <returns>A new tensor of the same shape.</returns>
    Tensor Exp(Tensor values);

    /// <summary>The natural logarithm of every value.</summary>
    /// <param name="values">The values, each above nothing for a logarithm that is a number.</param>
    /// <returns>A new tensor of the same shape.</returns>
    Tensor Log(Tensor values);

    /// <summary>The square root of every value.</summary>
    /// <param name="values">The values, none below nothing for a root that is a number.</param>
    /// <returns>A new tensor of the same shape.</returns>
    Tensor Sqrt(Tensor values);

    /// <summary>The logarithm of one plus the exponential of every value, worked out so that no value is lost at either end.</summary>
    /// <param name="values">The values.</param>
    /// <returns>A new tensor of the same shape.</returns>
    /// <remarks>
    /// What a binary cross-entropy on logits is written in: the softplus of a logit, less the logit where the answer is one.
    /// At forty it is forty rather than the infinity an exponential taken first gives, and at minus forty it is four
    /// quintillionths rather than the nothing one plus a number that small becomes.
    /// </remarks>
    Tensor Softplus(Tensor values);

    /// <summary>Divides one tensor by another of the same shape, value by value.</summary>
    /// <param name="left">The tensor divided.</param>
    /// <param name="right">A tensor of the same shape, divided by.</param>
    /// <returns>A new tensor; neither argument is changed.</returns>
    /// <exception cref="ArgumentException">The two shapes differ.</exception>
    Tensor Divide(Tensor left, Tensor right);

    /// <summary>Each row of a matrix less the logarithm of the sum of its exponentials: the logarithm of the row's shares.</summary>
    /// <param name="matrix">The matrix, one row for each example and one column for each class.</param>
    /// <returns>A new matrix of the same shape.</returns>
    /// <exception cref="ArgumentException">The tensor is not a matrix.</exception>
    /// <remarks>
    /// Each row is shifted by its largest value before anything is raised to a power, so a row of values in the thousands
    /// has shares as well as a row of small ones. A cross-entropy is written in it, and the shares themselves are its
    /// exponential.
    /// </remarks>
    Tensor LogSoftmax(Tensor matrix);

    /// <summary>The same values, in their order, under another shape holding as many.</summary>
    /// <param name="values">The values.</param>
    /// <param name="shape">The shape to lay them out along.</param>
    /// <returns>A new tensor of the given shape.</returns>
    /// <exception cref="ArgumentException">The shape holds another number of values.</exception>
    Tensor Reshape(Tensor values, Shape shape);

    /// <summary>
    /// The patches a window covers as it walks over a batch of images: one row for every place it stands, holding what it
    /// covers there.
    /// </summary>
    /// <param name="images">The images, laid out image by row by column by channel: channels last.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <returns>
    /// A new matrix with a row for every image and every place — image first, then row, then column — and in each row the
    /// window's values row by row, every channel of a place before the next place. Where the window stands on the border,
    /// it covers nothing there.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// The images are not a batch of rows, columns and channels, or the window cannot stand on them: a side or a stride
    /// below one, a border below nothing, a border given to a window padded as 'same', or a window larger than the image
    /// with its border.
    /// </exception>
    /// <remarks>
    /// A convolution is these patches times a kernel, one column for each channel it makes. The window's border, on each
    /// side, is what <see cref="Window.BordersOver"/> says for the images' rows and columns — the one rule every engine
    /// reads, so a window padded as 'same' stands at the same places on every engine.
    /// </remarks>
    Tensor Unfold(Tensor images, Window window);

    /// <summary>
    /// Puts every value of a matrix of patches back where <see cref="Unfold"/> took it from, adding the values that land
    /// on the same place.
    /// </summary>
    /// <param name="patches">The patches, laid out as <see cref="Unfold"/> lays them out.</param>
    /// <param name="images">The shape of the images the patches were taken from, channels last.</param>
    /// <param name="window">The window that took them.</param>
    /// <returns>A new tensor of the images' shape; what fell on the border is dropped.</returns>
    /// <exception cref="ArgumentException">
    /// The shape is not one of images, the window cannot stand on them, or the patches are not the ones such a window takes
    /// of such images.
    /// </exception>
    /// <remarks>Unfolding turned around, which is why each is the other's way back: the window stands where <see cref="Window.BordersOver"/> puts it.</remarks>
    Tensor Fold(Tensor patches, Shape images, Window window);
}
