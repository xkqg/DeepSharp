// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Import.Keras;

/// <summary>
/// Reads a model Keras 3 saved into a network here: the <c>.keras</c> archive it saves a model to, or the HDF5 file,
/// <c>.h5</c>, it saved one to before.
/// </summary>
/// <remarks>
/// <para>
/// The file describes its network as well as holding its numbers, so the network is built from that description: each
/// layer it names is written in Keras's words and lowered onto a <see cref="LayerStack"/>, as
/// <see cref="Sequential.Lower"/> lowers any description — a batch normalisation with Keras's epsilon and the complement
/// of its momentum, a window padded as TensorFlow's 'same' pads, the activation a Keras layer carries a layer of its own
/// after it. The numbers need no turning round: a dense layer's weights are inputs by outputs, and a convolution's kernel,
/// which Keras writes as a length for each side of its window — one along a series, rows and columns of an image, planes,
/// rows and columns of a volume — then channels in by channels out, is the same numbers in the same order as the slot keeps
/// them: the window's places times the channels in, by the channels out. Each goes into its slot by path — the
/// layer's place in the stack and the slot's name, <c>2.weight</c> — through <see cref="Network.Load"/>, so the numbers go
/// in all together or not at all, and every one that does not fit is named where the file holds it. A layer's numbers are
/// found by the name the file gives them, never by where they stand: Keras files them under its class, <c>layers/dense</c>,
/// while the layer is <c>dense_2</c>.
/// </para>
/// <para>
/// The loss is the one the model was compiled with — a binary or a categorical cross-entropy, or a mean squared error —
/// and the network ends where that loss takes over: a last sigmoid before a binary cross-entropy, or a last softmax before
/// a categorical one, trained on the chances it gives, is lifted into the loss, which applies it to every prediction; a
/// model trained on its logits is read as it stands.
/// </para>
/// <para>
/// What is read: a Sequential model working in single precision, of Dense layers; Conv1D, Conv2D and Conv3D convolutions;
/// MaxPooling and AveragePooling layers and GlobalMaxPooling and GlobalAveragePooling layers along one, two and three axes,
/// the global ones keeping their axes when Keras's <c>keepdims</c> says; BatchNormalization, LayerNormalization, Dropout
/// and SpatialDropout1D, SpatialDropout2D and SpatialDropout3D layers; and Flatten, Reshape, Activation and ReLU layers; and
/// the linear, relu, tanh and sigmoid activations. A pooling's stride is its window's size when it says none. What is
/// refused, at the layer that says it and every such layer at once: any other kind of layer or activation; a convolution
/// or a pooling striding otherwise along one axis than along another, a convolution dilated or with its channels in
/// groups; a window padded otherwise than 'valid' or 'same' — and 'causal' for a convolution along a series; any layer of
/// series, images or volumes with its channels first; a layer without a bias; a normalisation over another axis than the
/// last, or without its shift or its scale, or a layer normalisation by the root mean square alone; a dropout by a noise
/// shape; a reshape that leaves a length to be worked out; a relu with a cap, a slope or a threshold; a model of another
/// kind than a Sequential, of another precision, of an input whose lengths are not all stated, or saved without its loss;
/// and a loss that smooths its labels or adds a batch's losses up rather than taking their mean.
/// </para>
/// </remarks>
public sealed class KerasFile : IImporter
{
    /// <inheritdoc />
    /// <exception cref="ArgumentNullException">No file is given.</exception>
    /// <exception cref="FormatException">
    /// The file is neither a Keras archive nor an HDF5 file Keras saved a model to, or its model says what no network here
    /// is built of: every such thing at once, each where the file says it.
    /// </exception>
    public SavedNetwork Read(Stream file)
    {
        ArgumentNullException.ThrowIfNull(file);

        var saved = KerasSaved.Open(file);

        return new KerasModel(saved).Loaded(saved.Numbers);
    }
}
