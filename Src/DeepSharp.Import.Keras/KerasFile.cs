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
/// <see cref="Sequential.Lower(DeepSharp.Tensors.Shape, RandomStream)"/> lowers any description — a batch normalisation with Keras's epsilon and the complement
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
/// The loss is the one the model was compiled with — a binary, a binary focal or a categorical cross-entropy, a mean squared
/// or a mean absolute error, a Huber, a Poisson or a hinge loss, a cross-entropy's label smoothing read as its smoothing —
/// and the network ends where that loss takes over: a last sigmoid before a binary cross-entropy, or a last softmax before
/// a categorical one, trained on the chances it gives, is lifted into the loss, which applies it to every prediction; a
/// model trained on its logits is read as it stands.
/// </para>
/// <para>
/// What is read: a Sequential model, or a Functional one, working in single precision, of Dense layers — one that has a low-rank pair beside its held
/// kernel (<c>enable_lora</c>) too, from the archive Keras 3 saves, which holds the kernel merged with the pair and so begins a
/// pair of its own, and from the HDF5 file, which holds the kernel and both matrices as they are; Conv1D, Conv2D and Conv3D convolutions,
/// their transposed, depthwise and separable forms, and ZeroPadding layers; MaxPooling and AveragePooling layers and
/// GlobalMaxPooling and GlobalAveragePooling layers along one, two and three axes, the global ones keeping their axes when
/// Keras's <c>keepdims</c> says; BatchNormalization, LayerNormalization, Dropout and SpatialDropout1D, SpatialDropout2D and
/// SpatialDropout3D layers; GroupNormalization — of a group for every channel, an instance normalisation — and
/// RMSNormalization; an Embedding as the first layer over one category's place, the network here taking that category's ones
/// and noughts instead; LSTM and GRU layers — alone, or inside a Bidirectional layer that lays the two directions side by side — read
/// to their last state or at every step, a GRU as Keras 3 makes it (the reset gate taken after the hidden numbers' own bias), whose numbers
/// a layer keeps under its cell and whose gates stand in another order than the library's, so they are put in it; MultiHeadAttention called over one series, as <c>layer(x, x)</c> — with <c>use_causal_mask</c> where it says so — whose four kernels, each with an axis for the heads, go into the layer's one projection and its output layer, found by the names the two files give them; Flatten, Reshape, Activation, ReLU, LeakyReLU, PReLU, ELU and Softmax layers; and the linear, relu,
/// tanh, sigmoid, gelu, silu, swish, mish, leaky_relu, elu, selu and softmax activations. A dense layer or a convolution without a bias (<c>use_bias</c> false) is read as a layer without one, and a
/// normalisation without its scale or its shift (<c>scale</c> or <c>center</c> false) as one that keeps neither; a layer's
/// L1, L2 and L1L2 regularizers, its MaxNorm constraints — over the axes a unit's numbers lie along — and its <c>trainable</c>
/// switch as the rules of its numbers (a penalty the loss adds, a bound after every step, a layer held still), while any other
/// regularizer or constraint, a MaxNorm over other axes, a bound on a transposed convolution's kernel and a penalty on what a
/// layer makes are refused by name; a separable
/// convolution's depthwise stage has no bias, since Keras adds its bias once, after both stages. A pooling's stride is its
/// window's size when it says none. What is refused, at the layer that says it and every such layer at once: any other kind
/// of layer or activation; a window padded otherwise than 'valid' or 'same' — and 'causal' for a convolution along a
/// series; any layer of series, images or volumes with its channels first; a normalisation over another axis than the last,
/// or a layer normalisation scaled by <c>rms_scaling</c>, which Keras works out from the variance; an embedding anywhere but
/// first, over more than one place, masking the place nought or of low-rank factors; a dropout by a noise shape; a reshape that leaves a length to be
/// worked out; a recurrent layer that reads backwards, drops numbers out inside itself, keeps its state from batch to batch, hands its state
/// back, works with other activations than the hyperbolic tangent and the sigmoid, keeps no bias or penalises its numbers, a GRU that takes
/// its reset gate before the bias (the form Keras 2 made the default), and a Bidirectional layer that merges otherwise than by concat; attention over two series, with a mask handed in, keeping no bias, gated, over a window or over other axes than the steps, or with a value dimension other than its key dimension, or with heads that each ask in more channels than the step is split into (<c>num_heads</c> times <c>key_dim</c> other than the step's width, as Keras's own text-classification example has it), which the layer's projection refuses by the shape it holds; a relu with a cap, a slope or a threshold; a model of another kind than a Sequential or a Functional one, of another precision,
/// of an input whose lengths are not all stated, or saved without its loss; and a focal loss that smooths its labels, or a
/// loss that adds a batch's losses up rather than taking their mean. A description that says a layer keeps less than the file's numbers
/// hold is refused at the number the network has no slot for, and one that says it keeps more at the slot left empty.
/// </para>
/// <para>
/// A Functional model is read as steps: each layer takes the layers it names — an Add of two or more, a Concatenate along the last
/// axis, or any other layer of one layer — and a layer that takes other than the one listed before it says which, so a value
/// added back after the layers that worked on it, or everything made before it laid beside the input, is the network it is. A
/// model of several inputs is handed to the network as the columns of one row, each input a block of columns in the order the
/// model lists its inputs, so each must be a row of numbers. Refused at the layer that says it: a layer called more than once
/// (a shared layer) or with anything but the tensors it takes, layers stated in the form Keras 2 wrote, a Concatenate along
/// another axis than the last, a model that gives more than one tensor or one that is not its last layer's, and a separable
/// convolution that does not take the layer listed before it.
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
