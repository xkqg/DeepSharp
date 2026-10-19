// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// How the numbers of a layer begin: the draws they are taken from — and, where it is not as PyTorch starts a layer, how its
/// weights start, how its bias starts, or that it has no bias at all.
/// </summary>
/// <remarks>
/// The draws are all a start needs, so <c>new Beginning(draws)</c> starts a layer exactly as it always started: the weights evenly within one over the root of the inputs they are
/// read against, then the bias within the same, out of the draws in that order. Saying more changes only what is said: the
/// weights are drawn first and the bias second whatever starts them, so a layer that changes how its bias starts starts its
/// weights where it did. A bias that is left out draws nothing.
/// <para>
/// It states how a layer starts and nothing the layer does afterwards: a network's file holds the numbers a layer learned,
/// never how it began, so a layer rebuilt from a file is begun with the draws it is handed and then given the file's numbers.
/// What a file does say is that a layer has no bias, since the slot is not there to be given numbers.
/// </para>
/// </remarks>
/// <param name="Draws">The draws the start is taken from: the weights first, then the bias.</param>
public readonly record struct Beginning(Draws Draws)
{
    /// <summary>How the weights start; as PyTorch starts a linear layer's and a convolution's, unless said.</summary>
    public Initialiser? Weights { get; init; }

    /// <summary>How the bias starts; evenly within one over the root of the inputs, as PyTorch starts it, unless said.</summary>
    public Initialiser? Bias { get; init; }

    /// <summary>Whether the layer has no bias at all; it has one, unless said.</summary>
    public bool WithoutBias { get; init; }

    /// <summary>The low-rank pair a dense layer keeps beside its held kernel; none, unless said.</summary>
    /// <remarks>
    /// Like a bias, a pair is a number the layer has or has not, so it is said where the layer begins. A, the pair\'s first matrix,
    /// is drawn after the weights and the bias, evenly as He\'s uniform initialisation draws it, which is Keras\'s <c>he_uniform</c>;
    /// B starts at nothing and draws nothing. Only a dense layer takes one.
    /// </remarks>
    public LowRank? LowRank { get; init; }

    /// <summary>Refuses a beginning without draws, and one that wants no bias and says how to start it.</summary>
    /// <returns>This beginning.</returns>
    /// <exception cref="ArgumentNullException">There are no draws.</exception>
    /// <exception cref="ArgumentException">It wants no bias and says how its bias starts.</exception>
    internal Beginning Required()
    {
        ArgumentNullException.ThrowIfNull(Draws);

        if (WithoutBias && Bias is not null)
        {
            throw new ArgumentException("A layer without a bias has none to start: either it has no bias or it says how the bias starts.", nameof(Bias));
        }

        return this;
    }

    /// <summary>The weights of a layer, started as this beginning says.</summary>
    /// <param name="shape">The weights' shape.</param>
    /// <param name="fans">How many values they are read against and feed.</param>
    /// <returns>The weights.</returns>
    internal Tensor WeightsOf(Shape shape, Fans fans) => (Weights ?? new KaimingUniform()).Draw(shape, fans, Draws);

    /// <summary>The bias of a layer, started as this beginning says; nothing for a layer that has none.</summary>
    /// <param name="shape">The bias' shape.</param>
    /// <param name="fans">How many values the layer reads against and feeds.</param>
    /// <returns>The bias, or nothing.</returns>
    internal Tensor? BiasOf(Shape shape, Fans fans) => WithoutBias ? null : (Bias ?? new FanInUniform()).Draw(shape, fans, Draws);
}
