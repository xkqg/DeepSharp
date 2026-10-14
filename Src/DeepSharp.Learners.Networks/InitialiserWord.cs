// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Learners.Networks;

/// <summary>
/// The ways a declared layer's numbers can start, by the name of the initialiser that starts them.
/// </summary>
/// <remarks>
/// A declaration is words, so the starts it can name are the ones that need no number of their own: each is the
/// initialiser of that name with the numbers its class leaves unsaid. A start that needs a number — a constant, a
/// deviation other than the default — is written as code, where the initialiser takes it. A file writes each word in
/// lower case, and reads it in any case.
/// </remarks>
public enum InitialiserWord
{
    /// <summary>
    /// Evenly within one over the root of the inputs a weight is read against: <c>KaimingUniform</c> as PyTorch starts a
    /// linear layer's and a convolution's weights.
    /// </summary>
    KaimingUniform,

    /// <summary>Evenly within the root of six over the inputs and outputs together: Glorot's, which Keras starts a layer's weights at.</summary>
    GlorotUniform,

    /// <summary>Evenly within one over the root of the inputs: as PyTorch starts a bias.</summary>
    FanInUniform,

    /// <summary>All nought: as Keras starts a bias.</summary>
    Zeros,

    /// <summary>All one.</summary>
    Ones,

    /// <summary>Normally, around nought, with a deviation of five hundredths: as Keras's random normal.</summary>
    Normal,

    /// <summary>Normally, around nought, with a deviation of five hundredths, redrawn when it falls beyond two deviations.</summary>
    TruncatedNormal,

    /// <summary>A matrix whose rows or columns are at right angles to each other, of a gain of one.</summary>
    Orthogonal,
}
