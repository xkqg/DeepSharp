// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>What an <see cref="OutputActivation"/> is: the one place a loss, a stack's compile and an importer ask it.</summary>
public static class OutputActivationExtensions
{
    extension(OutputActivation activation)
    {
        /// <summary>The activation's word, as a message and a foreign file name it: <c>softmax</c>, <c>sigmoid</c>, <c>exp</c>, <c>margin</c>, <c>identity</c>.</summary>
        /// <returns>The word, in lower case.</returns>
        public string Word() => activation.ToString().ToLower(CultureInfo.InvariantCulture);

        /// <summary>The outputs through the activation: what a prediction is made of them.</summary>
        /// <param name="outputs">The network's raw outputs.</param>
        /// <param name="backend">The backend the arithmetic runs on.</param>
        /// <returns>The predictions, of the outputs' shape; the outputs themselves for the <see cref="OutputActivation.Identity"/>.</returns>
        public Tensor Applied(Tensor outputs, ITensorBackend backend) => activation switch
        {
            OutputActivation.Softmax => backend.Exp(backend.LogSoftmax(outputs)),
            OutputActivation.Sigmoid => backend.Sigmoid(outputs),
            OutputActivation.Exp => backend.Exp(outputs),
            OutputActivation.Margin => backend.Positive(outputs),
            _ => outputs,
        };

        /// <summary>Whether a network layer is this activation, so that a network compiled with a loss that applies it ends before that layer.</summary>
        /// <param name="layer">The layer.</param>
        /// <returns>Whether it is a <see cref="Sigmoid"/> for the sigmoid, or a <see cref="Softmax"/> for the softmax.</returns>
        public bool Closes(Layer layer) =>
            (layer is Sigmoid && activation == OutputActivation.Sigmoid) || (layer is Softmax && activation == OutputActivation.Softmax);

        /// <summary>Whether the activation is one a network has a layer for: the sigmoid and the softmax. The others a loss applies and no layer is.</summary>
        /// <returns>Whether a layer of a network can be this activation.</returns>
        public bool HasALayer() => activation is OutputActivation.Sigmoid or OutputActivation.Softmax;

        /// <summary>Whether a foreign file's word for an activation names this one, as an activation a layer of a network can be.</summary>
        /// <param name="foreignWord">The word the file writes an activation with: Keras's <c>softmax</c>, say.</param>
        /// <returns>Whether the activation has a layer and the word is its own.</returns>
        public bool OwnedBy(string foreignWord) => activation.HasALayer() && activation.Word() == foreignWord;
    }
}
