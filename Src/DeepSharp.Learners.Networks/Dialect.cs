// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Learners.Networks;

/// <summary>A setting PyTorch and Keras each leave unsaid, at a value of their own.</summary>
/// <typeparam name="T">What the setting holds: a number, or whether something is so.</typeparam>
/// <param name="Torch">What PyTorch leaves there.</param>
/// <param name="Keras">What TensorFlow and Keras leave there.</param>
/// <remarks>One row of the table a declaration's doors read what a caller leaves unsaid from, so no door decides it for itself.</remarks>
internal readonly record struct Dialect<T>(T Torch, T Keras);

/// <summary>What a vocabulary decides: which library's number fills in what a caller leaves unsaid, and what it cannot say.</summary>
internal static class VocabulariesExtensions
{
    extension(Vocabularies words)
    {
        /// <summary>What this vocabulary's library leaves unsaid of a setting the two leave at values of their own.</summary>
        /// <typeparam name="T">What the setting holds.</typeparam>
        /// <param name="setting">The setting, with both libraries' values.</param>
        /// <returns>The value this vocabulary's library leaves there.</returns>
        public T Unsaid<T>(Dialect<T> setting) => words == Vocabularies.Keras ? setting.Keras : setting.Torch;

        /// <summary>Refuses, in Keras's words, what this library runs only as PyTorch does, naming the door in PyTorch's words that says it.</summary>
        /// <param name="door">The door, as a declaration in PyTorch's words calls it.</param>
        /// <param name="why">Why Keras's words cannot say it.</param>
        /// <exception cref="NotSupportedException">The network is declared in Keras's words.</exception>
        public void RequirePyTorchs(string door, string why)
        {
            if (words == Vocabularies.Keras)
            {
                throw new NotSupportedException(
                    $"{why}, so a network declared in Keras's words cannot say it: declare it in PyTorch's, .WithTorch(network => network.{door}(…)).");
            }
        }
    }
}
