// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>What an importer that builds the network from a file's description asks of the description it wrote.</summary>
public static class DescriptionLoweringExtensions
{
    extension(Sequential description)
    {
        /// <summary>The description lowered onto a stack of layers for examples of one shape, a refusal of it a refusal of the file.</summary>
        /// <param name="each">The shape of one example.</param>
        /// <param name="where">Where the file's description stands, as a refusal names it: <c>config.json</c>.</param>
        /// <returns>The stack, its numbers the starts of draws that the file's own numbers replace.</returns>
        /// <exception cref="FormatException">
        /// The layers the description names cannot be built one after another: the reason on one line, after where the description stands.
        /// </exception>
        public LayerStack LoweredFor(Shape each, string where)
        {
            ArgumentNullException.ThrowIfNull(where);

            try
            {
                return description.Lower(each, new RandomStream(0));
            }
            catch (ArgumentException refused)
            {
                throw new FormatException($"{where}: {refused.Message.ReplaceLineEndings(" ")}", refused);
            }
        }
    }
}
