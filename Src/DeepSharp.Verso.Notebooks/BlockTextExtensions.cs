// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// A step as the text of the block that holds it.
/// </summary>
/// <remarks>
/// The block's text is the step exactly as the step writes itself, laid out one key to a line so a person can read
/// and edit it, with the same line break on every system. Whatever writes a block — a new block, a gesture on the
/// grid, a field in the properties panel — writes it this way.
/// </remarks>
internal static class BlockTextExtensions
{
    private static readonly JsonWriterOptions Indented = new() { Indented = true };

    extension(CourseStep step)
    {
        /// <summary>The text a block holding this step of a course starts with.</summary>
        /// <param name="catalog">The verbs a notebook knows.</param>
        /// <returns>The step's skeleton, with what has been said put in it, laid out as every block is: indented.</returns>
        public string AsBlockText(StepCatalog catalog)
        {
            var buffer = new ArrayBufferWriter<byte>();

            using (var writer = new Utf8JsonWriter(buffer, Indented))
            using (var skeleton = JsonDocument.Parse(step.Skeleton(catalog)))
            {
                skeleton.RootElement.WriteTo(writer);
            }

            return Encoding.UTF8.GetString(buffer.WrittenSpan).ReplaceLineEndings("\n");
        }
    }

    extension(IPipelineStep step)
    {
        /// <summary>The text a block holding this step shows.</summary>
        /// <returns>The step's JSON, indented.</returns>
        public string AsBlockText()
        {
            var buffer = new ArrayBufferWriter<byte>();

            using (var writer = new Utf8JsonWriter(buffer, Indented))
            {
                step.WriteTo(writer);
            }

            // A writer ends its lines the way the machine does on some runtimes; a saved notebook is the same file everywhere.
            return Encoding.UTF8.GetString(buffer.WrittenSpan).ReplaceLineEndings("\n");
        }

        /// <summary>The step on one line, as a card quotes it among other words.</summary>
        /// <returns>The step's JSON, with nothing between its tokens.</returns>
        public string AsLine()
        {
            var buffer = new ArrayBufferWriter<byte>();

            using (var writer = new Utf8JsonWriter(buffer))
            {
                step.WriteTo(writer);
            }

            return Encoding.UTF8.GetString(buffer.WrittenSpan);
        }
    }
}
