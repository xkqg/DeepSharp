// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>A declared part as the form writes it into a step's JSON, and draws it from there.</summary>
internal static class PartDeclarationExtensions
{
    extension(PartDeclaration part)
    {
        /// <summary>The part as a step's JSON holds it: its name under 'kind', then every setting it holds, each as the value it is.</summary>
        /// <returns>The part, as JSON.</returns>
        public JsonObject AsJson()
        {
            var written = new JsonObject { ["kind"] = part.Kind };

            foreach (var setting in part.Settings)
            {
                written[setting.Key] = setting.Value.Holds switch
                {
                    PartValues.Number => JsonValue.Create(setting.Value.Number),
                    PartValues.YesOrNo => JsonValue.Create(setting.Value.YesOrNo),
                    _ => JsonValue.Create(setting.Value.Text),
                };
            }

            return written;
        }
    }

    extension(PartsParameter parameter)
    {
        /// <summary>The parts as a step's JSON holds them under the parameter's key: one part as itself, several as a list.</summary>
        /// <param name="parts">The parts.</param>
        /// <returns>The parts, as JSON.</returns>
        public JsonNode AsJson(IReadOnlyList<PartDeclaration> parts) =>
            parameter.Single ? parts[0].AsJson() : new JsonArray([.. parts.Select(part => (JsonNode?)part.AsJson())]);
    }
}
