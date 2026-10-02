// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace DeepSharp.Pipelines;

/// <summary>
/// The JSON Schema of a pipeline file, written from the descriptions of the verbs it may hold.
/// </summary>
/// <remarks>
/// A projection of the steps, never a second description of them: every key, kind and word in it comes from
/// the same parameters the reader checks a file against. It says the shape of a file and nothing more — which
/// shares make a whole and where a step may stand are the reader's to say, because no schema can.
/// </remarks>
internal static class PipelineFileSchema
{
    /// <summary>Where the schema of this library's own verbs is published.</summary>
    public const string Address = "https://raw.githubusercontent.com/xkqg/DeepSharp/main/pipeline.schema.json";

    private static readonly JsonSerializerOptions Indented = new()
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary>Writes the schema of a file that may hold these verbs.</summary>
    /// <param name="verbs">What each verb is.</param>
    /// <returns>The schema, as an indented JSON document ending in a line break.</returns>
    public static string Write(IReadOnlyList<StepDescription> verbs)
    {
        var definitions = new JsonObject
        {
            ["step"] = new JsonObject
            {
                ["description"] = "One step of the pipeline, named by its verb.",
                ["type"] = "object",
                ["properties"] = new JsonObject
                {
                    ["step"] = new JsonObject { ["enum"] = new JsonArray([.. verbs.Select(verb => (JsonNode)verb.Verb)]) },
                },
                ["required"] = new JsonArray("step"),
                ["allOf"] = new JsonArray([.. verbs.Select(verb => (JsonNode)new JsonObject
                {
                    ["if"] = new JsonObject
                    {
                        ["properties"] = new JsonObject { ["step"] = new JsonObject { ["const"] = verb.Verb } },
                        ["required"] = new JsonArray("step"),
                    },
                    ["then"] = new JsonObject { ["$ref"] = $"#/$defs/{verb.Verb}" },
                })]),
            },
            ["fitted"] = new JsonObject
            {
                ["description"] = "What each step learned from the training rows, written by the fit rather than by a person: "
                                  + "an entry for each step that learned something, filed under the key of the steps it learned behind.",
                ["type"] = "array",
                ["items"] = new JsonObject
                {
                    ["type"] = "object",
                    ["properties"] = new JsonObject
                    {
                        ["step"] = AName(),
                        ["prefix"] = Prefix("The key of the steps up to the one that learned this: a SHA-256 digest, in hexadecimal."),
                        ["learned"] = new JsonObject
                        {
                            ["type"] = "object",
                            ["additionalProperties"] = new JsonObject
                            {
                                ["anyOf"] = new JsonArray(
                                    new JsonObject { ["type"] = "number" },
                                    new JsonObject { ["type"] = "string" },
                                    new JsonObject { ["type"] = "array", ["items"] = new JsonObject { ["type"] = "number" } },
                                    new JsonObject { ["type"] = "array", ["items"] = new JsonObject { ["type"] = "string" } }),
                            },
                        },
                    },
                    ["required"] = new JsonArray("step", "prefix", "learned"),
                    ["additionalProperties"] = false,
                },
            },
            ["skipped"] = new JsonObject
            {
                ["description"] = "The steps a run for a learner left out, written by the run rather than by a person: an entry for "
                                  + "each, filed under the key of the steps up to it; written only when a run left one out.",
                ["type"] = "array",
                ["items"] = new JsonObject
                {
                    ["type"] = "object",
                    ["properties"] = new JsonObject
                    {
                        ["step"] = AName(),
                        ["prefix"] = Prefix("The key of the steps up to the one left out: a SHA-256 digest, in hexadecimal."),
                    },
                    ["required"] = new JsonArray("step", "prefix"),
                    ["additionalProperties"] = false,
                },
            },
        };

        foreach (var verb in verbs)
        {
            definitions[verb.Verb] = Step(verb);
        }

        var schema = new JsonObject
        {
            ["$schema"] = "https://json-schema.org/draft/2020-12/schema",
            ["$id"] = Address,
            ["title"] = "A DeepSharp pipeline",
            ["description"] = "The steps of a pipeline in the order they were written, and what the fit learned from the training rows.",
            ["type"] = "object",
            ["properties"] = new JsonObject
            {
                ["version"] = new JsonObject
                {
                    ["description"] = "The version of the pipeline file this was written against; the first, when it is left out.",
                    ["type"] = "integer",
                    ["minimum"] = 1,
                    ["maximum"] = PipelineDeclaration.Version,
                },
                ["declaration"] = new JsonObject
                {
                    ["description"] = "The steps, in the order they run.",
                    ["type"] = "array",
                    ["items"] = new JsonObject { ["$ref"] = "#/$defs/step" },
                },
                ["fitted"] = new JsonObject { ["$ref"] = "#/$defs/fitted" },
                ["skipped"] = new JsonObject { ["$ref"] = "#/$defs/skipped" },
            },
            ["required"] = new JsonArray("declaration"),
            ["additionalProperties"] = false,
            ["$defs"] = definitions,
        };

        // What a run left out is written from the fourth version, and a file written against an earlier one says nothing of
        // it. A verb that came to mean something else in a version of the file is not read from a file written against an
        // earlier one either. "version" is left out of a file from before it was written, which the properties below leave
        // unchecked, so a file without one is held to the first version, as it is read.
        var since = verbs.Select(verb => verb.Since).Where(version => version > 1).Distinct().Order().ToArray();

        schema["allOf"] = new JsonArray(
        [
            new JsonObject
            {
                ["if"] = new JsonObject { ["properties"] = new JsonObject { ["version"] = new JsonObject { ["exclusiveMaximum"] = PipelineDocument.SkippedSince } } },
                ["then"] = new JsonObject { ["not"] = new JsonObject { ["required"] = new JsonArray("skipped") } },
            },
            .. since.Select(version => (JsonNode)new JsonObject
            {
                ["if"] = new JsonObject
                {
                    ["properties"] = new JsonObject { ["version"] = new JsonObject { ["exclusiveMaximum"] = version } },
                },
                ["then"] = new JsonObject
                {
                    ["properties"] = new JsonObject
                    {
                        ["declaration"] = new JsonObject
                        {
                            ["items"] = new JsonObject
                            {
                                ["properties"] = new JsonObject
                                {
                                    ["step"] = new JsonObject
                                    {
                                        ["not"] = new JsonObject
                                        {
                                            ["enum"] = new JsonArray([.. verbs
                                                .Where(verb => verb.Since >= version)
                                                .Select(verb => (JsonNode)verb.Verb)]),
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            }),
        ]);

        // A serializer ends its lines the way the machine does on some runtimes; the schema is the same text everywhere.
        return schema.ToJsonString(Indented).ReplaceLineEndings("\n") + "\n";
    }

    // More than spaces: the rule every name and every word a step holds is kept to.
    private static JsonObject AName() => new() { ["type"] = "string", ["pattern"] = "\\S" };

    // The key of the steps up to one: a SHA-256 digest, in lowercase hexadecimal.
    private static JsonObject Prefix(string description) => new()
    {
        ["description"] = description,
        ["type"] = "string",
        ["pattern"] = "^[0-9a-f]{64}$",
    };

    private static JsonObject Step(StepDescription verb)
    {
        var properties = new JsonObject { ["step"] = new JsonObject { ["const"] = verb.Verb } };
        var required = new JsonArray("step");

        foreach (var parameter in verb.Parameters)
        {
            foreach (var property in parameter.Accept(new SchemaOfAParameter()))
            {
                properties[property.Key] = property.Schema;
            }

            foreach (var key in parameter.RequiredKeys)
            {
                required.Add(key);
            }
        }

        return new JsonObject
        {
            ["description"] = verb.Purpose,
            ["type"] = "object",
            ["properties"] = properties,
            ["required"] = required,
            ["additionalProperties"] = false,
        };
    }

    /// <summary>One key of a step, and the schema of what it holds.</summary>
    /// <param name="Key">The key.</param>
    /// <param name="Schema">What a file may write under it.</param>
    private readonly record struct PropertySchema(string Key, JsonObject Schema);

    /// <summary>The schema of each kind, key by key.</summary>
    private sealed class SchemaOfAParameter : IStepParameterVisitor<IReadOnlyList<PropertySchema>>
    {
        public IReadOnlyList<PropertySchema> Visit(TextParameter parameter) => [Words(parameter)];

        public IReadOnlyList<PropertySchema> Visit(FilePathParameter parameter) => [Words(parameter)];

        public IReadOnlyList<PropertySchema> Visit(ColumnParameter parameter) => [Words(parameter)];

        public IReadOnlyList<PropertySchema> Visit(NewColumnParameter parameter) => [Words(parameter)];

        public IReadOnlyList<PropertySchema> Visit(ColumnsParameter parameter)
        {
            var columns = new JsonObject { ["description"] = parameter.Description, ["type"] = "array" };

            // None named is the key left out, and an empty list reads the same way.
            if (!parameter.Optional)
            {
                columns["minItems"] = 1;
            }

            // As the reader: a set names each column once, and only a list of roles may name one again.
            if (!parameter.Repeatable)
            {
                columns["uniqueItems"] = true;
            }

            columns["items"] = AName();

            return [new(parameter.Key, columns)];
        }

        public IReadOnlyList<PropertySchema> Visit(NumberParameter parameter)
        {
            var number = new JsonObject { ["description"] = parameter.Description, ["type"] = "number" };

            if (parameter.Above is { } above)
            {
                number["exclusiveMinimum"] = above;
            }
            else
            {
                number["minimum"] = parameter.AtLeast ?? -double.MaxValue;
            }

            number["maximum"] = double.MaxValue;

            return [new(parameter.Key, number)];
        }

        public IReadOnlyList<PropertySchema> Visit(WholeNumberParameter parameter) =>
        [
            new(parameter.Key, new JsonObject
            {
                ["description"] = parameter.Description,
                ["type"] = "integer",
                ["minimum"] = parameter.AtLeast ?? int.MinValue,
                ["maximum"] = int.MaxValue,
            }),
        ];

        public IReadOnlyList<PropertySchema> Visit(TrueOrFalseParameter parameter) =>
            [new(parameter.Key, new JsonObject { ["description"] = parameter.Description, ["type"] = "boolean" })];

        public IReadOnlyList<PropertySchema> Visit(ShareParameter parameter) =>
        [
            new(parameter.Key, new JsonObject
            {
                ["description"] = parameter.Description,
                ["type"] = "number",
                ["minimum"] = 0,
                ["maximum"] = 1,
            }),
        ];

        public IReadOnlyList<PropertySchema> Visit<TEnum>(OneOfParameter<TEnum> parameter)
            where TEnum : struct, Enum
        {
            var word = AWord(parameter.Choices);
            word["description"] = parameter.Description;

            return [new(parameter.Key, word)];
        }

        public IReadOnlyList<PropertySchema> Visit<TEnum>(SeveralOfParameter<TEnum> parameter)
            where TEnum : struct, Enum =>
        [
            new(parameter.Key, new JsonObject
            {
                ["description"] = parameter.Description,
                ["type"] = "array",
                ["minItems"] = 1,
                ["items"] = AWord(parameter.Choices),
            }),
        ];

        public IReadOnlyList<PropertySchema> Visit(FillStrategyParameter parameter)
        {
            var ways = new JsonArray();
            var named = parameter.Allowed.Where(name => !With.TakesAValue(name)).ToArray();
            var numbered = parameter.Allowed.Where(With.TakesAValue).ToArray();

            if (named.Length > 0)
            {
                ways.Add(new JsonObject { ["enum"] = new JsonArray([.. named.Select(name => (JsonNode)name)]) });
            }

            if (numbered.Length > 0)
            {
                ways.Add(new JsonObject
                {
                    ["type"] = "object",
                    ["properties"] = new JsonObject
                    {
                        [FillStrategyParameter.KindKey] = new JsonObject { ["enum"] = new JsonArray([.. numbered.Select(name => (JsonNode)name)]) },
                        [FillStrategyParameter.ValueKey] = new JsonObject
                        {
                            ["type"] = "number",
                            ["minimum"] = -double.MaxValue,
                            ["maximum"] = double.MaxValue,
                        },
                    },
                    ["required"] = new JsonArray([.. parameter.StrategyKeys.Select(key => (JsonNode)key)]),
                    ["additionalProperties"] = false,
                });
            }

            return [new(parameter.Key, new JsonObject { ["description"] = parameter.Description, ["anyOf"] = ways })];
        }

        public IReadOnlyList<PropertySchema> Visit(SplitSharesParameter parameter) =>
        [
            new("train", AShare(parameter.Description, orNothing: false)),
            new("validation", AShare("The share used while choosing between models.", orNothing: true)),
            new("test", AShare("The share kept back until the end.", orNothing: false)),
            new("predict", AShare("The share held back to predict on; none when it is left out.", orNothing: true)),
        ];

        public IReadOnlyList<PropertySchema> Visit(ColumnDeclarationsParameter parameter)
        {
            var column = new JsonObject();

            foreach (var property in parameter.Parts.SelectMany(part => part.Parameter.Accept(this)))
            {
                column[property.Key] = property.Schema;
            }

            // As the reader: a category was another kind before it became one, never a category.
            column[parameter.Was.Key]!["not"] = AWord([Vocabulary<ColumnKind>.WordFor(ColumnKind.Category, parameter.Was.Key)]);

            return
            [
                new(parameter.Key, new JsonObject
                {
                    ["description"] = parameter.Description,
                    ["type"] = "array",
                    ["minItems"] = 1,
                    ["items"] = new JsonObject
                    {
                        ["type"] = "object",
                        ["properties"] = column,
                        ["required"] = new JsonArray([.. parameter.RequiredColumnKeys.Select(key => (JsonNode)key)]),
                        ["additionalProperties"] = false,
                        ["allOf"] = new JsonArray([.. parameter.Parts.Where(part => part.Only is not null).Select(part => OnlyOn(parameter.Kind, part))]),
                    },
                }),
            ];
        }

        public IReadOnlyList<PropertySchema> Visit(PartsParameter parameter)
        {
            var part = new JsonObject
            {
                ["type"] = "object",
                ["oneOf"] = new JsonArray([.. parameter.Kinds.Select(OfTheName)]),
            };

            return
            [
                new(parameter.Key, parameter.Single
                    ? Described(part, parameter.Description)
                    : Described(
                        new JsonObject { ["type"] = "array", ["minItems"] = 1, ["items"] = part },
                        parameter.Description)),
            ];
        }

        // One name a part may give: its own word under 'kind', every setting it takes, and nothing else.
        private JsonObject OfTheName(PartKind kind)
        {
            var settings = new JsonObject { ["kind"] = AWord([kind.Name]) };

            foreach (var property in kind.Settings.SelectMany(setting => setting.Accept(this)))
            {
                settings[property.Key] = property.Schema;
            }

            return new JsonObject
            {
                ["description"] = kind.Purpose,
                ["properties"] = settings,
                ["required"] = new JsonArray([.. settings.Select(property => (JsonNode)property.Key)]),
                ["additionalProperties"] = false,
            };
        }

        private static JsonObject Described(JsonObject schema, string description)
        {
            schema["description"] = description;

            return schema;
        }

        // A part only one kind of column is written with, refused on a column of any other kind, as the reader refuses it.
        private static JsonNode OnlyOn(OneOfParameter<ColumnKind> kind, ColumnPart part) => new JsonObject
        {
            ["if"] = new JsonObject
            {
                ["properties"] = new JsonObject { [kind.Key] = AWord([Vocabulary<ColumnKind>.WordFor(part.Only!.Value, kind.Key)]) },
            },
            ["else"] = new JsonObject { ["not"] = new JsonObject { ["required"] = new JsonArray(part.Parameter.Key) } },
        };

        private static PropertySchema Words(StepParameter parameter)
        {
            var name = AName();
            name["description"] = parameter.Description;

            return new(parameter.Key, name);
        }

        // One of the words, in any case, as the reader takes it: the list first, so an editor offers it, and
        // the same words letter by letter in either case, so a word typed in capitals is not marked wrong.
        private static JsonObject AWord(IReadOnlyList<string> words) => new()
        {
            ["anyOf"] = new JsonArray(
                new JsonObject { ["enum"] = new JsonArray([.. words.Select(word => (JsonNode)word)]) },
                new JsonObject
                {
                    ["type"] = "string",
                    ["pattern"] = $"^(?:{string.Join('|', words.Select(InEitherCase))})$",
                }),
        };

        private static string InEitherCase(string word)
        {
            var pattern = new StringBuilder();

            foreach (var character in word)
            {
                pattern.Append(char.IsLetter(character)
                    ? $"[{char.ToUpperInvariant(character)}{char.ToLowerInvariant(character)}]"
                    : Regex.Escape(character.ToString()));
            }

            return pattern.ToString();
        }

        private static JsonObject AShare(string description, bool orNothing)
        {
            var share = new JsonObject { ["description"] = description, ["type"] = "number" };

            if (orNothing)
            {
                share["minimum"] = 0;
            }
            else
            {
                share["exclusiveMinimum"] = 0;
            }

            share["maximum"] = 1;

            return share;
        }
    }
}
