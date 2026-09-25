// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Every step says once what its parameters are, and everything else is derived from that: how it is
/// written, how it is read back, what a file may say about it, and the form a notebook shows for it. Before
/// this the keys were typed by hand on both sides — sixty-seven written and sixty-five read — and a key the
/// reader did not know was read back as if it had not been written.
/// </summary>
public class StepParameterContractTests
{
    private static StepCatalog Everything() => StepCatalog.BuiltIn().WithIndicators();

    private static IEnumerable<Type> StepTypes() =>
        new[] { typeof(Pdd).Assembly, typeof(AddIndicatorStep).Assembly }
            .SelectMany(assembly => assembly.GetTypes())
            .Where(type => type.IsClass && !type.IsAbstract)
            .Where(type => type.GetInterfaces().Any(
                face => face.IsGenericType && face.GetGenericTypeDefinition() == typeof(IPipelineStep<>)));

    public static TheoryData<string> EveryVerb()
    {
        var verbs = new TheoryData<string>();

        foreach (var description in Everything().Descriptions)
        {
            verbs.Add(description.Verb);
        }

        return verbs;
    }

    [Fact]
    public void EveryStepTypeDescribesItsParameters_EachKeyOnce_AndNoneCalledStep()
    {
        var types = StepTypes().ToArray();
        var catalog = Everything();

        Assert.NotEmpty(types);

        foreach (var type in types)
        {
            var verb = (string)type.GetProperty("Name", BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy)!.GetValue(null)!;
            var description = catalog.Describe(verb);
            var keys = description.Parameters.SelectMany(parameter => parameter.Keys).ToArray();

            Assert.Equal(keys.Length, keys.Distinct(StringComparer.Ordinal).Count());
            Assert.DoesNotContain("step", keys);
            Assert.All(description.Parameters, parameter => Assert.False(string.IsNullOrWhiteSpace(parameter.Description)));
        }
    }

    [Fact]
    public void TheCatalogListsEveryVerbItKnows_WithItsParameters()
    {
        var catalog = Everything();

        Assert.Equal(
            catalog.Descriptions.Select(description => description.Verb).Order(StringComparer.Ordinal),
            StepTypes()
                .Select(type => (string)type.GetProperty("Name", BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy)!.GetValue(null)!)
                .Distinct(StringComparer.Ordinal)
                .Order(StringComparer.Ordinal));

        Assert.Equal("normalise", catalog.Describe("normalise").Verb);
        Assert.Contains("scale", catalog.Describe("normalise").Parameters.SelectMany(parameter => parameter.Keys));
    }

    [Theory]
    [MemberData(nameof(EveryVerb))]
    public void EveryStepReadsItsTemplateBack_AndWritesExactlyTheKeysItsParametersName(string verb)
    {
        var catalog = Everything();
        var description = catalog.Describe(verb);

        using var template = JsonDocument.Parse(description.Template);
        var step = catalog.Read(template.RootElement);

        var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            step.WriteTo(writer);
        }

        using var written = JsonDocument.Parse(buffer.ToArray());
        var keys = written.RootElement.EnumerateObject().Select(property => property.Name);

        Assert.Equal(verb, written.RootElement.GetProperty("step").GetString());

        // Every key it has to write, and nothing it does not take; a share a new block leaves out is not written.
        Assert.Subset(
            description.Parameters.SelectMany(parameter => parameter.Keys).Prepend("step").ToHashSet(StringComparer.Ordinal),
            keys.ToHashSet(StringComparer.Ordinal));
        Assert.Superset(
            description.Parameters.SelectMany(parameter => parameter.RequiredKeys).Prepend("step").ToHashSet(StringComparer.Ordinal),
            keys.ToHashSet(StringComparer.Ordinal));
        Assert.Equal(
            template.RootElement.EnumerateObject().Select(property => property.Name).Order(StringComparer.Ordinal),
            keys.Order(StringComparer.Ordinal));
    }

    [Theory]
    [MemberData(nameof(EveryVerb))]
    public void AKeyNoParameterNames_IsRefusedAndNamed(string verb)
    {
        var catalog = Everything();
        var template = catalog.Describe(verb).Template;
        var withColour = template[..template.LastIndexOf('}')] + ",\"colour\":\"red\"}";

        using var document = JsonDocument.Parse(withColour);

        var refused = Assert.Throws<FormatException>(() => catalog.Read(document.RootElement));

        Assert.Contains("'colour'", refused.Message, StringComparison.Ordinal);
        Assert.Contains(verb, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryStepType_IsTheStepItNamesAsItself()
    {
        // A step is written by its type's description with the step cast to that type. A type that named
        // another as itself would compile, and then fail the first time anything tried to write it.
        // A step type is itself, or one of the leaves of a step that has two forms and one verb.
        Assert.All(StepTypes(), type => Assert.True(
            type.GetInterfaces()
                .Single(face => face.IsGenericType && face.GetGenericTypeDefinition() == typeof(IPipelineStep<>))
                .GetGenericArguments()[0]
                .IsAssignableFrom(type)));
    }

    [Theory]
    [MemberData(nameof(EveryVerb))]
    public void EveryVerb_SaysWhatItDoes_InASentence(string verb)
    {
        var purpose = Everything().Describe(verb).Purpose;

        Assert.False(string.IsNullOrWhiteSpace(purpose));
        Assert.True(char.IsUpper(purpose[0]), $"'{verb}': '{purpose}' does not start as a sentence does.");
        Assert.EndsWith(".", purpose, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryVerb_SaysFromWhichVersionOfTheFileItMeansWhatItSays()
    {
        // A verb that is new, or whose meaning changed, is refused by name in a file written against an older
        // version: loading it would run something the person who wrote the file never saw.
        var since = Everything().Descriptions.ToDictionary(description => description.Verb, description => description.Since);

        // New in the second version: dropping columns, encoding every category, putting rows in order. Changed
        // in it: the three splits, which place a row by what it says rather than where it stands, and the
        // warm-up drop, which reads the declared order rather than the file's.
        Dictionary<string, int> newer = new(StringComparer.Ordinal)
        {
            ["drop.columns"] = 2,
            ["drop.gaps"] = 2,
            ["encode.categories"] = 2,
            ["evidence.correlation"] = 2,
            ["evidence.profile"] = 2,
            ["order.by"] = 2,
            ["split.atRandom"] = 2,
            ["split.byTime"] = 2,
            ["split.stratified"] = 2,
            ["drop.warmup"] = 2,
            ["target.distribution"] = 2,
            ["target.labels"] = 2,
            ["target.ahead"] = 2,
        };

        Assert.All(since, each => Assert.Equal(newer.GetValueOrDefault(each.Key, 1), each.Value));
    }

    [Theory]
    [MemberData(nameof(EveryVerb))]
    public void EveryParameter_SurvivesTheFileWithAValueOtherThanTheOneABlockStartsWith(string verb)
    {
        // The template alone would miss a reader that quietly falls back to a default: the default and the
        // example are usually the same value. So every parameter is written with each other value it can
        // hold, and at least one of them has to come back exactly as it was written.
        var catalog = Everything();
        var description = catalog.Describe(verb);

        foreach (var parameter in description.Parameters)
        {
            var kept = 0;

            foreach (var alternative in parameter.Accept(new OtherValues()))
            {
                var file = JsonNode.Parse(description.Template)!.AsObject();

                foreach (var (key, value) in alternative)
                {
                    file[key] = value?.DeepClone();
                }

                IPipelineStep step;

                try
                {
                    using var document = JsonDocument.Parse(file.ToJsonString());
                    step = catalog.Read(document.RootElement);
                }
                catch (FormatException refused) when (refused.InnerException is ArgumentException)
                {
                    // A rule between two parameters — a quantile bound further out than half, an indicator
                    // with the wrong number of columns — refuses this value beside the others as written.
                    continue;
                }

                var written = JsonNode.Parse(Written(step))!.AsObject();

                foreach (var (key, value) in alternative)
                {
                    Assert.Equal(value?.ToJsonString(), written[key]?.ToJsonString());
                }

                kept++;
            }

            Assert.True(kept > 0, $"'{verb}': no value of '{parameter.Key}' other than its example survives the file.");
        }
    }

    [Theory]
    // A result named by nothing but spaces is left to the step, which puts it back in the column it came
    // from: loading that and writing it back would say something the file did not.
    [InlineData("""{"step":"maths","column":"fare","maths":"log1p","into":"  "}""", "into")]
    public void AValueTheStepDoesNotKeep_IsRefused(string json, string key)
    {
        using var document = JsonDocument.Parse(json);

        var refused = Assert.Throws<FormatException>(() => Everything().Read(document.RootElement));

        Assert.Contains($"'{key}'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("fare", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("""{"step":"fill.missing","column":"age","with":{"kind":"constant","value":-1,"colour":"red"}}""")]
    [InlineData("""{"step":"declare","remainder":"drop","columns":[{"name":"a","kind":"number","optional":false,"colour":"red"}]}""")]
    public void ANestedUnknownKey_IsRefused(string json)
    {
        using var document = JsonDocument.Parse(json);

        var refused = Assert.Throws<FormatException>(() => Everything().Read(document.RootElement));

        Assert.Contains("'colour'", refused.Message, StringComparison.Ordinal);
    }

    private static string Written(IPipelineStep step)
    {
        var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            step.WriteTo(writer);
        }

        return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
    }

    /// <summary>For each kind of parameter, the values it can hold other than the one a new block starts with.</summary>
    private sealed class OtherValues : IStepParameterVisitor<IEnumerable<Dictionary<string, JsonNode?>>>
    {
        private static IEnumerable<Dictionary<string, JsonNode?>> One(string key, JsonNode? value) => [new() { [key] = value }];

        public IEnumerable<Dictionary<string, JsonNode?>> Visit(TextParameter parameter) => One(parameter.Key, "other words");

        public IEnumerable<Dictionary<string, JsonNode?>> Visit(FilePathParameter parameter) => One(parameter.Key, "elsewhere.csv");

        public IEnumerable<Dictionary<string, JsonNode?>> Visit(ColumnParameter parameter) => One(parameter.Key, "other");

        public IEnumerable<Dictionary<string, JsonNode?>> Visit(NewColumnParameter parameter) => One(parameter.Key, "made");

        public IEnumerable<Dictionary<string, JsonNode?>> Visit(ColumnsParameter parameter) =>
            One(parameter.Key, new JsonArray([.. parameter.Example.DefaultIfEmpty("column").Select((_, at) => (JsonNode?)$"other{at}")]));

        public IEnumerable<Dictionary<string, JsonNode?>> Visit(NumberParameter parameter) =>
            [.. new[] { parameter.Example * 2, parameter.Example + 1, parameter.Example / 2 }.SelectMany(value => One(parameter.Key, value))];

        public IEnumerable<Dictionary<string, JsonNode?>> Visit(WholeNumberParameter parameter) =>
            [.. new[] { parameter.Example + 1, parameter.Example + 2 }.SelectMany(value => One(parameter.Key, value))];

        public IEnumerable<Dictionary<string, JsonNode?>> Visit(TrueOrFalseParameter parameter) => One(parameter.Key, !parameter.Example);

        public IEnumerable<Dictionary<string, JsonNode?>> Visit(ShareParameter parameter) =>
            [.. new[] { 0.25, 0.5 }.SelectMany(value => One(parameter.Key, value))];

        public IEnumerable<Dictionary<string, JsonNode?>> Visit<TEnum>(OneOfParameter<TEnum> parameter)
            where TEnum : struct, Enum =>
            [.. parameter.Choices
                .Where(choice => choice != parameter.Example.ToString().ToLowerInvariant())
                .SelectMany(choice => One(parameter.Key, choice))];

        public IEnumerable<Dictionary<string, JsonNode?>> Visit<TEnum>(SeveralOfParameter<TEnum> parameter)
            where TEnum : struct, Enum =>
            [.. parameter.Choices
                .Where(choice => !parameter.Example.Select(each => each.ToString().ToLowerInvariant()).SequenceEqual([choice]))
                .SelectMany(choice => One(parameter.Key, new JsonArray(choice)))];

        public IEnumerable<Dictionary<string, JsonNode?>> Visit(FillStrategyParameter parameter) =>
            [.. parameter.Allowed
                .Where(name => name != parameter.Example.Name)
                .SelectMany(name => One(
                    parameter.Key,
                    With.TakesAValue(name) ? new JsonObject { ["kind"] = name, ["value"] = 2.5 } : name))];

        public IEnumerable<Dictionary<string, JsonNode?>> Visit(SplitSharesParameter parameter) =>
        [
            new() { ["train"] = 0.6, ["validation"] = 0.2, ["test"] = 0.2, ["predict"] = 0.0 },
            new() { ["train"] = 0.5, ["validation"] = 0.2, ["test"] = 0.2, ["predict"] = 0.1 },
        ];

        public IEnumerable<Dictionary<string, JsonNode?>> Visit(ColumnDeclarationsParameter parameter) =>
            One(parameter.Key, new JsonArray(new JsonObject { ["name"] = "other", ["kind"] = "text", ["optional"] = true }));
    }

    [Fact]
    public void AVerbTheCatalogDoesNotKnow_HasNoDescription()
    {
        Assert.Throws<NotSupportedException>(() => StepCatalog.BuiltIn().Describe("feature.indicator"));
        Assert.Throws<ArgumentException>(() => StepCatalog.BuiltIn().Describe(" "));
    }
}
